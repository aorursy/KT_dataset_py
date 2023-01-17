import os
import gc
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

data_path = '../input/NFL-Punt-Analytics-Competition'
images_path = '../input/images'
# load data
player_role_data = pd.read_csv(data_path+'/play_player_role_data.csv', index_col=['GameKey', 'PlayID'])
play_information_data = pd.read_csv(data_path+'/play_information.csv', index_col=['GameKey', 'PlayID'])
concussion_data = pd.read_csv(data_path+'/video_review.csv', index_col=['GameKey', 'PlayID'])
concussion_data = concussion_data.replace({'Primary_Partner_GSISID': {'Unclear': np.nan}})
concussion_data.GSISID = concussion_data.GSISID.astype(float)
concussion_data.Primary_Partner_GSISID = concussion_data.Primary_Partner_GSISID.astype(float)

# get the punt-specific roles of both players involved in the concussion
concussion_data = concussion_data.merge(player_role_data[['Role', 'GSISID']], left_on=['GameKey', 'PlayID', 'GSISID'], right_on=['GameKey', 'PlayID', 'GSISID'])
concussion_data = concussion_data.rename(columns={'Role': 'P1'})
concussion_data = concussion_data.merge(player_role_data[['Role', 'GSISID']], left_on=['GameKey', 'PlayID', 'Primary_Partner_GSISID'], right_on=['GameKey', 'PlayID', 'GSISID'])
concussion_data = concussion_data.rename(columns={'Role': 'P2'})
# Offense and defensive positions
offense = ['PLS', 'P', 'GL', 'PLW', 'PRW', 'PRT', 'GR', 'PLT', 'PLG', 'PRG', 'PC', 'PPR', 'PPL',
           'PPLi', 'GLo', 'GRi', 'GRo', 'PPLo', 'GLi', 'PPRi', 'PPRo', 'PLM1']
defense = ['PDR2', 'PLR', 'PLR2', 'PDR4', 'PLL', 'PDL5', 'PDL4', 'VLi', 'PR', 'VLo', 'PLL2', 'PDL3',
           'PLL1', 'VR', 'VRo', 'VRi', 'PDL1', 'PDR1', 'PDL2', 'PDR3', 'VL', 'PLM', 'PFB', 
           'PDR5', 'PLR1', 'PDL6', 'PLL3', 'PLR3', 'PDR6', 'PDM']
# load ngs data for concussions
# files = os.listdir(data_path)
# ngs_files = [x for x in files if 'NGS' in x]
# ngs_raw = None
# for idx, file in enumerate(ngs_files):
#     print(file)
#     df = pd.read_csv(os.path.join(data_path, file),
#                     low_memory=False)
#     df = df.set_index(['GameKey', 'PlayID'])
#     df = df.loc[df.index.isin(concussion_data.index)]
#     ngs_raw = df if ngs_raw is None else ngs_raw.append(df)
#     del df
#     gc.collect()
# ngs_raw['GSISID'] = ngs_raw['GSISID'].astype(float)
# ngs_raw = ngs_raw.merge(player_role_data[['Role', 'GSISID']], left_on=['GameKey', 'PlayID', 'GSISID'], right_on=['GameKey', 'PlayID', 'GSISID'])
# Get breakdown of concussion data by player activity
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(221)
con_by_activity = concussion_data.Player_Activity_Derived.value_counts()
plt.bar(con_by_activity.index, con_by_activity.values)
plt.title('Concussion by Player Activity')

# Get breakdown of concussion data by impact type
ax = fig.add_subplot(222)
con_by_impact = concussion_data.Primary_Impact_Type.value_counts()
plt.bar(con_by_impact.index, con_by_impact.values)
plt.title('Concussion by Impact Type')

# Get breakdown of player positions
# getting concussions and giving concussions
ax = fig.add_subplot(223)
con_by_pos = concussion_data.P1.value_counts()
plt.bar(con_by_pos.index, con_by_pos.values)
plt.xticks(rotation=45)
plt.title('Players Receiving Concussions by Position')

ax = fig.add_subplot(224)
give_con_by_pos = concussion_data.P2.value_counts()
plt.bar(give_con_by_pos.index, give_con_by_pos.values)
plt.xticks(rotation=45)
plt.title('Players Giving Concussions by Position')
plt.show()


con_by_pos = concussion_data.P1.value_counts()
give_con_by_pos = concussion_data.P2.value_counts()
involved = con_by_pos.add(give_con_by_pos, fill_value=0).sort_values(ascending=False)
plt.bar(involved.index, involved.values)
plt.xticks(rotation=45)
plt.title('Players Involved in Concussions by Position')
plt.show()

trench = ['PRG', 'PDR1', 'PRT', 'PLT', 'PLW', 'PLG', 'PLS', 'PRW', 'PPR', 'PDL1', 'PDR2', 'PDL2', 'PDR3', 'PLL', 'PLL1']
skilled = ['PR', 'GR', 'GL', 'P', 'VR', 'PFB', 'VLo']
involved_trench = involved.loc[involved.index.isin(trench)].sum()
involved_skilled = involved.loc[involved.index.isin(skilled)].sum()
skill_vs_unskill = pd.Series(index=['Skilled', 'Unskilled'], data=[[involved_skilled], [involved_trench]])
plt.bar(skill_vs_unskill.index, skill_vs_unskill.values)
plt.xticks(rotation=45)
plt.title('Players Involved in Concussions by Position Type')
plt.show()
# Create pivot table of formations
formations = pd.pivot_table(player_role_data,index=['GameKey', 'PlayID'],columns=['Role'], aggfunc=lambda x: len(x.unique()))['GSISID'].fillna(0)
merged_data = pd.merge(formations, play_information_data, left_index=True, right_index=True)
merged_data.loc[merged_data.index.isin(concussion_data.index), 'concussed'] = 1
merged_data.concussed.fillna(0, inplace=True)
# Remove punt plays that are blocked or killed by penalties
yards_list = []
for i,yards in enumerate(merged_data.PlayDescription.str.split(' yard').str[0].str[-2:]):
    try:
        yards_list.append(float(yards))
    except ValueError:
        yards_list.append(-500)
merged_data['punt_yards'] = yards_list
merged_data['no_play'] = merged_data.PlayDescription.str.contains('No Play', regex=True)
merged_data['blocked'] = merged_data.PlayDescription.str.contains('BLOCKED', regex=True)
merged_data = merged_data[(merged_data.punt_yards != -500) & (merged_data.no_play == False) & (merged_data.blocked == False)]
X = (merged_data[defense].values).astype(None)
y = merged_data['concussed'].values
# Cluster analysis
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering
from sklearn.decomposition import PCA
n_clusters = 5
pca = PCA(n_components=n_clusters).fit(X) # use pca with kmeans because data are dichotomous 
model = KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1)
# Histograms of clusters
labels = model.fit_predict(X)
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(131)
n1 = plt.hist(labels,bins=range(0,len(np.unique(labels))+1))
plt.title('All Plays')
plt.xlabel('Cluster #')
plt.ylabel('# plays in cluster')
ax = fig.add_subplot(132)
n2 = plt.hist(labels[y==1],bins=range(0,len(np.unique(labels))+1))
plt.title('Concussion Plays')
plt.xlabel('Cluster #')
plt.ylabel('# plays in cluster')
ax = fig.add_subplot(133)
plt.bar(np.unique(labels),100*(n2[0]/n1[0]))
plt.xlabel('Cluster #')
plt.ylabel('% concussion plays')
plt.title('% Concussion Plays')
idxMax = np.argmax(100*(n2[0]/n1[0])) # index of cluster w/ highest % concussions

# Image of raw concussion play formations
ticks = np.arange(0,X.shape[1],1)
fig = plt.figure(figsize=(20, 9))
ax = fig.add_subplot(121)
ax.matshow(X[y==1],aspect='auto')
ax.set_xticks(ticks)
ax.set_xticklabels(defense,rotation=45)
plt.ylabel('Concussion Play')
plt.title('All Concussion Plays')

ax = fig.add_subplot(122)
cax = ax.matshow(X[(y==1) & (labels==idxMax)],aspect='auto')
fig.colorbar(cax)
ax.set_xticks(ticks)
ax.set_xticklabels(defense,rotation=45)
plt.ylabel('Concussion Play')
plt.title('Concussion Plays in Cluster %d' %idxMax)
plt.show()

# Bar plot of differences between mean of cluster idxMax and the rest
# For statistical significance, run t-tests? ANOVAs? Need to think about this more...
fig = plt.figure()
ax = fig.add_subplot(111)
bar_width = 0.2
cnt = 0
for i in range(0,5):
    if i != idxMax:
        plt.bar(np.arange(0,X.shape[1])+bar_width*(cnt-1),np.mean(X[labels==idxMax],axis=0)-np.mean(X[labels==i],axis=0),bar_width)
        cnt += 1
ax.set_xticks(ticks)
ax.set_xticklabels(defense,rotation=45)
plt.legend(['Cluster 1', 'Cluster 2','Cluster 3','Cluster 4'])
ax.set_title(r'Comparision: $\mu_{C0}$ - $\mu_{C1, C2, C3, or C4}$')
# gets all high-speed collisions to use for further analysis
# all_collisions = None
# coll_data = ngs_raw.reset_index()
# coll_data.Time = coll_data.Time.astype(np.datetime64)
# coll_data = coll_data.pivot_table(values=['dis', 'x', 'y'], index=['GameKey', 'PlayID', 'Time'], columns='Role').sort_index()
# # Converts distance to velocity in mph, courtesy of mtodisco10
# coll_data.dis = coll_data.dis.mul(20.455)
# for player in coll_data.x.columns:
# #     print(player)
#     player_dist = coll_data.x.sub(coll_data.x[player], axis=0).pow(2).add(coll_data.y.sub(coll_data.y[player], axis=0).pow(2)).pow(.5)
#     player_dist = player_dist[offense] if player in defense else player_dist[defense]
#     player_dist = player_dist[(player_dist < .5)].dropna(how='all', axis=0).stack().to_frame()
#     player_dist.columns = ['DIST']
#     player_vel = coll_data.dis
#     this_player_vel = player_vel.loc[:, player]
#     this_player_vel.columns = ['V2']
#     this_player_vel_change = this_player_vel.groupby(level=['GameKey', 'PlayID']).pct_change()
#     player_vel = player_vel[(player_vel > 7)].dropna(how='all', axis=0)
#     player_vel = player_vel[(this_player_vel_change < -.20)].stack().to_frame()
#     player_vel.columns = ['V1']
#     collisions = player_dist.merge(player_vel, left_index=True, right_index=True).reset_index(level='Role')[['Role', 'V1']]
#     collisions.columns = ['P1', 'V1']
#     collisions['V2'] = this_player_vel
#     collisions['P2'] = player
#     collisions.P1 = collisions.P1.astype(str)
#     collisions = collisions.loc[collisions.P1 != player]
#     collisions = collisions.reset_index().groupby(by=['GameKey', 'PlayID', 'P1']).first()
#     collisions = collisions.reset_index(['P1'])
#     all_collisions = collisions if all_collisions is None else all_collisions.append(collisions)
# all_collisions.to_csv('collisions.csv')
# Collision data, I changed the 7mph threshold to 10mph. If a play contains a
# high speed impact, it is assigned a label of "1"
collision_data = pd.read_csv('../input/collisions/collisions.csv', index_col=['GameKey', 'PlayID'])
collision_data_thresh = collision_data[collision_data['V1'] > 10]
# collision_idx = np.unique(collision_data_thresh[['GameKey','PlayID']],axis=0)
# merged_data = merged_data.reset_index()
# merged_data['collided'] = 0
# for i in collision_idx:
#     if sum((merged_data['GameKey']==i[0]) & (merged_data['PlayID']==i[1]))==1:
#         idx = merged_data[(merged_data['GameKey']==i[0]) & (merged_data['PlayID']==i[1])].index[0]
#         merged_data['collided'][idx] = 1
merged_data.loc[merged_data.index.isin(collision_data.index), 'collided'] = 1
merged_data.loc[~merged_data.index.isin(collision_data.index), 'collided'] = 0

# How many of the high speed impacts are in cluster idxMax
collisions = merged_data['collided']
fig = plt.figure()
ax = fig.add_subplot(121)
n3 = plt.hist(labels[collisions==1],bins=range(0,len(np.unique(labels))+1))
plt.title('High Speed Plays')
plt.xlabel('Cluster #')
plt.ylabel('# plays in cluster')
           
ax = fig.add_subplot(122)
plt.bar(np.unique(labels),100*(n3[0]/n1[0]))
plt.xlabel('Cluster #')
plt.ylabel('% high speed plays')
plt.title('% high speed plays')
# Creates an image of the starting punt formation
def create_formation(gameKey, playId):
    
    # get data
    formation = ngs_raw.loc[(gameKey, playId)]
    print(formation.Event.unique())
    formation = formation.loc[formation.Event == 'line_set']    
    
    fig, ax = plt.subplots(1,1,figsize=(15,10))
    ax.set_facecolor('green')
#     img = plt.imread(os.path.join(images_path, 'footballfield.png'))
#     ax.imshow(img, extent=[0, 120, 0, 53.3])
    ax.set_xlim([0, 120])
    ax.set_ylim([0, 53.3])
    formation['Pos'] = list(zip(formation.x,formation.y))
    formation_o = formation.loc[formation.Role.isin(offense), :]
    formation_d = formation.loc[formation.Role.isin(defense), :]
    formation_o = formation_o.pivot(values=['x', 'y'], index='Time', columns='Role').sort_index().fillna(method='ffill')
    formation_d = formation_d.pivot(values=['x', 'y'], index='Time', columns='Role').sort_index().fillna(method='ffill')

    formation_o = formation_o.loc[formation_o.index.isin(formation_d.index)]
    formation_d = formation_d.loc[formation_d.index.isin(formation_o.index)]
    colors_o = ['#0103f4' for x in formation_o.loc[:, 'x'].columns]
    colors_d = ['black' for x in formation_d.loc[:, 'x'].columns]
    sc_o = ax.scatter([], [], s=75, marker='o')
    sc_d = ax.scatter([], [], s=75, marker='x')
#     print(formation_o.x)
    sc_o.set_offsets(np.c_[formation_o.iloc[0].x.values, formation_o.iloc[0].y.values])
    sc_o.set_facecolor('none')
    sc_o.set_edgecolor(colors_o)
    
    sc_d.set_offsets(np.c_[formation_d.iloc[0].x.values, formation_d.iloc[0].y.values])
    sc_d.set_color(colors_d)

    plt.show()
# Creates and saves a gif of a given punt play
def create_punt_gif(gameKey, playId):
    %matplotlib inline
    ngs = ngs_raw.loc[(gameKey, playId)]
    fig, ax = plt.subplots(1,1,figsize=(15,10))
    img = plt.imread(os.path.join(images_path, 'footballfield.png'))
    ax.imshow(img, extent=[0, 120, 0, 53.3])
    ax.set_xlim([0, 120])
    ax.set_ylim([0, 53.3])
    ngs['Pos'] = list(zip(ngs.x, ngs.y))
    ngs_o = ngs.loc[ngs.Role.isin(offense), :]
    ngs_d = ngs.loc[ngs.Role.isin(defense), :]
    ngs_o = ngs_o.pivot(values=['x', 'y'], index='Time', columns='Role').sort_index().fillna(method='ffill')
    ngs_d = ngs_d.pivot(values=['x', 'y'], index='Time', columns='Role').sort_index().fillna(method='ffill')
#     play_df = play_df.fillna(method='ffill')
#     play_df = play_df.iloc[::5, :]
    ngs_o = ngs_o.loc[ngs_o.index.isin(ngs_d.index)]
    ngs_d = ngs_d.loc[ngs_d.index.isin(ngs_d.index)]
    colors_o = ['blue' for x in ngs_o.loc[:, 'x'].columns]
    colors_d = ['black' for x in ngs_d.loc[:, 'x'].columns]
    sc_o = ax.scatter([], [], marker='o')
    sc_d = ax.scatter([], [], marker='x')
    ngs.Event = ngs.Event.astype(str)
    ngs = ngs.groupby('Time').first()
    title = ax.text(60, 57, "")
    time = ax.text(110, 57, "")

    def animate(i):
#         print(ngs_o.loc[i, 'x'])
        sc_o.set_offsets(np.c_[ngs_o.loc[i, 'x'].values, ngs_o.loc[i, 'y'].values])
        sc_o.set_facecolor('none')
        sc_o.set_edgecolor(colors_o)
        sc_d.set_offsets(np.c_[ngs_d.loc[i, 'x'].values, ngs_d.loc[i, 'y'].values])
        sc_d.set_color(colors_d)
        time.set_text(i)
        if ngs.loc[i, 'Event'] != 'nan':
            title.set_text(ngs.loc[i, 'Event'].upper())

    ani = FuncAnimation(fig, animate, frames=ngs_o.index, interval=100, repeat=True)
    ani.save('footballplay{:s}-{:s}.gif'.format(str(gameKey), str(playId)), writer='imagemagick')
    plt.clf()