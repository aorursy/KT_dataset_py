from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

df = pd.read_csv('../input/nba201718.csv')

print(df.head(10))
print(df.tail(10))

df.info()
df.fillna(0, inplace=True)
len(df['Player'].unique())
df[df.groupby('Player').transform(len)['Tm'] > 1].head(10)
df = df.groupby('Player').first().reset_index()
df.info()
df.describe()
df.drop(df[df['MP'] < 240].index, inplace = True)
df = df.reset_index()
df.describe()
col_simple = ['TRB', 'AST', 'STL', 'BLK','PTS','3PA']
df_simple = df[col_simple]
df_simple_normalized = pd.DataFrame(normalize(df_simple), columns = df_simple.columns)
df_simple_normalized.index = df_simple.index
kmeans = KMeans(n_clusters=5, random_state = 123)
group = kmeans.fit_predict(df_simple_normalized.values)
df_simple = df_simple.assign(Group = pd.Series(group, index = df_simple.index))
print(df_simple['Group'].value_counts())
df = df.assign(Group =  df_simple['Group'].values)
df_simple_normalized = df_simple_normalized.assign(Group =  df_simple['Group'].values)

df_grouped = df_simple.groupby('Group').mean()
df_norm_grouped = df_simple_normalized.groupby('Group').mean()
    
df_grouped.T
by_pos = df.groupby('Group')['Pos'].value_counts()
print(by_pos)
# data transformation which equalizes maximum value of every skill,
# what makes mastery of them comparable

cluster_labels = ['sniper', 'classic center', 'versatile forward', 'playmaker', 'frontcourt']

df_draw = df_norm_grouped.copy()
maximums = []
for column in df_draw:
    maximums.append(df_draw[column].max())
top = max(maximums)
for i,column in enumerate(df_draw):
    df_draw[column] = df_draw[column] * top / maximums[i]

# plotting 5 radar charts
colors = ['red', 'blue', 'green', 'yellow', 'purple']
fig=plt.figure(figsize = (25,10))
for i in range(1, 6):
    stats = df_draw.loc[i-1]
    angles=np.linspace(0, 2*np.pi, len(col_simple), endpoint=False)
    stats=np.concatenate((stats,[stats[0]]))
    angles=np.concatenate((angles,[angles[0]]))
    ax = fig.add_subplot(1, 5, i, polar=True)
    ax.plot(angles, stats, 'o-', linewidth=3, color = colors[i-1])
    ax.fill(angles, stats, alpha=0.25, color = colors[i-1])
    ax.set_thetagrids(angles * 180/np.pi, col_simple)
    ax.set_yticklabels([])
    sub_title = 'Group ' + str(i-1) + ' - ' + cluster_labels[i-1]
    ax.set_title(sub_title)
    ax.title.set_fontsize(18)
    ax.set_rmin(0)
    ax.grid(True)
plt.show()

closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, df_simple_normalized[col_simple])
for i in range(5): 
    print("Group",i,"-",df.loc[closest[i],'Player'].split("\\")[0])
# calculating distance to nearest cluster's centre
dist = np.min(kmeans.transform(df_simple_normalized[col_simple].values), axis=1)
df = df.assign(dist = pd.Series(dist, index = df.index))

pos_color = {'PG':'red',
             'SG':'gold',
             'SF':'green',
             'PF': 'cyan',
             'C':'blue',
             'PG-SG': 'sandybrown',
             'SF-SG': 'olive'
        }

for g in range(5): 
    df_g = df.loc[df['Group']==g].sort_values(by=['MP'], ascending=False).head(25).reset_index()
    fig, ax = plt.subplots(figsize=(20,10)) 
    for p in df_g.Pos:
        ax.scatter(df_g.dist, df_g.MP, s = 150, c=df_g.Pos.map(pos_color))
    ax.legend(pos_color, labelspacing=2)
    leg = ax.get_legend()
    for nmr, c in enumerate(pos_color):
        leg.legendHandles[nmr].set_color(pos_color[c])
    title = "Group " + str(g) + ' - ' + cluster_labels[g]
    ax.set_title(title)
    ax.title.set_fontsize(24)
    for i, txt in enumerate(df_g.Player):
        ax.annotate(txt.split("\\")[0], (df_g.dist[i], df_g.MP[i]), fontsize = 16)
plt.show()