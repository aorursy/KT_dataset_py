import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

%matplotlib inline
df = pd.read_csv('../input/competitive-pokemon-dataset/pokemon-data.csv', delimiter=';')
mdf = pd.read_csv('../input/competitive-pokemon-dataset/move-data.csv', delimiter=',')

print('Number of pokemon: ', len(df))
df.sample()
print('Number of moves: ', len(mdf))
mdf.sample()
#Preliminary Processing:
mdf.columns = ['index', 'name', 'type', 'category', 'contest', 'pp', 'power', 'accuracy', 'generation']
mdf.set_index('index')
mdf['power'].replace('None', 0, inplace=True)
mdf['accuracy'].replace('None', 100, inplace=True)
mdf['power'] = pd.to_numeric(mdf['power'])
mdf['accuracy'] = pd.to_numeric(mdf['accuracy'])
mdf.sample()

df.columns = ['name', 'types', 'abilities', 'tier', 'hp', 'atk', 'def', 'spa', 'spd', 'spe', 'next_evos','moves']

#turn the lists into actual lists
df['next_evos'] = df.apply(lambda x: eval(x.next_evos), axis=1)
df['types'] = df.apply(lambda x: eval(x.types), axis=1)
df['abilities'] = df.apply(lambda x: eval(x.abilities), axis=1)
df['moves'] = df.apply(lambda x: eval(x.moves), axis=1)

df.set_index('name', inplace=True)
weird_moves = set()

for ind, row in df.iterrows():
    for move in row.moves:
        if "'" in move:
            weird_moves.add(move)
            
print(weird_moves)
weird_moves.remove("King's Shield")
weird_moves.remove("Forest's Curse")
weird_moves.remove("Land's Wrath")
weird_moves.remove("Nature's Madness")

df['moves'] = df.apply(
    lambda x: [move if move not in weird_moves else move.replace("'", "-")
                  for move in x.moves],
    axis = 1
)

removal_check_set = set()
for ind, row in df.iterrows():
    for move in row.moves:
        if "'" in move:
            removal_check_set.add(move)

removal_check_set
df['moves'] = df.apply(lambda x: set(x.moves), axis=1)
mdf = mdf[(mdf.pp != 1) | (mdf.name == 'Sketch')]
df.loc['Victini', 'moves'].add('V-create')
df.loc['Rayquaza', 'moves'].add('V-create') #technically should have Mega Rayquaza as well, but it's in AG
df.loc['Celebi', 'moves'].add('Hold Back')

for pok in ['Zygarde', 'Zygarde-10%', 'Zygarde-Complete']:
    df.loc[pok, 'moves'].add('Thousand Arrows')
    df.loc[pok, 'moves'].add('Thousand Waves')
    df.loc[pok, 'moves'].add('Core Enforcer')

for pok in ['Celebi', 'Serperior', 'Emboar', 'Samurott']: #'Mareep', 'Beldum', 'Munchlax' are all LC 
    df.loc[pok, 'moves'].add('Hold Back')

mdf = mdf[(mdf.name != 'Happy Hour') & (mdf.name != 'Celebrate') & (mdf.name != 'Hold Hands') & (mdf.name != 'Plasma Fists')]
def stage_in_evo(n):
    # returns number of evolutions before it
    #print(df[df['name'] == n]['name'])
    bool_arr = df.apply(lambda x: n in x['next_evos'] and (n+'-') not in x['next_evos'], axis=1) #gets index of previous evolution
    if ('-' in n and n.split('-')[0] in df.index and n != 'Porygon-Z'): #'-Mega' in n or  
        #megas and alternate forms should have same evolutionary stage as their base
        return stage_in_evo(n.split('-')[0])
    elif not any(bool_arr):
        return 1 # if there's nothing before it, it's the first
    else:
        return 1 + stage_in_evo(df.index[bool_arr][0])

def num_evos(n):
    if n not in df.index: #checks to ensure valid pokemon
        return n
    
    next_evos = df.loc[n, 'next_evos']
    if len(next_evos) > 0: #existence of next_evo
        if n in next_evos[0]: # if "next evo" is an alternate form
            return df.loc[n, 'stage'] #accounting for alternate forms
        else:
            return num_evos(next_evos[0])
    elif '-Mega' in n or (n.split('-')[0] in df.index and n != 'Porygon-Z'): 
        #this is checking if there is a pokemon with the same root name (e.g. Shaymin vs Shaymin-Sky)
        return df.loc[n.split('-')[0], 'stage']
    else:
        return df.loc[n, 'stage']
df['stage'] = df.apply(lambda x: stage_in_evo(x.name), axis=1)
df['num_evos'] = df.apply(lambda x: num_evos(x.name), axis=1)
df['evo_progress'] = df['stage']/df['num_evos'] 
del df['stage']
del df['num_evos']
df[(df.index == 'Scyther') |
   (df.index == 'Scizor') | 
   (df.index == 'Porygon') |
   (df.index == 'Porygon2') |
   (df.index == 'Porygon-Z')] #test
df['mega'] = df.apply(lambda x: 1 if '-Mega' in x.name else 0, axis=1)
df['alt_form'] = df.apply(lambda x: 1 if ('-' in x.name and 
                                                x.mega == 0 and 
                                                '-Alola' not in x.name and 
                                                x.name.split('-')[0] in df.index and
                                                x.name != 'Porygon-Z')
                                            else 0,
                                            axis = 1)
df[(df.index == 'Landorus-Therian') | 
   (df.index == 'Landorus') | 
   (df.index == 'Shaymin-Sky') | 
   (df.index == 'Blaziken-Mega') | 
   (df.index == 'Diglett-Alola') | 
   (df.index == 'Porygon2') |
   (df.index == 'Porygon-Z')] #test
df.loc[df.tier == 'OUBL','tier'] = 'Uber'
df.loc[df.tier == 'UUBL','tier'] = 'OU'
df.loc[df.tier == 'RUBL','tier'] = 'UU'
df.loc[df.tier == 'NUBL','tier'] = 'RU'
df.loc[df.tier == 'PUBL','tier'] = 'NU'
df = df[df['tier'].isin(['Uber', 'OU', 'UU', 'NU', 'RU', 'PU'])]
tiers = ['Uber', 'OU', 'UU', 'RU', 'NU', 'PU']
tier_mapping = {tier:num for num, tier in enumerate(tiers)}
df['tier_num'] = df.apply(lambda x: tier_mapping[x.tier], axis=1)
tier_mapping
df['num_moves'] = df.apply(lambda x: len(x.moves), axis=1)
df['bst'] = df['hp'] + df['atk'] + df['def'] + df['spa'] + df['spd'] + df['spe']
#df = df[['name', 'types', 'abilities', 'tier', 'hp', 'atk', 'def', 'spa', 'spd', 'spe', 'bst', 'next_evos','moves']]
mdf.loc['Frusuration', 'power'] = 102
mdf.loc['Return', 'power'] = 102
c = sns.color_palette('muted')
c = [c[5], c[1], c[3], c[4], c[2], c[0]]

ax = df.tier.value_counts().plot(kind='pie', autopct='%1.1f%%', colors=c, title='Percentage of Pokemon per Tier')
print('Total number of Pokemon: ', len(df))
tier_size = { t:len(df[df.tier == t]) for t in tiers}
tier_size
stats_df = df[['tier', 'tier_num', 'hp', 'atk', 'def', 'spa', 'spd', 'spe']]
stats_df = stats_df.reset_index()
stats_df = stats_df.melt(id_vars=['name', 'tier', 'tier_num']).sort_values('tier_num', ascending=True)
stats_df.columns = ('name', 'Tier', 'tier_num', 'Stat', 'Value')
#stats_df.Value = pd.to_numeric(stats_df.Value)

sns.set_context('talk')
fig, ax = plt.subplots(1,2, figsize=(25,8), gridspec_kw = {'width_ratios':[3, 1]})
g = sns.boxplot(data=stats_df, x="Stat", y="Value", order=['hp', 'atk', 'def', 'spa', 'spd', 'spe'],
                hue="Tier", palette="muted", ax=ax[0])
g2 = sns.boxplot(data=df, x='tier', y='bst', order=tiers, palette="muted", ax=ax[1])
#g2=sns.factorplot(x="Tier", y="Average", hue_order=['bst'],hue="Stat", data=temp2,
#                   kind="bar", palette="muted", aspect=1.5,  ax=ax[1])
ax[0].set(xlabel='Tier', ylabel='Stat Average', title='Distribution of Stats by Tier')
ax[1].set(xlabel='Tier', ylabel='BST Average', title='Distribution of BST by Tier')
None;
stats_df2 = df.loc[:, ['tier', 'hp', 'atk', 'def', 'spa', 'spd', 'spe']].reset_index().set_index(['name','tier'])
aggregates = {('Top {} Avg'.format(v),(lambda x, v=v: np.mean(np.sort(x)[::-1][:v]))) for v in range(1, 7)} 
stats_df2 = stats_df2.stack().groupby(['name','tier']).agg(aggregates).stack().reset_index()
stats_df2.columns = ['Name', 'Tier', 'Average', 'Stat Average']

plt.subplots(figsize=(17,7))
sns.boxplot(data=stats_df2.sort_values('Average'), hue='Tier', y='Stat Average', x='Average', 
            hue_order=tiers, palette='muted').set_title('Average of Top x Stats by Tier')
None;
stats_df = df[['tier', 'hp', 'atk', 'def', 'spa', 'spd', 'spe']] #'tier_num', , 'bst'
sns.pairplot(stats_df, hue='tier', hue_order=list(reversed(tiers)), plot_kws={'s':25},
               palette=list(reversed(sns.color_palette('muted'))))
from sklearn.decomposition import PCA

pca = PCA(n_components=6)
pca_result = pca.fit_transform(df[['hp', 'atk', 'def', 'spa', 'spd', 'spe']].values)
pca_df = df.copy()
pca_df = pca_df[['tier']]
pca_df['pca_1'] = pca_result[:,0]
pca_df['pca_2'] = pca_result[:,1] 
pca_df['pca_3'] = pca_result[:,2]
print(f'Variation per principal component: {pca.explained_variance_ratio_}')
sns.lmplot(data = pca_df, x='pca_1', y='pca_2', hue='tier', hue_order=tiers, fit_reg=False, palette='muted', scatter_kws={'s':25})

# below code from https://github.com/teddyroland/python-biplot/blob/master/biplot.py
xs = pca_df['pca_1']
ys = pca_df['pca_2']
xvector = pca.components_[0]
yvector = pca.components_[1]
dat = df[['hp', 'atk', 'def', 'spa', 'spd', 'spe']]

for i in range(len(xvector)):
    # arrows project features (ie columns from csv) as vectors onto PC axes
    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
              color='r', width=0.0005, head_width=0.0025)
    plt.text(xvector[i]*max(xs)*1.2, yvector[i]*max(ys)*1.2,
             list(dat.columns.values)[i], color='r')
    
#for i in range(len(xs)):
# circles project documents (ie rows from csv) as points onto PC axes
    #plt.plot(xs[i], ys[i], 'bo', markersize=5)
    #plt.text(xs[i]*1.2, ys[i]*1.2, list(dat.index)[i], color='b')
type_set = set()

for ind, row in df.iterrows():
    type_set |= set(row.types) #for use later

type_df_temp = df.copy()
type_df_temp['type 1'] = type_df_temp.apply(lambda x: sorted(x['types'])[0], axis=1)
type_df_temp['type 2'] = type_df_temp.apply(lambda x: sorted(x['types'])[-1], axis=1) #if a pokemon has a single type, type 2 = type 1

type_df = type_df_temp[['type 2', 'type 1']].groupby(['type 2', 'type 1']).size().reset_index()
type_df.columns = ['type 1', 'type 2', 'count']
type_pivoted_df = type_df.pivot('type 1', 'type 2', 'count')

plt.subplots(figsize=(8,8))
sns.heatmap(type_pivoted_df, annot=True, square=True, cmap='Blues', linecolor='grey', linewidths='0.05')
plt.gca().set(title='Frequency of Type Combinations')
#Get individual counts of type1 and type 2
type1_count = type_df_temp[['tier', 'type 1']].groupby(['tier', 'type 1']).size().reset_index()
type2_count = type_df_temp[['tier', 'type 2']].groupby(['tier', 'type 2']).size().reset_index()
type1_count.columns=['tier', 'type', 'count1']
type2_count.columns=['tier', 'type', 'count2']

#Get overall type frequency per tier
type_count = pd.merge(type1_count, type2_count, on=['tier', 'type'], how='outer')
type_count.fillna(value=0, inplace=True)
type_count['count'] = type_count['count1'] + type_count['count2']
type_count_ind = type_count.set_index(['tier','type'])
type_count['count'] = type_count.apply(lambda x: x['count']/np.sum(type_count_ind.loc[x['tier'], 'count'])
                                      , axis=1) # /np.sum(type_count_ind2.loc[x['tier'], 'count'])

#Format Table and Sort rows
type_count = type_count[['tier','type','count']]
type_count = type_count.set_index(['tier','type']).unstack()['count']
type_count['tier_nums'] = type_count.apply(lambda x: tier_mapping[x.name],axis=1)
type_count = type_count.sort_values(by='tier_nums', ascending=False)
del type_count['tier_nums']

colors = [(104,144,240), (184,184,208), (184,160,56), (248,88,136), 
          (160,64,160), (168,168,120), (152,216,216), (224,192,104), 
          (120,200,80), (112,88,152), (168,144,240), (240,128,48), 
          (192,48,40), (238,153,172), (248,208,48), (112,56,248), 
          (112,88,72), (168,184,32)]
colors = [tuple(i/255.0 for i in c)
               for c in colors]
#Plit
type_count.plot.bar(stacked=True, title='Distribution of Pokemon Types in Tiers', 
                     legend=False, figsize=(12, 7), sort_columns=True, width=0.8,
                    color=reversed(colors))
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), handles=handles[::-1])
mdf.set_index('name', inplace=True)

mdf['uber count'] = 0
mdf['ou count'] = 0
mdf['uu count'] = 0
mdf['ru count'] = 0
mdf['nu count'] = 0
mdf['pu count'] = 0

for ind, row in df.iterrows():
    for move in row.moves:
        mdf.loc[move, row.tier.lower() + ' count'] += 1
        
mdf['count'] = mdf['uber count'] + mdf['ou count'] + mdf['uu count'] + mdf['ru count'] + mdf['nu count'] + mdf['pu count']
#mdf = mdf.reset_index()
plt.figure(figsize=(20, 6))
mdf['count'].hist(bins=50, color=sns.color_palette('muted')[0])
plt.gca().set(title='Frequencies of Moves by the Number of Pokemon that Learn Them')
mdf.nlargest(20, 'count').index
mdf.nsmallest(20, 'count').index
for t in tiers:
    mdf[t + ' %'] = mdf[t.lower() + ' count']/tier_size[t]*100

exclusives = mdf[mdf['count'] <= 3][[t + ' %' for t in tiers]].unstack().reset_index()
del exclusives['name']
exclusives.columns=['Tier', 'Percent of Pokemon that Learn a Given Exclusive Move']
exclusives['Tier'] = exclusives.apply(lambda x: x['Tier'].split(' ')[0], axis=1)

normals = mdf.copy()[[t + ' %' for t in tiers]].unstack().reset_index()
del normals['name']
normals.columns=['Tier', 'Percent of Pokemon that Learn a Given Move']
normals['Tier'] = normals.apply(lambda x: x['Tier'].split(' ')[0], axis=1)

fig, ax = plt.subplots(1,2, figsize=(25,8))
sns.boxplot(data=exclusives, x='Tier', y='Percent of Pokemon that Learn a Given Exclusive Move', palette='muted', ax=ax[0])
sns.boxplot(data=normals, x='Tier', y='Percent of Pokemon that Learn a Given Move', palette='muted', ax=ax[1])
ax[0].set(title='Distribution for Exclusive Moves among the Tiers')
ax[1].set(title='Distribution for All Move samong the Tiers')
num_plots = 10
fig, ax = plt.subplots(1,num_plots, figsize=(25,8))

heatmap_df = mdf.sort_values('count', ascending=False)[[t + ' %' for t in tiers]]
num_elem = len(heatmap_df)
split_heat_df = []

for i in range(0, num_elem, int(num_elem/num_plots)):
    split_heat_df.append(heatmap_df.iloc[i:i+int(num_elem/num_plots)])

for hdf, axis, i in zip(split_heat_df, ax, range(0, num_elem, int(num_elem/num_plots))):
    sns.heatmap(data=heatmap_df.iloc[i:i+int(num_elem/num_plots)], 
                annot=False, cmap='Blues', ax = axis, cbar=False)
    axis.get_yaxis().set_visible(False)
    axis.set(title="{} to {}".format(i, i+int(num_elem/num_plots)-1))
    
fig.suptitle('Percent of Pokemon that Learn each Move by Tier')
exclusive_moves = set(mdf[mdf['count'] <= 3].index)
df['num_exclusive'] = df.apply(lambda x: len(exclusive_moves.intersection(x['moves'])), axis=1)
edf = df.loc[:, ['tier', 'tier_num', 'num_exclusive']]
edf['indicator'] = edf.apply(lambda x: 1 if x.num_exclusive > 0 else 0, axis=1)
edf = edf.pivot_table(values='indicator', index=['tier', 'tier_num'], columns='num_exclusive', fill_value=0, aggfunc=np.count_nonzero)
edf.reset_index(inplace=True)
edf.columns = ['Tier', 'tier_num'] + [str(i) for i in range(5)]
edf.sort_values('tier_num', inplace=True)
del edf['tier_num']
del edf['0']

for i in range(1,  5):
    edf[str(i)] = edf.apply(lambda x: x[str(i)]/tier_size[x['Tier']]*100, axis=1)

edf.set_index('Tier').plot.barh(stacked=True, color=sns.color_palette('muted'), figsize=(10, 5))
plt.gca().set(title='% of Pokemon by Tier that Learn 1+ Exclusive Moves')
plt.legend(title='# Exclusives')
highest_moves = []
#we do not want to count moves with recharge as half power,
#as they waste a turn and are not used commonly used in competitive
moves_w_recharge = {'Blast Burn', 'Frenzy Plant', 'Giga Impact', 'Hydro Cannon',
                      'Hyper Beam', 'Prismatic Laster', 'Roar of Time', 'Rock Wrecker',
                      'Shadow Half'}
# a special move that lowers special atk is not that useful,
# similarly a physical move that lowers atk is not that useful
moves_lower_attack = {'Overheat', 'Draco Meteor', 'Leaf Storm', 'Fleur Cannon', 
                      'Psycho Boost', 'Superpower'}
#these moves cause the user to fiant, so they will not be included
self_destroy = {'Explosion', 'Self-Destruct'}

def get_max_power(moves, typ, category, min_acc):
    moves = list(set(moves) - moves_w_recharge - moves_lower_attack - self_destroy)
    highest = np.max([mdf.loc[m, 'power'] if mdf.loc[m, 'category'] == category
                                          and mdf.loc[m, 'accuracy'] >= min_acc 
                                          and mdf.loc[m, 'type'] == typ
                                       else 0
                        for m in moves])
    return highest

def get_primary(x):
    atk_higher = x.atk >= x.spa
    spa_higher = x.spa >= x.atk
    candidates = []
    for t in x.types:
        candidates.append(x[t+'_physical'] if x.atk >= x.spa else 0)
        candidates.append(x[t+'_special'] if x.atk <= x.spa else 0)
    return np.max(candidates)

for t in type_set:
    df[t+'_physical'] = df.apply(lambda x: get_max_power(x.moves, t, 'Physical', 85), axis=1)
    df[t+'_special'] = df.apply(lambda x: get_max_power(x.moves, t, 'Special', 85), axis=1)
    highest_moves += [t+'_physical', t+'_special']

df['primary_attack'] = df.apply(get_primary, axis=1)
offensive_pokemon = df.apply(lambda x: max(x['atk'], x['spa']) > max(x['def'], x['spd']), axis=1)
a=sns.boxplot(data=df[offensive_pokemon], x='tier', y='primary_attack', palette='muted', order=tiers)
a.set(title='Distribution of Primary Attack Power for Offensive Pokemon')
None
move_set = set()

for ind, row in df.iterrows():
    move_set |= row.moves

move_df = df[['tier', 'tier_num']]

for m in move_set:
    move_df[m] = df.apply(lambda x: 1 if m in x.moves else 0, axis = 1)
from kmodes.kmodes import KModes
import pickle

num_clusters = list(range(1, 150))
cost = []
models = []
temp_data = move_df[list(move_set)]

'''
for c in num_clusters:
    km = KModes(n_clusters=c, init='Huang', n_init=5)
    clusters = km.fit(temp_data)
    cost.append(km.cost_)
    models.append(km)
    with open('model' + str(c) + '.pkl', 'wb') as f:
       pickle.dump(km, f)
'''

with open('../input/pokemonextras/kmodes-cost.pkl', 'rb') as f:
    cost = pickle.load(f)
plt.figure(figsize=(17,5))
plt.gca().set(title='Cost vs Number of Clusters for K-Modes', xlabel='Number of Clusters', ylabel='Cost')
plt.scatter(num_clusters, cost, s=20)
plt.plot(num_clusters, cost)
from collections import defaultdict
a_dict = defaultdict(int)

for ind, row in df.iterrows():
    for ability in row.abilities:
        a_dict[(ability, row.tier + ' count')] += 1


adf = pd.DataFrame(pd.Series(a_dict)).reset_index() #(columns=(['name'] + [t + ' Count' for t in tiers]))
adf.columns=['name', 'tier', 'count']
adf = adf.pivot_table(values='count', index='name', columns='tier', fill_value=-0)
adf['count'] = sum(adf[t + ' count'] for t in tiers)
plt.figure(figsize=(20, 4))
adf['count'].hist(bins=20, color=sns.color_palette('muted')[0])
plt.gca().set(title='Frequencies of Abilities by the Number of Pokemon that have Them')
bad_abilities = {'Comatose', 'Defeatist', 'Emergency Exit', 'Slow Start', 'Truant', 'Wimp Out', 'Stall'}
df['bad_ability'] = df.apply(lambda x: 1 if len(set(x['abilities']).intersection(bad_abilities)) == len(x['abilities'])
                                       else 0, axis=1)
df[df.bad_ability == 1]
evodf = df.loc[:, ['tier', 'tier_num', 'evo_progress']]
evodf['count'] = evodf.apply(lambda x: 1/tier_size[x.tier], axis=1) 
#so when we sum everything up, values will be normalized to tier size
evodf = evodf.pivot_table(values='count', index=['tier', 'tier_num'], columns='evo_progress', fill_value=0, aggfunc=np.sum)
evodf = evodf.sort_values('tier_num').reset_index()
del evodf['tier_num']
evodf.columns = ['Tier', '0.50', '0.67', '1.00']
evodf.set_index('Tier', inplace=True)

plt.figure(figsize=(20, 4))
evodf.plot.barh(stacked=True, color=sns.color_palette('muted')[2::-1], figsize=(12, 3), 
                title='Distribution of Evolutionary Stages by Tier')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),title='Evolutionary Progress')
distdf = df.loc[df['evo_progress'] < 1, ['tier', 'tier_num','next_evos']] #'evo progress',
distdf['dist'] = distdf.apply(lambda x: x.tier_num - np.max(df.loc[x.next_evos, 'tier_num']), axis=1)
print('The next evolution of unevolved pokemon are on average {:.2f}±{:.2f} tiers above.'
      .format(np.mean(distdf['dist']), np.std(distdf['dist'])))

freqdf = distdf.reset_index()
freqdf = freqdf[['dist', 'tier']].groupby('dist').count().reset_index()#.sort_values('tier_num
freqdf.columns = ['Distance', 'Count']
freqdf.set_index('Distance').plot.bar(title='Frequency of Tier Distances', legend=None)
plt.ylabel('Frequency')
altdf = df.loc[:, ['tier', 'tier_num', 'mega', 'alt_form']]
altdf['mega'] = altdf.apply(lambda x: x.mega/tier_size[x.tier], axis=1) 
altdf['alt_form'] = altdf.apply(lambda x: x.mega/tier_size[x.tier], axis=1)
altdf['normal'] = altdf.apply(lambda x: 1/tier_size[x.tier] if x['mega'] == 0 and x['alt_form'] == 0 else 0, axis=1)
#so when we sum everything up, values will be normalized to tier size

#altdf = altdf.pivot_table(values='count', index=['tier', 'tier_num'], columns='evo progress', fill_value=0, aggfunc=np.sum)
altdf = altdf.groupby(['tier', 'tier_num']).agg(np.sum).reset_index().sort_values('tier_num')
del altdf['tier_num']
altdf.columns = ['Tier', 'Mega', 'Alternate', 'Base']
altdf.set_index('Tier', inplace=True)

plt.figure(figsize=(20, 4))
altdf.plot.barh(stacked=True, color=sns.color_palette('muted')[2::-1], figsize=(12, 3), 
                title='Distribution of Forms by Tier')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),title='Form')
alt = df.loc[(df['alt_form'] == 1), ['tier_num']]
mega = df.loc[(df['mega'] == 1), ['tier_num']]
alt_base = df.loc[set(map(lambda x: x.split('-')[0], alt.index)), ['tier_num']]
mega_base = df.loc[set(map(lambda x: x.split('-')[0], mega.index)), ['tier_num']]

alt['Form'] = 'Alternate'
mega['Form'] = 'Mega'
alt_base['Form'] = 'Alternate Base'
mega_base['Form'] = 'Mega Base'

combined = pd.concat([mega, mega_base, alt, alt_base])
combined.columns = ['Tier', 'Form']
plt.gca().invert_yaxis()
plt.gca().set(title='Distribution of Forms among Tiers', yticklabels=['']+tiers)
sns.boxplot(data=combined, x='Form', y='Tier', palette='muted')
bstdf = df[['tier', 'bst']].groupby('tier').agg([np.mean, np.std])
bstdf.columns = ['bst_mean', 'bst_std']
df2 = df.reset_index().merge(bstdf, left_on='tier', right_on='tier')

under = df2[(df2['bst'] < df2['bst_mean'] - 2*df2['bst_std']) & (df2['tier'] != 'PU')]
over = df2[(df2['bst'] > df2['bst_mean'] + 2*df2['bst_std']) & (df2['tier'] != 'Uber')]

under[['tier', 'name', 'types', 'abilities', 'hp', 'atk', 'def', 'spa', 'spd', 'spe', 'bst', 'bst_mean', 'evo_progress', 'num_moves', 'num_exclusive']]
over[['tier', 'name', 'types', 'abilities', 'hp', 'atk', 'def', 'spa', 'spd', 'spe', 'bst', 'bst_mean', 'evo_progress', 'num_moves', 'num_exclusive', 'bad_ability']]
#defenses only: {'Acid Armor': 2, 'Barrier': 2,' Cotton Guard': 3, 'Iron Defense': 2, 'Stockpile': 2, 'Amnesia': 2
stat_increasing = {'Coil': 3, 'Hone Claws': 2, 'Belly Drum': 6, 'Bulk Up': 2, 'Clangorous Soulblaze': 4, 
                   'Dragon Dance': 2, 'Shell Smash': 4, 'Shift Gear': 3, 'Swords Dance': 2, 'Work Up': 2,
                   'Cosmic Power': 2,  'Defend Order': 2, 'Calm Mind': 2, 'Geomancy': 6, 
                   'Nasty Plot': 2, 'Quiver Dance': 3, 'Tail Glow': 3, 'Agility': 2, 'Automize': 2, 'Rock Polish': 2}


df['stat_inc_move'] = df.apply(lambda x: np.max([0]+[stat_increasing[v] for v in x.moves.intersection(stat_increasing)]), axis=1)
atk_inc_ability = {'Huge Power', 'Pure Power'}
df['atk_inc_ability'] = df.apply(lambda x: 1 if len(set(x.abilities).intersection({'Huge Power', 'Pure Power'})) > 0 else 0, axis=1)
recovery = {'Heal Order', 'Milk Drink', 'Moonlight', 'Morning Sun', 'Purify', 'Recover',
            'Roost', 'Shore Up', 'Slack Off', 'Soft-Boiled', 'Synthesis', 'Strength Sap', 'Wish'}
df['recovery_move'] = df.apply(lambda x: 1 if len(x.moves.intersection(recovery)) > 0 else 0, axis=1)
priority = {'Fake Out', 'Extreme Speed', 'Feint', 'Aqua Jet', 'Bullet Punch', 'Ice Shard', 'Accelerock'
            'Mach Punch', 'Shadow Sneak', 'Sucker Punch', 'Vacuum Wave', 'Water Shuriken'}
df['priority_stab'] = df.apply(lambda x: 1 if any([(mdf.loc[m, 'type'] in x.types) 
                                                   for m in x.moves.intersection(priority)]) else 0,
                               axis=1)
#To do this efficiently, we will simply create a dictionary of moves and abilities.
#We will map all of them to themselves to start, then alter the variations
ability_set = set()
move_set = set()
type_set = set()

for ind, row in df.iterrows():
    ability_set |=  set(row.abilities) #union
    move_set |= row.moves
    type_set |= set(row.types)

ability_dict = {s:{s} for s in ability_set if s not in {
                   'Battle Armor', 'White Smoke', 'Full Metal Body', 'Solid Rock', 'Prism Armor', 'Gooey', 
                   'Magnet Pull', 'Shadow Tag', 'Inner Focus', 'Insomnia', 'Vital Spirit', 'Limber', 'Magma Armor', 
                  'Own Tempo', 'Oblivious', 'Water Veil', 'Sweet Veil', 'Aroma Veil', 'Hyper Cutter', 'Big Pecks',
                   'Triage', 'Heatproof', 'Iron Barbs', 'Quick Feet', 'Flare Boost', 'Toxic Boost'
               }} #dictionary of sets

#We will not consolidate weather-variations because the viability of various weather conditions varies

ability_dict['Shell Armor'].add('Battle Armor')
ability_dict['Clear Body'] |= {'White Smoke', 'Full Metal Body'}
ability_dict['Filter'] |= {'Solid Rock', 'Prism Armor'}
ability_dict['Tangling Hair'].add('Gooey')

# Below are cases where the abilities aren't identical, but close enough
ability_dict['Arena Trap'] |= {'Magnet Pull', 'Shadow Tag'} 
ability_dict['Guts'] |= {'Quick Feet', 'Flare Boost', 'Toxic Boost'} # Marvel scale is excluded from this because it boosts defense
ability_dict['Immunity'] |= {'Inner Focus', 'Insomnia', 'Vital Spirit', 'Limber', 'Magma Armor', 
          'Own Tempo', 'Oblivious', 'Water Veil', 'Sweet Veil', 'Aroma Veil'}
ability_dict['Keen Eye'] |= {'Hyper Cutter', 'Big Pecks'} 
ability_dict['Prankster'].add('Triage')
ability_dict['Thick Fat'].add('Heatproof')
ability_dict['Rough Skin'].add('Iron Barbs')
#water absorb and dry skin?
entry_hazards = {'Toxic Spikes', 'Stealth Rock', 'Spikes'}
df['entry_hazards'] = df.apply(lambda x: 1 if len(x.moves.intersection(entry_hazards)) > 0 else 0, axis=1)

hazard_clear = {'Rapid Spin'} #we may later exclude/add defog 
df['hazard_clear'] = df.apply(lambda x: 1 if len(x.moves.intersection(hazard_clear)) > 0 else 0, axis=1)

phazing_moves = {'Roar', 'Whirlwind', 'Dragon Tail', 'Circle Throw'}
df['phazing_moves'] = df.apply(lambda x: 1 if len(x.moves.intersection(phazing_moves)) > 0 else 0, axis=1)

switch_attack = {'U-turn', 'Volt Switch'}
df['switch_attack'] = df.apply(lambda x: 1 if len(x.moves.intersection(switch_attack)) > 0 else 0, axis=1)

#strong moves (>65 power) that have a >30% chance of causing side effects with an accuracy over 85%
high_side_fx_prob = {'Steam Eruption','Sludge Bomb', 'Lava Plume', 'Iron Tail', 'Searing Shot', 
                     'Rolling Kick', 'Rock Slide', 'Poison Jab', 'Muddy Water', 'Iron Head',
                    'Icicle Crash', 'Headbutt', 'Gunk Shot', 'Discharge', 'Body Slam', 'Air Slash'}
df['high_side_fx_prob'] = df.apply(lambda x: 1 if len(x.moves.intersection(high_side_fx_prob)) > 0 else 0, axis=1)

constant_dmg = {'Seismic Toss', 'Night Shade'}
df['constant_dmg'] = df.apply(lambda x: 1 if len(x.moves.intersection(constant_dmg)) > 0 else 0, axis=1)

trapping_move = {'Mean Look', 'Block', 'Spider Web'}
df['trapping_move'] = df.apply(lambda x: 1 if len(x.moves.intersection(trapping_move)) > 0 else 0, axis=1)
stats = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'] #bst excluded
forms = ['evo_progress', 'mega', 'alt_form']
moves_based = ['num_moves','num_exclusive', 'bad_ability', 'stat_inc_move', 'recovery_move', 'priority_stab',
              'entry_hazards', 'hazard_clear', 'phazing_moves', 'switch_attack', 'high_side_fx_prob', 'constant_dmg',
              'trapping_move']
ability_based = ['atk_inc_ability', 'bad_ability']

df_y = df.loc[:, 'tier_num']
df_x = df.loc[:, stats + forms + moves_based + ability_based + highest_moves] 
#remove bst because it is just a sum of the individual stats
ability_set -= atk_inc_ability | bad_abilities | set(adf[adf['count'] <= 2].index)
move_set -=  stat_increasing.keys() | recovery | priority | set(mdf[(mdf['count'] <= 3) | (mdf['power'] > 0)].index) \
             | entry_hazards | hazard_clear | phazing_moves | switch_attack | high_side_fx_prob | constant_dmg | trapping_move

for a in ability_dict.keys():
    df_x[a] = df.apply(lambda x: 1 if len(ability_dict[a].intersection(x.abilities))>0 else 0, axis = 1)
for m in move_set:
    df_x[m] = df.apply(lambda x: 1 if m in x.moves else 0, axis = 1)
for t in type_set:
    df_x[t] = df.apply(lambda x: 1 if t in x.types else 0, axis = 1)
counts = df_x.astype(bool).sum(axis=0)
#let's exclude our engineered features and let the model decide if those are important
counts.drop(stats + forms + moves_based + ability_based + highest_moves, inplace=True)
cutoff = list(range(1, 30))
num_vars = [len(df_x.drop(counts[counts<=c].index.values, axis=1).columns) for c in cutoff]
fig, ax = plt.subplots(figsize=(15, 4))
ax.set(title='Number of Features vs Minimum Number of Pokemon with that Feature', 
       xlabel='Min Pokemon with Feature',ylabel='Number of Features')
ax.scatter(cutoff, num_vars, s=50)
None
df_x.drop(counts[counts<=7].index.values, axis=1, inplace=True)
len(df_x.columns)
import statsmodels.api as sm

temp_df_x = df_x.copy()
temp_df_x['Intercept'] = np.ones((len(temp_df_x),))
ols_model = sm.OLS(df_y, temp_df_x).fit()

def fitted_to_tier(num):
    if num < 0: return 0
    elif num > 5: return 5
    else: return num

results = df_y.to_frame()
results['Fitted Values'] = ols_model.predict(temp_df_x)
results['Modified Fitted Values'] = results['Fitted Values'].map(fitted_to_tier)
results['Residuals'] = df_y.values - ols_model.predict(temp_df_x)

fig, ax = plt.subplots(1, 2, figsize=(15, 4))
ax[0].set(title='Residuals vs Fitted Values')
ax[1].set(title='Residuals vs Fitted Values as Contour Plot')
sns.regplot(data=results, x='Fitted Values', y='Residuals', ax=ax[0], scatter_kws={'s':8})
sns.kdeplot(data=results[['Fitted Values', 'Residuals']], ax=ax[1])
fig, ax = plt.subplots(figsize=(10, 3))
ax.set(title='Distribution of Residuals', ylabel='Count', xlabel='Residuals')
results['Residuals'].hist(ax=ax, bins=15)
results['Studentized Residuals'] = ols_model.outlier_test().student_resid
fig, ax = plt.subplots(figsize=(10, 4))
ax.set(title='Studentized Residuals vs Fitted Values')
sns.regplot(data=results, x='Fitted Values', y='Studentized Residuals', ax=ax)
results[abs(results['Studentized Residuals']) >= 3]
from sklearn import linear_model
from sklearn import feature_selection
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample 
alphas, rmse, num_sig = np.arange(0.0001, 0.1, 0.0001), [], []

for a in alphas:
    reg = linear_model.Lasso(alpha=a, normalize=True, max_iter=10000)
    '''
    temp_rmse, temp_sig = [], []
    for i in range(500):
        temp_x, temp_y = resample(df_x, df_y, n_samples=400)
        reg.fit(temp_x, temp_y)
        temp_rmse.append(np.sqrt(mean_squared_error(df_y, reg.predict(df_x))))
        temp_sig.append(reg.coef_)
    
    temp_sig = zip(*temp_sig) #flips
    temp_sig2 = [(np.mean(coef), np.std(coef)) for coef in temp_sig]
    
    rmse.append(np.mean(temp_rmse))
    num_sig.append(np.sum([1 for c in temp_sig2 if abs(c[0]) >= c[1] and abs(c[0]) > 0.0001]))
    '''
    reg.fit(df_x, df_y)
    rmse.append(np.sqrt(mean_squared_error(df_y, reg.predict(df_x))))
    num_sig.append(sum([1 for c in reg.coef_ if abs(c) > 0.0001]))
    #not c[0]-2*c[1] <= 0 <= c[0]+2*c[1]
def easy_dual_plot(x, y1, y2, figsize=(12, 4), title='', xlabel='', ylabel1='', ylabel2=''):
    fig, ax = plt.subplots(figsize=figsize) #2, 1, 
    ax.set(title=title)
    ax.plot(x, y1, color='b')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel1, color='b')
    ax.tick_params('y', colors='b')

    ax2 = ax.twinx()
    ax2.plot(x, y2, color='r')
    ax2.set_ylabel(ylabel2, color='Red')
    ax2.tick_params('y', colors='r')

easy_dual_plot(alphas, rmse, num_sig, title='RMSE and # Non-Zero Features vs Alpha',
               xlabel='Alpha', ylabel1='RMSE', ylabel2='# 65% Confidence Non-zero Features')
reg = linear_model.Lasso(alpha=0.003, normalize=True, max_iter=100000)
reg.fit(df_x, df_y)
'''
temp_sig=[]
for i in range(1000):
    temp_x, temp_y = resample(df_x, df_y, n_samples=400)
    reg.fit(temp_x, temp_y)
    temp_rmse.append(np.sqrt(mean_squared_error(df_y, reg.predict(df_x))))
    temp_sig.append(reg.coef_)
    
temp_sig = zip(*temp_sig) #flips
reg_coef = [np.mean(coef) for coef in temp_sig]
std = [np.std(coef) for coef in temp_sig]
'''

coef = sorted([(prop, c) for c, prop in zip(reg.coef_, df_x.columns)], key=(lambda x: -abs(x[1])))
coef = list(filter(lambda x: abs(x[1]) > 0.0001, coef))
print('{} non-zero features'.format(sum([1 for c in coef if abs(c[1]) > 0.0001])))
print()
print("{0:<18} {1:>6}".format('Feature', 'Coeff')) #{2:>10} , 'Std'
print('-'*26)
for c in coef:
    #print("{0:<18} {1:>6.3f} {2:>10.3f} {3}".format(*c, ('  65% Confident' if abs(c[1]) > 2*abs(c[2]) else '')))
    print("{0:<18} {1:>6.3f}".format(*c)) #{2:>10.3f}
import sklearn.ensemble as ens
import sklearn.tree as tree
from sklearn.model_selection import cross_val_score

model = tree.DecisionTreeRegressor(max_depth=15, max_leaf_nodes = 30)
model.fit(df_x, df_y)

train_rmse = np.mean(np.sqrt(mean_squared_error(df_y, model.predict(df_x))))
cv_rmse = np.mean(np.sqrt(-cross_val_score(model, df_x, df_y, cv=10, scoring='neg_mean_squared_error')))
print('Train Error: {}'.format(train_rmse))
print('CV Error:    {}'.format(cv_rmse))
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
g = tree.export_graphviz(model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                feature_names=df_x.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
df_x.drop(list(move_set.intersection(set(df_x.columns))), axis=1, inplace=True)

model2 = tree.DecisionTreeRegressor(max_depth=10, max_leaf_nodes = 20)
model2.fit(df_x, df_y)

train_rmse = np.mean(np.sqrt(mean_squared_error(df_y, model2.predict(df_x))))
cv_rmse = np.mean(np.sqrt(-cross_val_score(model2, df_x, df_y, cv=10, scoring='neg_mean_squared_error')))
print('Train Error: {}'.format(train_rmse))
print('CV Error:    {}'.format(cv_rmse))
dot_data = StringIO()
g = tree.export_graphviz(model2, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                feature_names=df_x.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
model3 = ens.RandomForestRegressor(n_estimators=15, criterion='mae', max_depth=50, max_features='auto')
model3.fit(df_x, df_y)

train_rmse = np.mean(np.sqrt(mean_squared_error(df_y, model3.predict(df_x))))
cv_rmse = np.mean(np.sqrt(-cross_val_score(model3, df_x, df_y, cv=10, scoring='neg_mean_squared_error')))
print('Train Error: {}'.format(train_rmse))
print('CV Error:    {}'.format(cv_rmse))
model4 = ens.GradientBoostingRegressor(loss='ls')
#interestingly, least squares worked better than least absolute deviance
model4.fit(df_x, df_y)

train_rmse = np.mean(np.sqrt(mean_squared_error(df_y, model4.predict(df_x))))
cv_rmse = np.mean(np.sqrt(-cross_val_score(model4, df_x, df_y, cv=10, scoring='neg_mean_squared_error')))
print('Train Error: {}'.format(train_rmse))
print('CV Error:    {}'.format(cv_rmse))
final_model = ens.GradientBoostingRegressor(loss='ls', criterion='mae')
#interestingly, least squares worked better than least absolute deviance
final_model.fit(df_x, df_y)
np.sqrt(mean_squared_error(df_y, final_model.predict(df_x)))

train_rmse = np.mean(np.sqrt(mean_squared_error(df_y, model4.predict(df_x))))
cv_rmse = np.mean(np.sqrt(-cross_val_score(model4, df_x, df_y, cv=10, scoring='neg_mean_squared_error')))
print('Train Error: {}'.format(train_rmse))
print('CV Error:    {}'.format(cv_rmse))
max_depth_range = list(range(1, 10))
train_errors, cv_errors = [],[]

for d in max_depth_range:
    final_model = ens.GradientBoostingRegressor(loss='ls', criterion='mae', max_depth = d)
    final_model.fit(df_x, df_y)
    train_errors.append(np.mean(np.sqrt(mean_squared_error(df_y, model4.predict(df_x)))))
    cv_errors.append(np.mean(np.sqrt(-cross_val_score(model4, df_x, df_y, cv=10, scoring='neg_mean_squared_error'))))
easy_dual_plot(max_depth_range, cv_errors, train_errors, title='Training and CV Error vs Max Depth',
               xlabel='Max Depth', ylabel1='CV RMSE', ylabel2='Training RMSE')
learning_rate = np.arange(0.05, 0.5, 0.05)
train_errors, cv_errors = [],[]

for l in learning_rate:
    final_model = ens.GradientBoostingRegressor(loss='ls', criterion='mae', max_depth = 5, learning_rate=l)
    final_model.fit(df_x, df_y)
    train_errors.append(np.mean(np.sqrt(mean_squared_error(df_y, model4.predict(df_x)))))
    cv_errors.append(np.mean(np.sqrt(-cross_val_score(model4, df_x, df_y, cv=10, scoring='neg_mean_squared_error'))))
easy_dual_plot(learning_rate, cv_errors, train_errors, title='Training and CV Error vs Learning Rate',
               xlabel='Learning Rate', ylabel1='CV RMSE', ylabel2='Training RMSE')
final_model = ens.GradientBoostingRegressor(loss='ls', criterion='mae', max_depth=5)
final_model.fit(df_x, df_y)

train_rmse = np.mean(np.sqrt(mean_squared_error(df_y, model4.predict(df_x))))
cv_rmse = np.mean(np.sqrt(-cross_val_score(model4, df_x, df_y, cv=10, scoring='neg_mean_squared_error')))
print('Train Error: {}'.format(train_rmse))
print('CV Error:    {}'.format(cv_rmse))
importances = final_model.feature_importances_[:25]
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(25):
    print("{}. feature {} ({})".format(f + 1, df_x.columns[indices[f]], importances[indices[f]]))
df['prediction'] = final_model.predict(df_x)
df['rounded_prediction'] = round(df['prediction'])
df['residuals'] = df['tier_num'] - df['prediction']
df.loc[df['rounded_prediction'] != df['tier_num'],
       ['tier', 'tier_num', 'prediction', 'rounded_prediction', 'residuals']].sort_values('residuals').sample(20)
fig, ax = plt.subplots(figsize=(10, 3))
ax.set(title='Distribution of Residuals', ylabel='Count', xlabel='Residuals')
results['Residuals'].hist(ax=ax, bins=15)