import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import gc # garbage collect to clean memory

import seaborn as sns

import datetime

# Any results you write to the current directory are saved as output.
train_set = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv', 

                        usecols=['event_id', 'game_session', 'timestamp', 'installation_id', 'event_count', 'event_code', 'game_time', 'title', 'type', 'world'], 

                        dtype= {'event_id':'category', 'game_session':'category', 'installation_id':'category', 'event_count':'int16', 'event_code':'int16', 'game_time':'int32', 'title':'category', 'type':'category', 'world':'category'},

                        parse_dates=['timestamp']) 

train_labels_set = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv', dtype={'game_session':'category', 'installation_id':'category','title':'category','num_correct':'int8','num_incorrect':'int8','accuracy_group':'int8','accuracy':'float16'})
test_set = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv', 

                        usecols=['event_id', 'game_session', 'timestamp', 'installation_id', 'event_count', 'event_code', 'game_time', 'title', 'type', 'world'], 

                        dtype= {'event_id':'category', 'game_session':'category', 'installation_id':'category', 'event_count':'int16', 'event_code':'int16', 'game_time':'int32', 'title':'category', 'type':'category', 'world':'category'},

                        parse_dates=['timestamp'])

pd.options.display.max_columns = None

pd.options.display.max_rows = 25 
plt.pie(train_labels_set['accuracy_group'].value_counts().values, labels=train_labels_set['accuracy_group'].unique(),

        autopct='%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')

plt.show()
sns.set_style("darkgrid")

chart = sns.distplot(train_set['game_time'].apply(np.log1p), color="deeppink", label='train set')

sns.kdeplot(test_set['game_time'].apply(np.log1p), color="teal", label='test set')

chart.set_xlabel('Game_time (log(milisecond))')

chart.set_title("Game time distribution")
str(datetime.timedelta(seconds=(np.expm1(11.5)/1000)))
sns.set_style("darkgrid")

chart = sns.distplot(train_set.groupby('game_session').game_time.sum().apply(np.log1p), color="deeppink", label='train set')

sns.kdeplot(test_set.groupby('game_session').game_time.sum().apply(np.log1p), color="teal", label='test set')

chart.set_xlabel('Game_time (log(milisecond))')

chart.set_title("Distribution of the game time per game session")
game_time_set = train_set.loc[(train_set['game_time']>0)].copy()

game_time_set['game_session'] = game_time_set.game_session.cat.remove_unused_categories() 
sns.set_style("darkgrid")

chart = sns.distplot(game_time_set.groupby('game_session').game_time.sum().apply(np.log1p), color="deeppink", label='train set')

# sns.kdeplot(test_set.groupby('game_session').game_time.sum().apply(np.log1p), color="teal", label='test set')

chart.set_xlabel('Game_time (log(milisecond))')

chart.set_title("Distribution of the game time per game session")
str(datetime.timedelta(seconds=(np.expm1(15)/1000)))
game_time_set = train_set.loc[(train_set['game_time']>0) & (train_set['type']=='Assessment')].copy()

game_time_set['game_session'] = game_time_set.game_session.cat.remove_unused_categories() 
sns.set_style("darkgrid")

chart = sns.distplot(game_time_set.groupby('game_session').game_time.sum().apply(np.log1p), color="deeppink", label='train set')

# sns.kdeplot(test_set.groupby('game_session').game_time.sum().apply(np.log1p), color="teal", label='test set')

chart.set_xlabel('Game_time (log(milisecond))')

chart.set_title("Distribution of the game time per game session")
import datetime



fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False, dpi=100)

chart = sns.distplot(train_set['timestamp'].dt.dayofweek, color="teal", ax=axes[0])

chart.set_title("Day of the week distribution (train_set)")

chart = sns.distplot(test_set['timestamp'].dt.dayofweek, color="teal", ax=axes[1])

chart.set_title("Day of the week distribution (test_set)")
cat_order = ['Watering Hole (Activity)', 

'Fireworks (Activity)',

'Sandcastle Builder (Activity)', 

'Flower Waterer (Activity)', 

'Bottle Filler (Activity)', 

'Bug Measurer (Activity)',

'Chicken Balancer (Activity)',

'Egg Dropper (Activity)',  

'All Star Sorting', 

'Air Show', 

'Scrub-A-Dub', 'Dino Drink', 

'Bubble Bath', 'Dino Dive', 'Crystals Rule', 

'Chow Time',

'Pan Balance', 'Happy Camel', 

'Leaf Leader',

'Mushroom Sorter (Assessment)', 

'Bird Measurer (Assessment)',

'Cauldron Filler (Assessment)', 

'Cart Balancer (Assessment)', 

'Chest Sorter (Assessment)',

'Welcome to Lost Lagoon!', 

'Magma Peak - Level 1',

'Magma Peak - Level 2',

'Tree Top City - Level 1',

'Tree Top City - Level 2',

'Tree Top City - Level 3',

'Ordering Spheres', 'Slop Problem',

'Costume Box', 

'12 Monkeys', "Pirate's Tale",

'Treasure Map',

'Rulers', 

'Crystal Caves - Level 1', 'Crystal Caves - Level 2', 'Crystal Caves - Level 3', 

'Balancing Act', 'Lifting Heavy Things','Honey Cake', 'Heavy, Heavier, Heaviest']
plt.figure(figsize=(20,15))

chart = sns.countplot(x='title',order=cat_order,hue='type',dodge=False,data=train_set)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')

chart.set_title("Activity's title distribution and type of the activities (train_set)")
plt.figure(figsize=(20,15))

chart = sns.countplot(x='title',order=cat_order,hue='type',dodge=False,data=test_set)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')

chart.set_title("Activity's title distribution and type of the activities (test_set)")
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False, dpi=100)

chart = sns.countplot(train_set['type'], ax=axes[0])

chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')

chart.set_title("Type distribution (train_set)")

chart = sns.countplot(test_set['type'], ax=axes[1])

chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')

chart.set_title("Type distribution (test_set)")
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False, dpi=100)

chart = sns.countplot(x='world',hue='type',data=train_set, ax=axes[0])#dodge=False,

chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')

chart.set_title("Type in world distribution (train_set)")

chart = sns.countplot(x='world',hue='type',data=test_set, ax=axes[1])#dodge=False,

chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')

chart.set_title("Type in world distribution (test_set)")
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False, dpi=100)

chart = sns.countplot(x='world',data=train_set.loc[train_set['type']=='Assessment'], ax=axes[0])

chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')

chart.set_title("Assessment distribution among world (train_set)")

chart = sns.countplot(x='world',data=test_set.loc[train_set['type']=='Assessment'], ax=axes[1])

chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')

chart.set_title("Assessment distribution among world (test_set)")
train_xy = train_set.join(train_labels_set.set_index(['game_session','installation_id','title']), on=['game_session','installation_id','title'])
train_xy = train_xy.astype({'title': 'category'}, copy=False)
sns.set(style = "white", font_scale=1)

corr=train_xy.corr()

# mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)]= True

# set up the matplot figure

f, ax = plt.subplots(figsize=(15,10))

f.suptitle("Correlation matrix", fontsize = 16)



cmap = sns.diverging_palette(220,10,as_cmap=True)



sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink":.5})
var = 'title'

sns.set_style("darkgrid")

f, ax = plt.subplots(figsize=(6, 8))

fig = sns.boxplot(x=var, y="accuracy_group", data=train_labels_set)

fig.axis(ymin=-0, ymax=3.5)

fig.set_title('Box plot of accuracy_group according to the assessment title')

plt.xticks(rotation=90);
cross_title_group = pd.crosstab(train_labels_set["accuracy_group"],train_labels_set["title"])
x = [[0,1,2,3,4]]*4

y = [[0]*5,[1]*5,[2]*5,[3]*5]

plt.scatter(x,y,data=train_labels_set, s=cross_title_group)

plt.ylabel('accuracy groups')

plt.xticks([0,1,2,3,4], ['Bird Measurer', 'Cart Balancer', 'Cauldron Filler', 'Chest Sorter', 'Mushroom Sorter'], rotation=45, horizontalalignment='right')

plt.yticks([0,1,2,3],[0,1,2,3])

plt.title('Predominance of accuracy group per assessment title')

plt.show()
pd.set_option('max_colwidth', 500)

train_set.loc[train_set['type']=='Assessment'].groupby('world').title.unique()
train_labels_set.groupby(["installation_id"])["accuracy_group"].count().nlargest(5)

train_labels_set.groupby(["game_session"])["accuracy_group"].count().nlargest(5)
train_labelled = train_set.loc[train_set['installation_id'].isin(train_labels_set['installation_id'])]
train_xy_ass = train_labelled.loc[train_labelled['type']=='Assessment'].iloc[:,[1, 2, 3, 9]].merge(train_labels_set.iloc[:, [0, 6]].set_index('game_session'), on='game_session')

train_xy_ass['installation_id'] = train_xy_ass.installation_id.cat.remove_unused_categories() 
cross_world_group = pd.crosstab(train_xy_ass["accuracy_group"],train_xy_ass["world"])
cross_world_group
x = [[0,1,2]]*4

y = [[0]*3,[1]*3,[2]*3,[3]*3]

plt.scatter(x,y,data=train_xy_ass, s=cross_world_group/100)

plt.ylabel('accuracy groups')

plt.xticks([0,1,2], ['Crystal caves', 'Magmapeak', 'Treetopcity'], rotation=45, horizontalalignment='right')

plt.yticks([0,1,2,3],[0,1,2,3])

plt.title('Predominance of accuracy group per assessment world')

plt.show()
train_xy_ass.groupby('installation_id').accuracy_group.nunique().hist()

plt.xticks([1,2,3,4],[1,2,3,4])

plt.xlabel('Number of distinct accuracy group')

plt.ylabel('Number of installation_id')

plt.title('Histogramme of the number of distinct accuracy group per installation_id')
pd.options.display.min_rows = 40

train_xy_ass.groupby(['installation_id', 'accuracy_group']).agg({'game_session':'first', 'timestamp': 'max'})
nb_assessment = train_set.loc[train_set['type']=='Assessment']['game_session'].nunique()

nb_label = train_labels_set['game_session'].nunique()

print("%0.2f pourcent assessment without attempt"%(100*(nb_assessment - nb_label)/nb_assessment))
print("%0.0f game sessions mean per installation id"%(train_set.groupby(['installation_id']).game_session.nunique().mean()))
graph = train_set.groupby(['installation_id']).game_session.nunique().plot()

graph.axes.get_xaxis().set_ticks([])

plt.title('Histogramme of number of distinct game session per installation id')

plt.ylabel('number of distinct game session')
fig = sns.violinplot( y=train_set.groupby(['installation_id']).game_session.nunique())

fig.set_title('Violin plot of number of distinct game session per installation id')
print("%0.0f mean assessment per installation id"%(train_set.loc[train_set['type']=='Assessment'].groupby(['installation_id']).game_session.nunique().mean()))
graph = train_set.loc[train_set['type']=='Assessment'].groupby(['installation_id']).game_session.nunique().plot()

graph.axes.get_xaxis().set_ticks([])

plt.title('Histogramme of number of assessment per installation id')

plt.ylabel('number of assessment')
fig = sns.violinplot( y=train_set.loc[train_set['type']=='Assessment'].groupby(['installation_id']).game_session.nunique())

fig.set_title('Violin plot of number of assessment per installation id')
train_set.loc[~(train_set['game_session'].isin(train_labels_set['game_session'])) & (train_set['type']=='Assessment')]['installation_id'].nunique()
acc_no_attempt = train_labels_set.loc[train_labels_set['installation_id']

                     .isin(train_set.loc[~(train_set['game_session'].isin(train_labels_set['game_session'])) & 

              (train_set['type']=='Assessment')]['installation_id'])].groupby(['accuracy_group']).installation_id.nunique()
acc_all_attempt = train_labels_set.loc[~train_labels_set['installation_id']

                     .isin(train_set.loc[~(train_set['game_session'].isin(train_labels_set['game_session'])) & 

              (train_set['type']=='Assessment')]['installation_id'])].groupby(['accuracy_group']).installation_id.nunique()
df_group_attempt = pd.DataFrame(acc_no_attempt/acc_no_attempt.sum()*100)

df_group_attempt.columns = ['no_attempt']

col_all_attempt = acc_all_attempt/acc_all_attempt.sum()*100
df_group_attempt['all_attempt'] = col_all_attempt
df_group_attempt
x = [[0,1]]*4

y = [[0]*2,[1]*2,[2]*2,[3]*2]

plt.scatter(x,y,data=train_labels_set, s=df_group_attempt*100)

plt.ylabel('accuracy groups')

plt.xticks([0,1], ['no attempt', 'all attempt'], rotation=45, horizontalalignment='right')

plt.yticks([0,1,2,3],[0,1,2,3])

plt.title('Normalized distribution of accuracy group on installation id having assessment with or without attempt')

plt.show()
df_conting = pd.DataFrame(acc_no_attempt)

df_conting.columns = ['no_attempt']

df_conting['all_attempt'] = acc_all_attempt
df_conting
x = [[0,1]]*4

y = [[0]*2,[1]*2,[2]*2,[3]*2]

plt.scatter(x,y,data=train_labels_set, s=df_conting)

plt.ylabel('accuracy groups')

plt.xticks([0,1], ['no attempt', 'all attempt'], rotation=45, horizontalalignment='right')

plt.yticks([0,1,2,3],[0,1,2,3])

plt.title('Distribution of accuracy group on installation id having assessment with or without attempt')

plt.show()
from scipy import stats

stat, p, dof, expected = stats.chi2_contingency(df_conting)

print('dof=%d' % dof)

print(expected)

# interpret test-statistic

prob = 0.95

critical = stats.chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) >= critical:

	print('Dependent (reject H0)')

else:

	print('Independent (fail to reject H0)')

# interpret p-value

alpha = 1.0 - prob

print('significance=%.3f, p=%.3f' % (alpha, p))

if p <= alpha:

	print('Dependent (reject H0)')

else:

	print('Independent (fail to reject H0)')