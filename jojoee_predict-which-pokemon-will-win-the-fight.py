import sys
import os.path
import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D

%matplotlib inline
plt.style.use('ggplot')
seed = 1
np.random.seed(seed)
print(sys.version)
print(matplotlib.__version__)
def line():
    print("\n----------------------------------------------------------------\n")
# pokemon_df = pd.read_csv('./input/pokemon.csv')
# combat_df = pd.read_csv('./input/combats.csv')

pokemon_df = pd.read_csv("../input/pokemon.csv")
combat_df = pd.read_csv("../input/combats.csv")
print('pokemon_df')
display(pokemon_df.head())
display(pokemon_df.describe())
display(pokemon_df.shape)
pokemon_df.info()
line()

print('combat_df')
display(combat_df.head())
display(combat_df.describe())
display(combat_df.shape)
combat_df.info()
"""
1.1 Replace missing data

from point 1.1 we saw 1 row that have missing name, we'll fill it
@see https://bulbapedia.bulbagarden.net/wiki/List_of_Pok%C3%A9mon_by_National_Pok%C3%A9dex_number
"""

display(pokemon_df.loc[pokemon_df['Name'].isnull()==True])
name = 'Primeape'
pokemon_df.loc[62, 'Name'] = name
display(pokemon_df[pokemon_df['Name']==name])
"""
1.2 Rename columns
"""

pokemon_df.columns = ['#', 'name', 'type1', 'type2', 'hp', 'atk', 'def', 'sp.atk', 'sp.def', 'speed', 'generation', 'legendary']
combat_df.columns = ['first', 'second', 'winner']

display(pokemon_df.head())
display(combat_df.head())
"""
1.3 Create new attrs
"""

# pokemon
# - total, total stat attr
pokemon_df['total'] = pokemon_df['hp'] + pokemon_df['atk'] + pokemon_df['def'] + \
    pokemon_df['sp.atk'] + pokemon_df['sp.def'] + pokemon_df['speed']

# combat
# - loser, loser Pokemon id
# - is_first_win, boolean
# - diff_stat, between first and second
no_total_dict = dict(zip(pokemon_df['#'], pokemon_df['total']))
cols = ['first', 'second', 'winner']
combat_stat_df = combat_df[cols].replace(no_total_dict)
combat_df['loser'] = combat_df.apply(lambda x: x['first'] if x['first'] !=  x['winner'] else x['second'], axis=1)
combat_df['is_first_win'] = combat_df['first'] == combat_df['winner']
combat_df['diff_stat'] = combat_stat_df['first'] - combat_stat_df['second']

print('pokemon')
display(pokemon_df.head())
line()

print('combat')
display(combat_stat_df.head())
display(combat_df.head())
"""
1.4 Create new DataFrame (fight_df)
"""

# fight_df, to see "win_ratio"
nfirsts = combat_df['first'].value_counts()
nseconds = combat_df['second'].value_counts()
nfights = nfirsts + nseconds
fight_df = pd.DataFrame({
    'nfights': nfights,
    'nwins': combat_df['winner'].value_counts()
}, columns=['nfights', 'nwins'])
fight_df['win_ratio'] = fight_df['nwins'] / fight_df['nfights']
fight_df = fight_df.sort_values(by='win_ratio')

print('fight_df example, data structure')
display(fight_df.head())
fight_df.info()
line()

print('fight_df example, check missing / incorrect data')
display(fight_df.loc[fight_df['win_ratio'].isnull() | (fight_df['win_ratio'] > 1)])
line()

print('fight_df, fill missing data with 0')
fight_df.loc[231, ['nfights', 'nwins', 'win_ratio']] = 0
display(fight_df.tail())
"""
1.5 Merge new DataFrame (fight_df) into main DataFrame
"""

print('Merge into main DataFrame, to see "win_ratio" for each Pokemon')
fight_df['#'] = fight_df.index
pokemon_fight_df = pokemon_df.copy()
win_ratio_dict = dict(zip(fight_df['#'], fight_df['win_ratio']))
pokemon_fight_df['win_ratio'] = pokemon_fight_df['#'].replace(win_ratio_dict)
display(pokemon_fight_df.head())
line()

# check Pokemon that have no fight
no_fight_pokemon_df = pokemon_fight_df.loc[(pokemon_fight_df['win_ratio'] > 1) | pokemon_fight_df['win_ratio'].isnull()]
print('Pokemon that have no fight: %d' % no_fight_pokemon_df.shape[0])
display(no_fight_pokemon_df)
line()

# interpolation on missing "win_ratio"
print('Interpolation on missing "win_ratio"')
print('for no-fight-pokemon based on "total" attr with "LinearRegression" model')
print('no_fight_pokemon_df, before interpolation')
display(no_fight_pokemon_df)

linreg = LinearRegression() # create LinearRegression model for interpolation
linreg.fit(have_fight_pokemon_df['total'].values.reshape(-1, 1), have_fight_pokemon_df['win_ratio'].values.reshape(-1, 1))
no_fight_pokemon_df['win_ratio'] = linreg.predict(no_fight_pokemon_df['total'].values.reshape(-1, 1))
print('no_fight_pokemon_df, after interpolation')
display(no_fight_pokemon_df)
line()

print('have_fight_pokemon_df, visualize data pattern')
have_fight_pokemon_df = pokemon_fight_df.loc[pokemon_fight_df['win_ratio'] <= 1] # only pokemon that have fight
sns.lmplot(x='total', y='win_ratio', data=have_fight_pokemon_df)
"""
2.1 Top 5 percent of high-stats Pokemon
"""

# top 5
n_top5_percent = round(0.05 * pokemon_df.shape[0])
top_five_df = pokemon_df.sort_values(['total'], ascending=False)[:n_top5_percent]

# print and plot
print('Number of top 5 percent: %d' % n_top5_percent)
plt.figure()
sns.barplot(x='total', y='name', data=top_five_df, estimator=sum)
top_five_df.head()
"""
2.2 In top 5 percent of highest-stats, how many legendary in there ?
"""

legendary_df = pokemon_df.loc[pokemon_df['legendary'] == True]
n_legendary = legendary_df.shape[0]

top5_legendary_df = top_five_df.loc[top_five_df['legendary'] == True]
n_top5_legendary = top5_legendary_df.shape[0]

print('Top highest stats')
print('Number of top 5%:', n_top5_percent)
print('Number of legendary in top 5%%: %d' % n_top5_legendary)
print('Percentage of legendary in top 5%%: %.2f %%' % (n_top5_legendary * 100 / n_top5_percent))
"""
2.3 Attribute distribution & correlation, pokemon that have high def-stat, also have high hp-stat ?
"""

drop_columns = ['#', 'name', 'type1', 'type2', 'generation', 'legendary']

 # hp vs def
sns.jointplot(x='hp', y='def', data=pokemon_df)

 # eacho
sns.pairplot(pokemon_df.drop(drop_columns, 1))

# corr
plt.figure(figsize=(20, 9))
sns.heatmap(pokemon_fight_df.drop(drop_columns, 1).corr(), annot=True)
plt.show()
"""
2.4 Most powerful type of pokemon based on type
"""

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
sns.barplot(x='total', y='type1', data=pokemon_df, ax=ax1)
sns.barplot(x='total', y='type2', data=pokemon_df, ax=ax2)
"""
2.5 Number of Pokemon for each type
"""

# type1 and typ2 in same graph
frame = pokemon_df.copy()
vals1 = [frame['type1'].value_counts()[key] for key in frame['type1'].value_counts().index]
vals2 = [frame['type2'].value_counts()[key] for key in frame['type1'].value_counts().index]
inds = np.arange(len(frame['type1'].value_counts().index))
width = .45
color1 = np.random.rand(3)
color2 = np.random.rand(3)
handles = [patches.Patch(color=color1, label='type1'), patches.Patch(color=color2, label='type2')]
plt.bar(inds, vals1, width, color=color1)
plt.bar(inds + width, vals2, width, color=color2)
plt.gca().set_xticklabels(frame['type1'].value_counts().index)
plt.gca().set_xticks(inds + width)
plt.xticks(rotation=90)
plt.legend(handles=handles)

# type1 and type2
f, ax = plt.subplots(2, 1, figsize=(15, 8))
sns.countplot('type1', data=pokemon_df, ax=ax[0], order=pokemon_df['type1'].value_counts().index)
sns.countplot('type2', data=pokemon_df, ax=ax[1], order=pokemon_df['type2'].value_counts().index)

# type1 and type2 (crosstab)
# pokemon_df['type2'] = pokemon_df['type2'].fillna("None")
type_cross = pd.crosstab(pokemon_df['type1'], pokemon_df['type2'])
type_cross.plot.bar(stacked=True, figsize=(14, 4))
plt.legend(bbox_to_anchor=(0.01, 0.99), loc='upper left', ncol=5, fontsize=8, title='type2')
plt.show()
"""
2.6 Stats distribution
"""

stats = pokemon_df.dtypes[pokemon_df.dtypes=='int64'].index
stats = stats[1:]
fig = plt.figure(figsize=(13, 8))
for i, stat in enumerate(stats):
    fig.add_subplot(3, 3, i + 1)
    plt.hist(pokemon_df[stat], bins=60)
    plt.title(stat)

print('stats', stats)
plt.show()

# only "total"
print('"total" attr')
print(pokemon_df['total'].describe())
sns.distplot(pokemon_df['total'])
"""
2.7 Stats distribution (per each type1)
"""

types = pokemon_df['type1'].unique()
stats = pokemon_df.dtypes[pokemon_df.dtypes=='int64'].index
stats = stats[1:]
fig = plt.figure(figsize=(13, 15))
for j, typ in enumerate(types):
    for i, stat in enumerate(stats):
        fig.add_subplot(20, 8, (j * 8) + i + 1)
        tmp_df = pokemon_df[pokemon_df['type1']==typ]
        plt.hist(tmp_df[stat], bins=10)
        if (((j * 8) + i) % 8 == 0):
            plt.ylabel(typ)
        if (j == 0):
            plt.title(stat)

print('types', types)
print('stats', stats)
plt.show()
"""
2.8 Average stat per each type1
"""

tmp_df = pokemon_df.groupby(['type1'])['total'].mean()
tmp_df.plot(kind='bar')
"""
2.9 Average stat per each generation
"""

pokemon_groups = pokemon_df.groupby('generation')
pokemon_groups_mean = pokemon_groups.mean()

# total
sns.pointplot(x=pokemon_groups_mean.index.values, y=pokemon_groups_mean['total'])

# other stats
fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(15, 10))
sns.pointplot(x=pokemon_groups_mean.index.values, y=pokemon_groups_mean['atk'], color='red', ax=axes[0][0])
sns.pointplot(x=pokemon_groups_mean.index.values, y=pokemon_groups_mean['def'], color='blue', ax=axes[0][1])
sns.pointplot(x=pokemon_groups_mean.index.values, y=pokemon_groups_mean['hp'], color='black', ax=axes[1][0])
sns.pointplot(x=pokemon_groups_mean.index.values, y=pokemon_groups_mean['speed'], color='green', ax=axes[1][1])
sns.pointplot(x=pokemon_groups_mean.index.values, y=pokemon_groups_mean['sp.atk'], color='orange', ax=axes[2][0])
sns.pointplot(x=pokemon_groups_mean.index.values, y=pokemon_groups_mean['sp.def'], color='purple', ax=axes[2][1])
"""
3.1 Create train data
"""

train_df = combat_df.copy()
hp_dict = dict(zip(pokemon_fight_df['#'], pokemon_fight_df['hp']))
atk_dict = dict(zip(pokemon_fight_df['#'], pokemon_fight_df['atk']))
def_dict = dict(zip(pokemon_fight_df['#'], pokemon_fight_df['def']))
sp_atk_dict = dict(zip(pokemon_fight_df['#'], pokemon_fight_df['sp.atk']))
sp_def_dict = dict(zip(pokemon_fight_df['#'], pokemon_fight_df['sp.def']))
speed_dict = dict(zip(pokemon_fight_df['#'], pokemon_fight_df['speed']))
total_dict = dict(zip(pokemon_fight_df['#'], pokemon_fight_df['total']))
win_ratio_dict = dict(zip(pokemon_fight_df['#'], pokemon_fight_df['win_ratio']))

# create attrs
train_df['first_hp'] = train_df['first'].replace(hp_dict)
train_df['first_atk'] = train_df['first'].replace(atk_dict)
train_df['first_def'] = train_df['first'].replace(def_dict)
train_df['first_sp.atk'] = train_df['first'].replace(sp_atk_dict)
train_df['first_sp.def'] = train_df['first'].replace(sp_def_dict)
train_df['first_speed'] = train_df['first'].replace(speed_dict)
train_df['first_total'] = train_df['first'].replace(total_dict)
train_df['first_win_ratio'] = train_df['first'].replace(win_ratio_dict)
train_df['second_hp'] = train_df['second'].replace(hp_dict)
train_df['second_atk'] = train_df['second'].replace(atk_dict)
train_df['second_def'] = train_df['second'].replace(def_dict)
train_df['second_sp.atk'] = train_df['second'].replace(sp_atk_dict)
train_df['second_sp.def'] = train_df['second'].replace(sp_def_dict)
train_df['second_speed'] = train_df['second'].replace(speed_dict)
train_df['second_total'] = train_df['second'].replace(total_dict)
train_df['second_win_ratio'] = train_df['second'].replace(win_ratio_dict)

# create label
train_df['is_first_win'] = train_df.apply(lambda x: 1 if x['is_first_win'] == True else 0, axis=1)

# remove no need attrs
no_need_columns = ['first', 'second', 'winner', 'loser']
train_df = train_df.drop(no_need_columns, axis=1)

# split train & test data
y = train_df['is_first_win']
drop_columns = ['is_first_win', 'first_win_ratio', 'second_win_ratio', 'diff_stat']
x = train_df.drop(drop_columns, axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=seed)

# create result
acc = {}

# display
print('train_df')
display(train_df.head())
line()

print('x_train', x_train.shape)
display(x_train.head())
line()

print('x_test', x_test.shape)
display(x_test.head())
line()

print('y_train', y_train.shape)
display(y_train.head())
line()

print('y_test', y_test.shape)
display(y_test.head())
line()

print('visualize correlation between "win_ratio" and "other attributes"')
print('we notice that "speed" is seems to be a significant attribute to win')
tmp_df = train_df.copy()
drop_columns = [c for c in tmp_df.columns if c.lower()[:6] == 'second'] + ['is_first_win', 'diff_stat']
tmp_df = tmp_df.drop(drop_columns, 1)
print(tmp_df.corr().sort_values(by='first_win_ratio', ascending=False)['first_win_ratio'])
plt.figure(figsize=(18, 8))
sns.heatmap(tmp_df.corr(), annot=True)
plt.show()
"""
3.2 Perform model & evaluation
"""
"""
model: LogisticRegression
"""

logreg = LogisticRegression()
logreg.fit(x_train, y_train)

name = 'LogisticRegression'
acc[name] = round(logreg.score(x_test, y_test) * 100, 2)
acc[name]
"""
model: kNN
"""

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

name = 'kNN'
acc[name] = round(knn.score(x_test, y_test) * 100, 2)
acc[name]
"""
model: GaussianNB
"""

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)

name = 'GaussianNB'
acc[name] = round(gaussian.score(x_test, y_test) * 100, 2)
acc[name]
"""
model: Perceptron
"""

perceptron = Perceptron()
perceptron.fit(x_train, y_train)

name = 'Perceptron'
acc[name] = round(perceptron.score(x_test, y_test) * 100, 2)
acc[name]
"""
model: DecisionTreeClassifier
"""

decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)

name = 'DecisionTreeClassifier'
acc[name] = round(decision_tree.score(x_test, y_test) * 100, 2)
acc[name]
"""
model: RandomForestClassifier
"""

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)

name = 'RandomForestClassifier'
acc[name] = round(random_forest.score(x_test, y_test) * 100, 2)
acc[name]
"""
model: Ridge
"""

clf = Ridge(alpha=1.0)
clf.fit(x_train, y_train)

name = 'Ridge'
acc[name] = round(clf.score(x_test, y_test) * 100, 2)
acc[name]
"""
model: Lasso
"""

clf = Lasso(alpha=0.1)
clf.fit(x_train, y_train)

name = 'Lasso'
acc[name] = round(clf.score(x_test, y_test) * 100, 2)
acc[name]
"""
model: LinearDiscriminantAnalysis
"""

clf = LinearDiscriminantAnalysis()
clf.fit(x_train, y_train)

name = 'LinearDiscriminantAnalysis'
acc[name] = round(clf.score(x_test, y_test) * 100, 2)
acc[name]
"""
3.3. Model summary
"""

# we got RandomForestClassifier as a winner
acc_df = pd.DataFrame(list(acc.items()), columns=['name', 'acc'])
acc_df = acc_df.sort_values(by='acc', ascending=False)
display(acc_df)

# but we want to know more about the importance features
effective = pd.DataFrame()
effective['feature_name'] = x.columns.tolist()
effective['feature_importance'] = random_forest.feature_importances_
effective = effective.sort_values(by='feature_importance', ascending=False)
display(effective)
