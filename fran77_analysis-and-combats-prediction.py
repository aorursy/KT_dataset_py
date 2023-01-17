# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')

combats = pd.read_csv('/kaggle/input/pokemon-challenge/combats.csv')

test = pd.read_csv('/kaggle/input/pokemon-challenge/tests.csv')
data.head()
len(data)
plt.figure(figsize=(12, 6))

sns.countplot(data.Generation)
data.groupby(['Generation'])['#'].count()
data['Total'] = data['HP'] + data['Attack'] + data['Defense'] + data['Sp. Atk'] + data['Sp. Def'] + data['Speed']
plt.figure(figsize=(12,6))

sns.boxplot(x="Generation", y="Total", data=data)
# The 4th generation seems to be the best (by the mean)
generations = data.groupby('Generation')[['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary']].mean()
generations = generations.reset_index()

generations.index = generations.index + 1
generations
generations.plot(x="Generation", y=['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'], kind="bar", figsize=(15, 6))
plt.figure(figsize=(12,6))

sns.lineplot(data=generations[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']])
plt.figure(figsize=(10,10))

data.groupby(['Type 1'])['#'].count().plot.pie(autopct='%1.0f%%', pctdistance=0.9, labeldistance=1.1, startangle=90)
data.groupby(['Type 1'])['#'].count().sort_values(ascending=False)
# There are a lot of Water and Normal types
type_g = data.groupby(['Generation', 'Type 1'])['#'].count().reset_index()
type_g = type_g.sort_values(['Generation', '#'])
type_g.groupby('Generation').tail(1)
# Water leads the first 3 generations then Normal for the 2 next
plt.figure(figsize=(12, 6))

sns.distplot(data.Total, hist=False)
# Strongest

data.iloc[data.Total.nlargest(10).index.values][['Name', 'Total']]
# Weakest

data.iloc[data.Total.nsmallest(10).index.values][['Name', 'Total']]
g=sns.catplot(x='Type 1', y='Total', kind='bar', data=data, size=6)

g.set_xticklabels(rotation=90)
# Best type is Dragon
data.groupby(['Type 1'])['Total'].mean().reset_index().sort_values('Total', ascending=False)
plt.figure(figsize=(10,6))

sns.countplot(data.Legendary)
plt.figure(figsize=(10,6))

sns.violinplot(x='Legendary', y='Total', data=data)
fig, ax = plt.subplots(figsize=(9, 7))

# Draw the two density plots

legendary = data[data.Legendary == True]

not_legendary = data[data.Legendary == False]



ax = sns.kdeplot(legendary.Attack, legendary.Defense,

                 cmap="Reds", shade=True, shade_lowest=False)

ax = sns.kdeplot(not_legendary.Attack, not_legendary.Defense,

                 cmap="Blues", shade=True, shade_lowest=False)



# Add labels to the plot

red = sns.color_palette("Reds")[-2]

blue = sns.color_palette("Blues")[-2]

ax.text(25, 0, "Non Legendary", size=16, color=blue)

ax.text(125, 150, "Legendary", size=16, color=red)
combats.head()
best = combats.groupby('Winner').count().reset_index().iloc[:,:2].sort_values('First_pokemon', ascending=False)[:5]
best
best = pd.merge(best, data, left_on='Winner', right_on='#')
best[['Winner', 'Name']]
worst = combats.groupby('Winner').count().reset_index().iloc[:,:2].sort_values('First_pokemon', ascending=True)[:5]
worst
worst = pd.merge(worst, data, left_on='Winner', right_on='#')
worst[['Winner', 'Name']]
# Convert to categorical



data['Type 1'] = data['Type 1'].astype('category').cat.codes

data['Type 2'] = data['Type 2'].astype('category').cat.codes

data['Legendary'] = data['Legendary'].astype('category').cat.codes
info = data[['#', 'Name']]
first = data.copy()

second = data.copy()
second.columns = ['#_s', 'Name_s', 'Type 1_s', 'Type 2_s', 'HP_s', 'Attack_s', 'Defense_s', 'Sp. Atk_s',

       'Sp. Def_s', 'Speed_s', 'Generation_s', 'Legendary_s', 'Total_s']



first.columns = ['#_f', 'Name_f', 'Type 1_f', 'Type 2_f', 'HP_f', 'Attack_f', 'Defense_f', 'Sp. Atk_f',

       'Sp. Def_f', 'Speed_f', 'Generation_f', 'Legendary_f', 'Total_f']
train = pd.merge(combats, first, left_on='First_pokemon', right_on='#_f')

train = pd.merge(train, second, left_on='Second_pokemon', right_on='#_s')
# First pokemon win ?

train['First_win'] = train['First_pokemon'] == train['Winner']

train['First_win'] = train['First_win'].astype('category').cat.codes
train
train['diff_HP'] = train['HP_f'] - train['HP_s']

train['diff_Attack'] = train['Attack_f'] - train['Attack_s']

train['diff_Defense'] = train['Defense_f'] - train['Defense_s']

train['diff_Sp. Atk'] = train['Sp. Atk_f'] - train['Sp. Atk_s']

train['diff_Sp. Def'] = train['Sp. Def_f'] - train['Sp. Def_s']

train['diff_Speed'] = train['Speed_f'] - train['Speed_s']

train['diff_Total'] = train['Total_f'] - train['Total_s']
train = train[['First_pokemon', 'Second_pokemon', 'Type 1_f', 'Type 2_f', 'Type 1_s', 'Type 2_s',

       'Legendary_f', 'Legendary_s', 'diff_HP', 'diff_Attack', 'diff_Defense',

       'diff_Sp. Atk', 'diff_Sp. Def', 'diff_Speed', 'diff_Total', 'First_win']]
train
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier
X = train.drop("First_win", axis=1)

Y = train["First_win"]
# Split 20% test, 80% train



X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=0)
# Logistic Regression



log = LogisticRegression(max_iter=100)

log.fit(X_train, Y_train)

Y_pred_log = log.predict(X_val)

acc_log = accuracy_score(Y_pred_log, Y_val)

acc_log
rf = RandomForestClassifier()



# search the best params

grid = {'n_estimators':[100,200], 'max_depth': [10,20,50]}



clf_rf = GridSearchCV(rf, grid, cv=10)

clf_rf.fit(X_train, Y_train)



Y_pred_rf = clf_rf.predict(X_val)

# get the accuracy score

acc_rf = accuracy_score(Y_pred_rf, Y_val)

print(acc_rf)
knn = KNeighborsClassifier()



# values we want to test for n_neighbors

param_grid = {'n_neighbors': np.arange(1, 20)}



clf_knn = GridSearchCV(knn, param_grid, cv=5)



#fit model to data

clf_knn.fit(X_train, Y_train)



Y_pred_knn = clf_knn.predict(X_val)

# get the accuracy score

acc_knn = accuracy_score(Y_pred_rf, Y_val)

print(acc_knn)
# LGBM Classifier



lgbm = LGBMClassifier(random_state=0)

lgbm.fit(X_train, Y_train)

Y_pred_lgbm = lgbm.predict(X_val)

acc_lgbm = accuracy_score(Y_pred_lgbm, Y_val)

acc_lgbm
clf_xgb = XGBClassifier(learning_rate=0.02, n_estimators=100, max_depth = 50)



clf_xgb.fit(X_train, Y_train)



Y_pred_xgb = clf_xgb.predict(X_val)

# get the accuracy score

acc_xgb = accuracy_score(Y_pred_xgb, Y_val)

print(acc_xgb)
# We will use the LGBM model
test = pd.merge(test, first, left_on='First_pokemon', right_on='#_f')

test = pd.merge(test, second, left_on='Second_pokemon', right_on='#_s')
test['diff_HP'] = test['HP_f'] - test['HP_s']

test['diff_Attack'] = test['Attack_f'] - test['Attack_s']

test['diff_Defense'] = test['Defense_f'] - test['Defense_s']

test['diff_Sp. Atk'] = test['Sp. Atk_f'] - test['Sp. Atk_s']

test['diff_Sp. Def'] = test['Sp. Def_f'] - test['Sp. Def_s']

test['diff_Speed'] = test['Speed_f'] - test['Speed_s']

test['diff_Total'] = test['Total_f'] - test['Total_s']
test = test[['First_pokemon', 'Second_pokemon', 'Type 1_f', 'Type 2_f', 'Type 1_s', 'Type 2_s',

       'Legendary_f', 'Legendary_s', 'diff_HP', 'diff_Attack', 'diff_Defense',

       'diff_Sp. Atk', 'diff_Sp. Def', 'diff_Speed', 'diff_Total']]
test
X = train.drop("First_win", axis=1)

Y = train["First_win"]
lgbm = LGBMClassifier(random_state=0)

lgbm.fit(X, Y)



Y_test = lgbm.predict(test)
test = test[['First_pokemon', 'Second_pokemon']]

test['First_Win'] = Y_test
test.head(20)
# With the names

f = first[['#_f', 'Name_f']]

s = second[['#_s', 'Name_s']]



test = pd.merge(test, f, left_on='First_pokemon', right_on='#_f')

test = pd.merge(test, s, left_on='Second_pokemon', right_on='#_s')
test = test[['Name_f', 'Name_s', 'First_Win']]

test.columns = ['Name_First_Pokemon', 'Name_Second_Pokemon', 'First_Win']
test.head(50)