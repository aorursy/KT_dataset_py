# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
pokemon = pd.read_csv('../input/pokemon.csv')
pokemon1 = pd.read_csv('../input/pokemon.csv')
pokemon['Generation'] = pokemon1['Generation']
combats = pd.read_csv('../input/combats.csv')
pokemon.info()
column = list(pokemon.columns)
# finding the missing values
pokemon.isnull().sum()
#Pointing out the row with missing name
pokemon[pokemon['Name'].isnull()]

#Entering custom value to the Column.
pokemon.ix[62,'Name'] = 'Primeape'

#Checking for missing value in 'Name'
pokemon['Name'].isnull().any()
#Filling the missing values in 'Type 2' with 'None'

pokemon['Type 2'].fillna('None', inplace = True)

pokemon['Type 2'].isnull().sum()
pokemon["Type 1"].value_counts(dropna = False)
type1list = list(pokemon["Type 1"].drop_duplicates())
len(type1list)
pokemon["Type 2"].value_counts(dropna = False)
type2list = list(pokemon["Type 2"].drop_duplicates())
len(type2list)
for index, row in pokemon.iterrows():
    if row['Type 1'] in type2list:
        pokemon.at[index, 'Type 1'] = type2list.index(row['Type 1'])
    else :
        pokemon.at[index, 'Type 1'] = 20

for index, row in pokemon.iterrows():
    if row['Type 2'] in type2list:
        pokemon.at[index, 'Type 2'] = type2list.index(row['Type 2'])
    else :
        pokemon.at[index, 'Type 2'] = 20

pokemon = pokemon.applymap(lambda x: 1 if x == True else x)
pokemon = pokemon.applymap(lambda x: 0 if x == False else x)
pokemon.head()
pokemon['Generation'].value_counts(dropna = False)
Generation1 = list(pokemon['Generation'].value_counts(dropna = False))
for index, row in pokemon.iterrows():
    if row['Generation'] in Generation1:
        pokemon.at[index, 'Generation'] = Generation1.index(row['Type 1'])
combats = pd.read_csv('../input/combats.csv')
first_battle = combats['First_pokemon'].value_counts(dropna = False)
second_battle = combats['Second_pokemon'].value_counts(dropna = False)
winner_count = combats['Winner'].value_counts(dropna = False) 
total_battle = first_battle + second_battle
win_percentage = winner_count / total_battle

win_percentage.head()
combats.Winner[combats.Winner == combats.First_pokemon] = 0
combats.Winner[combats.Winner == combats.Second_pokemon] = 1

combats.head()
#creating dictionaries
type_df = pokemon.iloc[:, 0:4]
type_df = type_df.drop('Name', axis=1)
stats_df = pokemon.drop(['Type 1', 'Type 2', 'Name', 'Generation'], axis=1)

type_dict = type_df.set_index('#').T.to_dict('list')
stats_dict = stats_df.set_index('#').T.to_dict('list')
def replace_things(data):
    #map each battles to pokemon data
    
    data['First_pokemon_stats'] = data.First_pokemon.map(stats_dict)
    data['Second_pokemon_stats'] = data.Second_pokemon.map(stats_dict)

    data['First_pokemon'] = data.First_pokemon.map(type_dict)
    data['Second_pokemon'] = data.Second_pokemon.map(type_dict)

    return data
def calculate_stats(data):
    #calculate stats difference
    
    stats_col = ['HP_diff', 'Attack_diff', 'Defense_diff', 'Sp.Atk_diff', 'Sp.Def_diff', 'Speed_diff', 'Legendary_diff']
    diff_list = []

    for row in data.itertuples():
        diff_list.append(np.array(row.First_pokemon_stats) - np.array(row.Second_pokemon_stats))

    stats_df = pd.DataFrame(diff_list, columns=stats_col)
    data = pd.concat([data, stats_df], axis=1)
    data.drop(['First_pokemon_stats', 'Second_pokemon_stats'], axis=1, inplace=True)

    return data


combats = replace_things(combats)
combats = calculate_stats(combats)
combats = combats.drop(['First_pokemon','Second_pokemon'], axis=1)

combats.head(15)
# Preparaing the Fit-data
y_train_full = combats['Winner']
x_train_full = combats.drop('Winner', axis=1)

from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(x_train_full, y_train_full, test_size=0.25, random_state=42)
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

clf_dict = {'log reg': LogisticRegression(), 
            'naive bayes': GaussianNB(), 
            'random forest': RandomForestClassifier(n_estimators=100),
            'knn': KNeighborsClassifier(),
            'linear svc': LinearSVC(),
            'ada boost': AdaBoostClassifier(n_estimators=100),
            'gradient boosting': GradientBoostingClassifier(n_estimators=100),
            'CART': DecisionTreeClassifier()}

for name, clf in clf_dict.items():
    model = clf.fit(x_train, y_train)
    pred = model.predict(x_cv)
    print('Accuracy of {}:'.format(name), accuracy_score(pred, y_cv))
test_df = pd.read_csv('../input/tests.csv')
prediction_df = test_df.copy()
test_df = replace_things(test_df)
test_df = calculate_stats(test_df)
test_df = test_df.drop(['First_pokemon','Second_pokemon'], axis=1)
test_df.head()
classifier = RandomForestClassifier(n_estimators=100)
model = classifier.fit(x_train_full, y_train_full)
prediction = model.predict(test_df)

#prediction_df is created at the very beginning, it's the same thing as test_df before it's changed.
prediction_df['Winner'] = prediction
prediction_df['Winner'][prediction_df['Winner'] == 0] = prediction_df['First_pokemon']
prediction_df['Winner'][prediction_df['Winner'] == 1] = prediction_df['Second_pokemon']


prediction_df.to_csv('submission.csv', index=False)