# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import warnings



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/SeventhGenPokemon3.csv", encoding='latin-1')

print(data.info())
print(data.groupby('Type2').Type2.count())
data_want = data.copy()

data_want = data_want.drop(['Normal_Dmg', 'Fire_Dmg', 'Water_Dmg', 'Eletric_Dmg', 'Grass_Dmg', 'Ice_Dmg', 

                            'Fight_Dmg', 'Poison_Dmg', 'Ground_Dmg', 'Flying_Dmg', 'Psychic_Dmg', 'Bug_Dmg', 

                            'Rock_Dmg', 'Ghost_Dmg', 'Dragon_Dmg', 'Dark_Dmg', 'Steel_Dmg', 'Fairy_Dmg', 

                           'isAlolan', 'hasAlolan'], axis=1)

print(data_want.info())
import seaborn as sns

from pandas.plotting import scatter_matrix

import matplotlib.cm as cm

import matplotlib.pyplot as plt
corr = data_want.corr()

hm = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
bh = sns.relplot(y='Base_Total', x='Base_Happiness', size='Female_Pct', hue='Egg_Steps', col='Generation',

                 col_wrap=True, data=data_want)
bh = sns.relplot(y='Base_Total', x='Base_Happiness', size='Female_Pct', hue='Egg_Steps', col='Generation',

                 row='Legendary', data=data_want)
hgt = sns.relplot(y='Base_Total', x='Height.m.', data=data_want, hue='Generation')
hgt = sns.relplot(y='Base_Total', x='Weight.kg.', data=data_want, hue='Generation')
data_want = data_want.drop(['National', 'Mega_Evolutions', 'Region', 'Male_Pct'], axis=1)

print(data_want.info())
data_want['Exp_Speed'] = data_want['Exp_Speed'].map({'Erratic':1, 'Fast':2, 'Fluctuating':3, 'Medium':4, 'Medium Fast':5, 

                                       'Medium Slow':6, 'Slow':7})

data_want['Group1'] = data_want['Group1'].map({'Amorphous':1, 'Bug':2, 'Ditto':3, 'Dragon':4, 'Fairy':5, 'Field':6, 

                                    'Flying':7, 'Grass':8, 'Human-like':9, 'Mineral':10, 'Monster':11,

                                    'None':12, 'Water 1':13, 'Water 2':14, 'Water 3':15})

data_want['Group2'] = data_want['Group2'].map({'Amorphous':1, 'Bug':2, 'Ditto':3, 'Dragon':4, 'Fairy':5, 'Field':6, 

                                    'Flying':7, 'Grass':8, 'Human-like':9, 'Mineral':10, 'Monster':11,

                                    'None':12, 'Water 1':13, 'Water 2':14, 'Water 3':15})

data_want['Type1'] = data_want['Type1'].map({'bug':1, 'dark':2, 'dragon':3, 'electric':4, 'fairy':5, 'fighting':6,

                                            'fire':7, 'flying':8, 'ghost':9, 'grass':10, 'ground':11, 'ice':12,

                                            'normal':13, 'poison':14, 'psychic':15, 'rock':16, 'steel':17,

                                            'water':18})

data_want['Type2'] = data_want['Type2'].map({'bug':1, 'dark':2, 'dragon':3, 'electric':4, 'fairy':5, 'fighting':6,

                                            'fire':7, 'flying':8, 'ghost':9, 'grass':10, 'ground':11, 'ice':12,

                                            'normal':13, 'poison':14, 'psychic':15, 'rock':16, 'steel':17,

                                            'water':18})

data_want['Capt_Rate'] = data_want['Capt_Rate'].replace('30 (Meteorite)255 (Core)', '255')

data_want['Capt_Rate'] = data_want['Capt_Rate'].astype(int)

print(data_want.info())
corr = data_want.corr()

plt.figure(figsize=(25, 25))

hm = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
data_want=data_want.drop(['Type1', 'Type2', 'Generation'], axis=1)

data_want=data_want.drop('Name', axis=1)
from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

from sklearn import model_selection

Y = data_want.Base_Total

X = data_want.drop('Base_Total',axis=1)

validation_size = 0.20

seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)





scaler = preprocessing.StandardScaler().fit(X_train)

pipeline = make_pipeline(preprocessing.StandardScaler(),

                         RandomForestRegressor(n_estimators=400))

hyperparameters = {'randomforestregressor__min_samples_split': [2],

                   'randomforestregressor__min_samples_leaf': [1],

                   'randomforestregressor__max_features': ['sqrt'],

                   'randomforestregressor__max_depth': [None],

                  'randomforestregressor__bootstrap': [False]}



clf = GridSearchCV(pipeline, hyperparameters, cv=10)

clf.fit(X_train, Y_train)



Y_pred = clf.predict(X_validation)

print(r2_score(Y_validation, Y_pred))

print(mean_squared_error(Y_validation, Y_pred))
