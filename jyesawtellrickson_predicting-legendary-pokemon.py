# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import re

from sklearn.metrics import f1_score, confusion_matrix

import xgboost as xgb

pokemon = pd.read_csv('../input/pokemon/Pokemon.csv')

pokemon.head()
pokemon.Legendary.value_counts()
# Creat column for each type.

for t in set(pokemon['Type 1'].values.tolist()):

    pokemon[t] = (pokemon[['Type 1', 'Type 2']]==t).any(axis=1).fillna(0)

# Get total number of types.

pokemon['Number of Types'] = (~pokemon[['Type 1', 'Type 2']].isnull()).sum(axis=1)

# Drop types columns.

pokemon.drop(['Type 1', 'Type 2'], axis=1, inplace=True)

# Get Mega type

pokemon['Mega'] = pokemon.Name.apply(lambda x: re.search('^[a-zA-Z]+Mega [a-zA-z]+', x) is not None)

# Set index.

pokemon.set_index(['#', 'Name'], inplace=True)
pokemon.head()
y = pokemon.Legendary

X = pokemon.drop(['Legendary'], axis=1)

X_cols = pokemon.drop(['Legendary'], axis=1).columns



y_train = pokemon.query('Generation < 6').Legendary

X_train = pokemon.query('Generation < 6').drop(['Legendary'], axis=1)



y_test = pokemon.query('Generation == 6').Legendary

X_test = pokemon.query('Generation == 6').drop(['Legendary'], axis=1)
# xg_reg = xgb.XGBRegressor(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 0.1,

#                 max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg = xgb.XGBRegressor(objective ='binary:logistic')





xg_reg.fit(X_train,y_train)



preds = xg_reg.predict(X_test)



#print(f1_score(y_test, preds))
# What's the best cutoff?

from numpy import linspace

for i in linspace(0, 1, 11):

    print('Cutoff: {0:.1f}, F1 Score: {1:.3f}'.format(i, f1_score(y_test, preds > i)))
pokemon['Legendary Predicted XGB'] = xg_reg.predict(X)>=0.2



pokemon['Legendary Predicted XGB'].value_counts()

xgb.plot_importance(xg_reg)
pokemon[pokemon.Mega==True].Legendary.value_counts()
confusion_matrix(pokemon['Legendary Predicted XGB'], pokemon['Legendary'])
pokemon[(pokemon['Legendary']!=pokemon['Legendary Predicted XGB'])&(pokemon.Legendary==False)]
import matplotlib.pyplot as plt



fig = plt.figure(dpi=180)

ax = plt.subplot(1,1,1)

xgb.plot_tree(xg_reg, ax = ax)#, feature_names=list(X_cols))

 

plt.tight_layout()

#plt.savefig("tree_structure.pdf")

plt.show()
