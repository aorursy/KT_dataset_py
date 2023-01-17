# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import copy



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pokemon = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')

combat = pd.read_csv('/kaggle/input/pokemon-challenge/combats.csv')

test = pd.read_csv('/kaggle/input/pokemon-challenge/tests.csv')
pokemon.head()
pokemon.describe(include='all').transpose()
pokemon1 = pokemon.copy()
# After going through the data i have found out that Legendary pokemon have higher chance to win.



pokemon1.Legendary = pokemon1.Legendary.astype(int)
pokemon1 = pokemon1.drop(['Name', '#'],axis=1)

pokemon1.head()
pokemon1.info()
x = pd.DataFrame(columns=pokemon1.columns.values)

y = pd.DataFrame(columns=pokemon1.columns.values)



for i in range(len(combat)-1):

    x.loc[i] = pokemon1.loc[combat.loc[i, 'First_pokemon']-1]

    y.loc[i] = pokemon1.loc[combat.loc[i, 'Second_pokemon']-1]
x.head()
x = x.rename(columns={'Type 1':'Type_1_1', 'Type 2':'Type_2_1','HP':'HP1','Attack':'Attack1','Defense':'Defense1','Sp. Atk':'Sp. Atk1','Sp. Def':'Sp. Def1','Speed':'Speed1','Generation':'Generation1','Legendary':'Legendary1'})

y = y.rename(columns={'Type 1':'Type_1_2', 'Type 2':'Type_2_2','HP':'HP2','Attack':'Attack2','Defense':'Defense2','Sp. Atk':'Sp. Atk2','Sp. Def':'Sp. Def2','Speed':'Speed2','Generation':'Generation2','Legendary':'Legendary2'})
x.head()
y.head()
final = pd.concat([x,y],axis=1)

final.head()
final['result']=np.zeros(()).astype(int)
final.head()
for i in range(len(combat)-1):

    if combat.loc[i,'First_pokemon'] == combat.loc[i,'Winner']:

        final.loc[i,'result'] = 1
final.head()
final['HP_diff']=np.zeros(()).astype(int)

final['Attack_diff']=np.zeros(()).astype(int)

final['Defense_diff']=np.zeros(()).astype(int)

final['Sp_atk_diff']=np.zeros(()).astype(int)

final['Sp_def_diff']=np.zeros(()).astype(int)

final['Speed_diff']=np.zeros(()).astype(int)
for i in range(len(final)-1):

    final.loc[i,'HP_diff']=final.loc[i,'HP1'] - final.loc[i,'HP2']

    final.loc[i,'Attack_diff']=final.loc[i,'Attack1'] - final.loc[i,'Attack2']

    final.loc[i,'Defense_diff']=final.loc[i,'Defense1'] - final.loc[i,'Defense2']

    final.loc[i,'Sp_atk_diff']=final.loc[i,'Sp. Atk1'] - final.loc[i,'Sp. Atk2']

    final.loc[i,'Sp_def_diff']=final.loc[i,'Sp. Def1'] - final.loc[i,'Sp. Def2']

    final.loc[i,'Speed_diff']=final.loc[i,'Speed1'] - final.loc[i,'Speed2']
final.head()
final1 = final.copy()
final1 = final1.drop(['HP1','HP2','Attack1','Attack2','Defense1','Defense2','Sp. Atk1','Sp. Atk2', 'Sp. Def1','Sp. Def2','Speed1','Speed2'], axis=1)

final1.head()
final2 = final1.copy()

final2 = final2.drop(['Type_1_1','Type_2_1','Type_1_2','Type_2_2'], axis=1)

final2.head()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
X = final2.drop(['result'], axis=1)

y = final2['result']
X_train, X_test, y_train, y_test = train_test_split(X,y)
logistic = LogisticRegression()

logistic.fit(X_train, y_train)

logistic.predict(X_test)

round(logistic.score(X_test, y_test)*100, 2)
random = RandomForestClassifier()

random.fit(X_train, y_train)

random.predict(X_test)

round(random.score(X_test, y_test)*100, 2)
decision = DecisionTreeClassifier()

decision.fit(X_train, y_train)

decision.predict(X_test)

round(decision.score(X_test, y_test)*100, 2)
neighbor = KNeighborsClassifier()

neighbor.fit(X_train, y_train)

neighbor.predict(X_test)

round(neighbor.score(X_test, y_test)*100, 2)