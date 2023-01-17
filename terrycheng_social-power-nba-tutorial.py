import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
nba_2017_nba_players_with_salary = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_nba_players_with_salary.csv')

nba_2017_endorsements = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_endorsements.csv')

nba_2017_player_wikipedia = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_player_wikipedia.csv')
nba_2017_nba_players_with_salary.head(20)
plt.subplots(figsize=(16,16))

ax = plt.axes()

corr = nba_2017_nba_players_with_salary.corr()

sns.heatmap(corr, square = True, xticklabels=corr.columns.values,yticklabels=corr.columns.values, annot = True)
nba_2017_nba_players_with_salary.info()
nba_2017_nba_players_with_salary.isnull().sum()
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



y = nba_2017_nba_players_with_salary['SALARY_MILLIONS']

X = nba_2017_nba_players_with_salary



X = X.drop(['Unnamed: 0', 'Rk', 'PLAYER', '3P%', 'FT%', 'TEAM', 'SALARY_MILLIONS'], axis=1)

X['POSITION'] = X['POSITION'].map({"PG":1, "SG":2, "SF":3, "PF":4, "C":5, "PF-C":6})
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.2, random_state=1)

print(train_X.shape, val_X.shape, train_y.shape, val_y.shape)
train_X
first_model = RandomForestRegressor(n_estimators=200, max_depth=2, random_state=0).fit(train_X, train_y)



#Returns the coefficient of determination R^2 of the prediction.

first_model.score(train_X, train_y)
first_model.score(val_X, val_y)
import eli5

from eli5.sklearn import PermutationImportance



# Make a small change to the code below to use in this problem. 

perm = PermutationImportance(first_model, random_state=1).fit(val_X, val_y)



# uncomment the following line to visualize your results

eli5.show_weights(perm, feature_names = val_X.columns.tolist(), top=100)
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=10, weights='uniform')

knn.fit(X, y) 
print(X[:][8:9])

print('Expected Salary:', knn.predict(X[:][8:9]))
knn.kneighbors(X, return_distance=False)
neigh =  knn.kneighbors(X, return_distance=False)



for i in neigh[8]:

    print(y[i])
print(X[:][18:19])

print('Expected Salary:', knn.predict(X[:][8:9]))
neigh =  knn.kneighbors(X, return_distance=False)



for i in neigh[18]:

    print(y[i])