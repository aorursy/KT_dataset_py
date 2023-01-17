# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

data_set = pd.read_csv('../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')

data_set.head()
data_set.describe()
data_set.info()
plt.figure(figsize = (30,30))

sns.heatmap(data_set.corr(),annot = True,cmap = plt.cm.Blues)

plt.show()
data_set = data_set.drop(['gameId','redKills','redDeaths','blueGoldDiff','redGoldDiff','blueExperienceDiff','redExperienceDiff','blueCSPerMin','redCSPerMin','blueAssists','redAssists','blueWardsPlaced','redWardsPlaced','blueWardsDestroyed','redWardsDestroyed','redFirstBlood','blueAvgLevel',

                          'redAvgLevel','blueHeralds','redHeralds','blueTotalJungleMinionsKilled',

                          'redTotalJungleMinionsKilled','blueTowersDestroyed','redTowersDestroyed'],axis = 1)
plt.figure(figsize = (15,15))

sns.heatmap(data_set.corr(),annot = True,cmap = plt.cm.Blues)

plt.show()
grid = sns.PairGrid(data=data_set, vars=['blueKills', 'blueTotalExperience', 'blueTotalGold','blueDeaths','blueTotalMinionsKilled'], hue='blueWins', height=5, palette='Set1')

grid.map_diag(plt.hist)

grid.map_offdiag(plt.scatter)
grid1 = sns.PairGrid(data=data_set, vars=['blueGoldPerMin', 'blueTotalExperience', 'blueTotalGold'], hue='blueKills', height=3, palette='Set1')

grid1.map_diag(plt.hist)

grid1.map_offdiag(plt.scatter)
data_set
y = data_set.iloc[:,0].values

x = data_set.iloc[:,1:].values
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)

x_test
from xgboost import XGBClassifier

classifier = XGBClassifier(n_estimators = 300,learning_rate = 0.1)

classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score



cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)

print(cm,"\n accuracy:",accuracy)
from sklearn.decomposition import PCA

pca = PCA(n_components = 1)

reduction = pca.fit_transform(x_train)

reduction_test = pca.transform(x_test)

print(pca.explained_variance_ratio_)
data_reduced = pd.DataFrame(data = reduction,columns = ['pc1'])
data_reduced
reduction_test
x_reduced = data_reduced.iloc[:,:].values
reduced_class = XGBClassifier(n_estimators = 300,learning_rate = 0.01)

reduced_class.fit(x_reduced,y_train)

y_reduced_pred = reduced_class.predict(reduction_test)
from sklearn.metrics import confusion_matrix,accuracy_score



cm = confusion_matrix(y_test, y_reduced_pred)

accuracy = accuracy_score(y_test, y_reduced_pred)

print(cm,"\n accuracy:",accuracy)