import numpy as np 

import pandas as pd 

from sklearn.manifold import TSNE

from sklearn.metrics import r2_score

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.cm as cm

%matplotlib inline
features = ['X118',

            'X127',

            'X47',

            'X315',

            'X311',

            'X179',

            'X314',

            'X261']
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
tsne = TSNE(random_state=2016,perplexity=80)

x = tsne.fit_transform(pd.concat([train[features],test[features]]))
plt.figure(figsize=(12,12))

plt.scatter(x[:train.shape[0],0],x[:train.shape[0],1], c=train.y, cmap=cm.coolwarm)
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import KFold
score = 0

splits = 5

kf = KFold(n_splits=splits)

y = train.y.ravel()

for train_index, test_index in kf.split(range(train.shape[0])):

    blind = x[:train.shape[0]][test_index]

    vis = x[:train.shape[0]][train_index]

    knn = KNeighborsRegressor(n_neighbors=80,weights='uniform',p=2)

    knn.fit(vis,y[train_index])

    score +=(r2_score(y[test_index],(knn.predict(blind))))

print(score/splits)
score = 0

splits = 5

kf = KFold(n_splits=splits)

y = train.y.ravel()

for train_index, test_index in kf.split(range(train.shape[0])):

    blind = train[features].loc[test_index]

    vis = train[features].loc[train_index]

    knn = KNeighborsRegressor(n_neighbors=80,weights='uniform',p=2)

    knn.fit(vis,y[train_index])

    score +=(r2_score(y[test_index],(knn.predict(blind))))

print(score/splits)