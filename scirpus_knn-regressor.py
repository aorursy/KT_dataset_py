import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import KFold

from sklearn.metrics import r2_score

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
features = ['X236', 'X127', 'X267', 'X261', 'X383', 'X275', 'X311', 'X189', 'X328',

            'X104', 'X240', 'X152', 'X265', 'X276', 'X162', 'X238', 'X52', 'X117', 'X342',

            'X264', 'X316', 'X339', 'X312', 'X71', 'X77', 'X340', 'X115', 'X38', 'X341',

            'X206', 'X75', 'X203', 'X292', 'X65', 'X221', 'X151', 'X345', 'X198', 'X73',

            'X327', 'X48', 'X196', 'X310']
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

y = train.y.ravel()

train = train[features]

test = test[features]
score = 0

splits = 5

kf = KFold(n_splits=splits)



for train_index, test_index in kf.split(range(train.shape[0])):

    blind = train.loc[test_index]

    vis = train.loc[train_index]

    knn = KNeighborsRegressor(n_neighbors=100,weights='uniform',p=2)

    knn.fit(vis,y[train_index])

    score +=(r2_score(y[test_index],(knn.predict(blind))))

print(score/splits)
knn = KNeighborsRegressor(n_neighbors=100,weights='uniform',p=2)

knn.fit(train,y)

knnpreds = knn.predict(train)

print(r2_score(y,knnpreds))
_ = plt.plot(knnpreds)