import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
import kagglegym

env = kagglegym.make()

observation = env.reset()

train = observation.train
train.fillna(0, inplace=True)
gf = train.copy(True)

gf = train.pivot('timestamp', 'id', 'technical_20')

y = train.pivot('timestamp', 'id', 'y')

gf.fillna(0, inplace=True)

y.fillna(0, inplace=True)
print (np.corrcoef(gf[train.id[0]].values, y[train.id[0]].values)[0, 1])
import matplotlib.pyplot as plt



X = gf[train.id[0]].values

Y = y[train.id[0]].values

plt.plot(X, color='r')

plt.show()
X = np.diff(X)

plt.plot(X, color='r')
X = np.diff(X)

plt.plot(X)

plt.show()
print (np.corrcoef(X, Y[2:])[0, 1])
print(np.corrcoef(gf[train.id[47]].diff().fillna(0).diff().fillna(0).values, y[train.id[47]].values)[0, 1])