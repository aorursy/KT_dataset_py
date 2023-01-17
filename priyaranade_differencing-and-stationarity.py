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

gf = gf.set_index('timestamp', 'id')
print (np.corrcoef(train['technical_20'].values, train.y.values)[0, 1])
import matplotlib.pyplot as plt

X = gf.loc[gf.id == train.id[0]]['technical_20'].values

Y = gf.loc[gf.id == train.id[0]]['y'].values

plt.plot(X, color='r')

plt.show()
X = np.diff(X)

plt.plot(X, color='r')
X = np.diff(X)

plt.plot(X)

plt.show()
print (np.corrcoef(X, Y[2:])[0, 1])
X = gf.loc[gf.id == train.id[47]]['technical_20'].values

Y = gf.loc[gf.id == train.id[47]]['y'].values

X = np.diff(X)

X = np.diff(X)

print (np.corrcoef(X, Y[2:])[0, 1])