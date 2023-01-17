%matplotlib inline
#モジュールの読み込み
import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

print(os.listdir("../input"))
#train_set = pd.read_csv('../input/training_set.csv', nrows=100)
train_meta = pd.read_csv('../input/training_set_metadata.csv')
train_meta.head()
plt.scatter(train_meta.hostgal_specz,train_meta.target)
plt.show()
plt.scatter(train_meta.hostgal_photoz,train_meta.target)
plt.show()
plt.scatter(train_meta.target, (train_meta.hostgal_photoz*train_meta.hostgal_specz))
plt.show()
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train_meta.hostgal_photoz, train_meta.target, train_meta.hostgal_specz)
plt.scatter(train_meta.hostgal_photoz,train_meta.hostgal_specz)
plt.show()
plt.scatter((train_meta.ra**2+train_meta.decl**2),train_meta.target)
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train_meta.target, (train_meta.hostgal_photoz*train_meta.hostgal_specz),(train_meta.ra**2+train_meta.decl**2))
train_meta2 = train_meta[train_meta['hostgal_specz'] >0.0]
plt.scatter(train_meta2.hostgal_specz,train_meta2.target)
plt.show()
