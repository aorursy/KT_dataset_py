# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv("../input/train.csv")
# Read the data
# Any results you write to the current directory are saved as output.
# X = df[['x', 'y', 'accuracy', 'time']]
# y = df['place_id']
df = df[(df.x <= 1.1) & (df.y <= 1.1) & (df.x >= 1.0) & (df.y >= 1.0)]
df['hour'] = (df['time']/60)%24
X = df.as_matrix(['x', 'y', 'hour'])
y = np.ravel(df.as_matrix(['place_id']))

clf = KNeighborsClassifier(n_neighbors=15) 
clf.fit(X,y)
y_pred = clf.predict(X)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(elev=12, azim=40)              # elevation and angle
ax.dist=12                                  # distance
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
plt.show()
