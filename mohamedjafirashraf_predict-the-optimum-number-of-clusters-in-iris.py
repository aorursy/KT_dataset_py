# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #data visulization
import seaborn as sns #advanced data visualization
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings
warnings.filterwarnings 
#load the data
df = pd.read_csv('../input/iris/Iris.csv', index_col=False)
df.set_index('Id', inplace=True)
df.head()
df.Species.value_counts()
df.info()
df.describe()
#Count the Data type 
fig = plt.figure(1, figsize=(8, 6))
df.dtypes.value_counts().plot(kind='bar')
df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].hist()
#scatter plot
fig = plt.figure(1, figsize=(8, 6))
sns.scatterplot(x='SepalLengthCm', y ='SepalWidthCm', data=df, label='Sepal')
sns.scatterplot(x='PetalLengthCm', y ='PetalWidthCm', data=df, label='Petal')
plt.xlabel('Length')
plt.ylabel('Width')
df.corr()
fig = plt.figure(1, figsize=(8, 6))
sns.heatmap(df.corr(),cmap="Blues", linewidth=0.3, cbar_kws={"shrink": .8})
sns.pairplot(df, hue='Species')
from sklearn.cluster import KMeans

x = df.iloc[:, [0, 1, 2, 3]].values

sse = []
k_rng = range(1,11)
for k in k_rng:
    km = KMeans(n_clusters=k, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km.fit(x)
    sse.append(km.inertia_)
fig = plt.figure(1, figsize=(8, 6))
plt.plot(k_rng, sse)
plt.xlabel('k')
plt.ylabel('Sum of Squared Error')
km = KMeans(n_clusters=3)
km_pred = km.fit_predict(x)
km_pred
km.cluster_centers_
df.head()
df['cluster'] = km_pred
fig = plt.figure(1, figsize=(8, 6))
df['cluster'].value_counts().plot(kind='bar')
plt.legend()
#2D visual of the final prediction
fig = plt.figure(1, figsize=(11, 7))
plt.scatter(x[km_pred == 0, 0], x[km_pred == 0, 1], alpha=0.7, label = 'Iris-setosa', color='yellow')
plt.scatter(x[km_pred == 1, 0], x[km_pred == 1, 1], alpha=0.7, label = 'Iris-versicolour', color='green')
plt.scatter(x[km_pred == 2, 0], x[km_pred == 2, 1], alpha=0.7, label = 'Iris-virginica', color='violet')

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:,1], s = 100, marker='*', c='black', label = 'Centroids')

plt.legend()
#3D visual of the final prediction
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()

ax.set_xlabel('Iris-setosa')
ax.set_ylabel('Iris-versicolour')
ax.set_zlabel('Iris-virginica')

ax.scatter(x[:, 0], x[:, 1], x[:, 2], x[:,3], c= km_pred.astype(np.float))