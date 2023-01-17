# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn import preprocessing

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,12)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
scaler = preprocessing.MinMaxScaler()
data = pd.read_excel("/kaggle/input/lab-52-dataset/mobile.xlsx")
data
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
scaled_data
clusterizators = []
for x in range(2, 11):
    kmean = KMeans(n_clusters=x)
    kmean.fit(scaled_data)
    clusterizators.append(kmean)
inertias = [x.inertia_ for x in clusterizators]
plt.plot(range(2, 11), inertias)
clusterizator = clusterizators[3]
centroids = scaler.inverse_transform(clusterizator.cluster_centers_)
centroids
# 1 асоциальные бичи
# 2 средний пользователь, 
# 3 смсочники
# 4 любители трепаться часами, много деняк тратят на связь, презирают смски
# 5 смсочники x2
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize = (15,12))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(data["Количество SMS за месяц"], data["Количество звонков"], data["Среднемесячный расход"], c=clusterizator.labels_, alpha=0.2, zorder=-1, lw=0)
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c="red", s=150, alpha=1)
clusterizator = clusterizators[1]
centroids = scaler.inverse_transform(clusterizator.cluster_centers_)
centroids
fig = plt.figure(figsize = (15,12))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(data["Количество SMS за месяц"], data["Количество звонков"], data["Среднемесячный расход"], c=clusterizator.labels_, alpha=0.2)
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c="red", s=150, alpha=1)