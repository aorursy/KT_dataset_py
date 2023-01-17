# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from matplotlib import pyplot as plt
import seaborn as sns; sns.set() 
import re
from sklearn.metrics import confusion_matrix
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
main_file_path = '../input/NBA 2017-2018 Data.csv'
#read data into DataFrame
data = pd.read_csv(main_file_path)
data.head()
data.tail()
data.columns = ['TEAM', 'DATE', 'HOMEADV', 'WL', 'MIN', 'PTS', 'FGM', 'FGA', 'FGPercentage',
       '3PM', '3PA', '3Percentage', 'FTM', 'FTA', 'FTPercentage', 'OREB', 'DREB', 'REB', 'AST',
       'STL', 'BLK', 'TOV', 'PF', 'PlusMinus']
data.head()
data.tail()
data.WL.replace("(W)", 1, regex=True, inplace=True)
data.WL.replace("(L)", 0, regex=True, inplace=True)

data.HOMEADV.replace("(@)", 0, regex=True, inplace=True)
data.HOMEADV.replace("(vs)", 1, regex=True, inplace=True)
data.tail()
data.isnull().sum()
data.shape
#summarize data
data.describe()
features = data[[ 'PTS', 'FGM', 'FGA', 'FGPercentage',
       '3PM', '3PA', '3Percentage', 'FTM', 'FTA', 'FTPercentage', 'OREB', 'DREB', 'REB', 'AST',
       'STL', 'BLK', 'TOV', 'PF']].values
target = data['WL'].values
features[:,3]
fig = plt.figure()
fig.suptitle('Scatter Plot for clusters')
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('FG Percentage')
ax.set_ylabel('Points')
ax.scatter(features[:,3],features[:,0])
kmeans = KMeans(n_clusters=2)
kmeans.fit(features)
y_kmeans = kmeans.predict(features)
fig = plt.figure()
fig.suptitle('Scatter Plot for clusters')
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('FG Percentage')
ax.set_ylabel('Points')
plt.scatter(features[:,3], features[:, 0], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 3], centers[:, 0], c='black', s=200, alpha=0.5);
def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(features.shape[0])[:n_clusters]
    centers = features[i]
    
    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(features, centers)
        
        # 2b. Find new centers from means of points
        new_centers = np.array([features[labels == i].mean(0)
                                for i in range(n_clusters)])
        
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    
    return centers, labels

centers, labels = find_clusters(features, 2)
plt.scatter(features[:, 3], features[:, 0], c=labels,
            s=50, cmap='viridis');
fig = plt.figure()
fig.suptitle('Scatter Plot for clusters')
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('FG Percentage')
ax.set_ylabel('Points')
enters, labels = find_clusters(features, 4, rseed=0)
plt.scatter(features[:, 3], features[:, 0], c=labels,
            s=50, cmap='viridis');
labels = KMeans(6, random_state=0).fit_predict(features)
plt.scatter(features[:, 3], features[:, 0], c=labels,
            s=50, cmap='viridis');
from sklearn.cluster import KMeans
km.fit(features)
km = KMeans(n_clusters = 2, random_state=90)
# km.cluster_centers_
print(confusion_matrix(data['WL'],km.labels_))
print(classification_report(df['WL'],km.labels_))