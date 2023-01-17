# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
L0_station = (1, 25)
import zipfile



path = '../input/bosch-stations-one-hot-enc-train-test/stations_one_hot_train.csv'

one_hot_stations = pd.read_csv(path)



L0_one_hot = one_hot_stations.iloc[:,L0_station[0]:L0_station[1]]



pd.options.display.max_columns = None

pd.options.display.max_rows = None

pd.options.display.max_colwidth = None
# Drop rows with all 0 for each station

L0_one_hot = L0_one_hot.loc[~(L0_one_hot==0).all(axis=1)]



print("Parts in L0:{}".format(len(L0_one_hot)))
L0_one_hot.insert(0, "Id",one_hot_stations["Id"])
L0_one_hot.head()
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from matplotlib import pyplot as plt

from sklearn.cluster import DBSCAN
column_names = L0_one_hot.columns[1:]
L0_one_hot = L0_one_hot.sample(len(L0_one_hot))
dbscan = DBSCAN()

number_of_splits = 10
from random import randint

cluster_number_list = []



for i in range(5):

    split_number = randint(1, number_of_splits)

    preds_split = dbscan.fit_predict(L0_one_hot[int((split_number-1)*len(L0_one_hot)/number_of_splits):int(split_number*len(L0_one_hot)/number_of_splits)][column_names])

    cluster_number_list.append(len(np.unique(preds_split)))
max_cluster_for_l0 = int(np.max(cluster_number_list))

print("MAX NUMBER OF CLUSTERS: {}\n".format(max_cluster_for_l0))

print(cluster_number_list)
L0_one_hot.sort_values(by=['Id'], inplace=True)
n_clusters = max_cluster_for_l0

kmeans = KMeans(n_clusters=n_clusters)

pred = kmeans.fit_predict(L0_one_hot[column_names])



pred += 1

print(kmeans.inertia_)
L0_one_hot.insert(1, "ClusterL0", pred)

L0_one_hot.sample(10)
ids_clusters = pd.DataFrame({"Id": one_hot_stations['Id'], "ClusterL0": 0})

ids_clusters.loc[L0_one_hot.index, ['ClusterL0']] = L0_one_hot['ClusterL0']

ids_clusters.to_csv("Cluster_L0_train.csv", index=False)
from random import randint
L0_one_hot[L0_one_hot["ClusterL0"] == randint(1, n_clusters)].head()
L0_one_hot[L0_one_hot["ClusterL0"] == randint(1, n_clusters)].head()
L0_one_hot[L0_one_hot["ClusterL0"] == randint(1, n_clusters)].head()
L0_one_hot[L0_one_hot["ClusterL0"] == randint(1, n_clusters)].head()
path_test = '../input/bosch-stations-one-hot-enc-train-test/stations_one_hot_test.csv'

one_hot_stations_test = pd.read_csv(path_test)



L0_one_hot_test = one_hot_stations_test.iloc[:,L0_station[0]:L0_station[1]]



# Drop rows with all 0 for each station

L0_one_hot_test = L0_one_hot_test.loc[~(L0_one_hot_test==0).all(axis=1)]



print("Parts in L0_test:{}".format(len(L0_one_hot_test)))



L0_one_hot_test.insert(0, "Id",one_hot_stations_test["Id"])
pred_test = kmeans.predict(L0_one_hot_test[column_names])

pred_test += 1



L0_one_hot_test.insert(1, "ClusterL0", pred_test)



ids_clusters_test = pd.DataFrame({"Id": one_hot_stations_test['Id'], "ClusterL0": 0})

ids_clusters_test.loc[L0_one_hot_test.index, ['ClusterL0']] = L0_one_hot_test['ClusterL0']

ids_clusters_test.to_csv("Cluster_L0_test.csv", index=False)
L0_one_hot_test[L0_one_hot_test["ClusterL0"] == 1].head(10)
L0_one_hot[L0_one_hot["ClusterL0"] == 1].head(10)
L0_one_hot_test[L0_one_hot_test["ClusterL0"] == n_clusters].head(10)
L0_one_hot[L0_one_hot["ClusterL0"] == n_clusters].head(10)