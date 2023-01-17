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
L1_station = (25, 27)

L2_station = (27, 30)
import zipfile



zf = zipfile.ZipFile('../input/bosch-production-line-performance/train_numeric.csv.zip') 

train_numeric_chunks = pd.read_csv(zf.open('train_numeric.csv'), iterator=True, chunksize=100000)



path = '../input/bosch-stations-one-hot-enc-train-test/stations_one_hot_train.csv'

one_hot_stations = pd.read_csv(path)



L1_one_hot = one_hot_stations.iloc[:,L1_station[0]:L1_station[1]]

L2_one_hot = one_hot_stations.iloc[:,L2_station[0]:L2_station[1]]



pd.options.display.max_columns = None

pd.options.display.max_rows = None

pd.options.display.max_colwidth = None
# Drop rows with all 0 for each station

L1_one_hot = L1_one_hot.loc[~(L1_one_hot==0).all(axis=1)]

L2_one_hot = L2_one_hot.loc[~(L2_one_hot==0).all(axis=1)]





print("Parts in L1:{}".format(len(L1_one_hot)))

print("Parts in L2:{}".format(len(L2_one_hot)))
L1_one_hot.insert(0, "Id",one_hot_stations["Id"])

L2_one_hot.insert(0, "Id",one_hot_stations["Id"])
def get_numeric_frame():

    for data_frame in train_numeric_chunks:

        yield data_frame



get_df_numeric = get_numeric_frame()     

df_numeric = next(get_df_numeric)
while True:

    try:

        response_column = pd.concat([response_column, df_numeric[['Response']]])

    except:

        response_column = df_numeric[['Response']]

    try:

        df_numeric = next(get_df_numeric)

    except:

        break



L1_one_hot.insert(1, 'Response', response_column['Response'])

L2_one_hot.insert(1, 'Response', response_column['Response'])
L1_one_hot.head()
L2_one_hot.head()
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
path_test = '../input/bosch-stations-one-hot-enc-train-test/stations_one_hot_test.csv'

one_hot_stations_test = pd.read_csv(path_test)



L1_one_hot_test = one_hot_stations_test.iloc[:,L1_station[0]:L1_station[1]]

L2_one_hot_test = one_hot_stations_test.iloc[:,L2_station[0]:L2_station[1]]





# Drop rows with all 0 for each station

L1_one_hot_test = L1_one_hot_test.loc[~(L1_one_hot_test==0).all(axis=1)]

L2_one_hot_test = L2_one_hot_test.loc[~(L2_one_hot_test==0).all(axis=1)]





print("Parts in L1_test:{}".format(len(L1_one_hot_test)))

print("Parts in L2_test:{}".format(len(L2_one_hot_test)))





L1_one_hot_test.insert(0, "Id",one_hot_stations_test["Id"])

L2_one_hot_test.insert(0, "Id",one_hot_stations_test["Id"])
L1_one_hot_test.head()
L2_one_hot_test.head()
column_names = L1_one_hot.columns[2:]
inertias = []



for i in range(2, 4):

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(L1_one_hot[column_names])

    inertias.append(kmeans.inertia_)



plt.plot(range(2, 4), inertias, marker='o')

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('inertia')

plt.show()
n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters)

pred_y = kmeans.fit_predict(L1_one_hot[column_names])



pred_y += 1



L1_one_hot.insert(2, "ClusterL1", pred_y)
ids_clusters = pd.DataFrame({"Id": one_hot_stations['Id'], "ClusterL1": 0})

ids_clusters.loc[L1_one_hot.index, ['ClusterL1']] = L1_one_hot['ClusterL1']

ids_clusters.to_csv("Cluster_L1_train.csv", index=False)
L1_one_hot.loc[L1_one_hot["ClusterL1"] == 1].head(100)
L1_one_hot.loc[L1_one_hot["ClusterL1"] == 2].head(100)
L1_one_hot.loc[L1_one_hot["ClusterL1"] == 3].head(100)
pred_test = kmeans.predict(L1_one_hot_test[column_names])

pred_test += 1



L1_one_hot_test.insert(1, "ClusterL1", pred_test)



ids_clusters_test = pd.DataFrame({"Id": one_hot_stations_test['Id'], "ClusterL1": 0})

ids_clusters_test.loc[L1_one_hot_test.index, ['ClusterL1']] = L1_one_hot_test['ClusterL1']

ids_clusters_test.to_csv("Cluster_L1_test.csv", index=False)
column_names = L2_one_hot.columns[2:]
inertias = []



for i in range(4, 7):

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(L2_one_hot[column_names])

    inertias.append(kmeans.inertia_)

    

plt.plot(range(4, 7), inertias, marker='o')

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('inertia')

plt.show()
n_clusters = 6

kmeans = KMeans(n_clusters=n_clusters)

pred_y = kmeans.fit_predict(L2_one_hot[column_names])



pred_y +=1



L2_one_hot.insert(2, "ClusterL2", pred_y)
ids_clusters = pd.DataFrame({"Id": one_hot_stations['Id'], "ClusterL2": 0})

ids_clusters.loc[L2_one_hot.index, ['ClusterL2']] = L2_one_hot['ClusterL2']

ids_clusters.to_csv("Cluster_L2_train.csv", index=False)
L2_one_hot.loc[L2_one_hot["ClusterL2"] == 1].head(100)
L2_one_hot.loc[L2_one_hot["ClusterL2"] == 2].head(100)
L2_one_hot.loc[L2_one_hot["ClusterL2"] == 6].head(100)
pred_test = kmeans.predict(L2_one_hot_test[column_names])

pred_test += 1



L2_one_hot_test.insert(1, "ClusterL2", pred_test)



ids_clusters_test = pd.DataFrame({"Id": one_hot_stations_test['Id'], "ClusterL2": 0})

ids_clusters_test.loc[L2_one_hot_test.index, ['ClusterL2']] = L2_one_hot_test['ClusterL2']

ids_clusters_test.to_csv("Cluster_L2_test.csv", index=False)