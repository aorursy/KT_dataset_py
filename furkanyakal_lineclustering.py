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

L1_station = (25, 27)

L2_station = (27, 30)

L3_station = (30, 53)
import zipfile



zf = zipfile.ZipFile('../input/bosch-production-line-performance/train_numeric.csv.zip') 

train_numeric_chunks = pd.read_csv(zf.open('train_numeric.csv'), iterator=True, chunksize=100000)



path = '../input/bosch-dataset/station_one_hot.csv'

one_hot_stations = pd.read_csv(path)



L0_one_hot = one_hot_stations.iloc[:,L0_station[0]:L0_station[1]]

L1_one_hot = one_hot_stations.iloc[:,L1_station[0]:L1_station[1]]

L2_one_hot = one_hot_stations.iloc[:,L2_station[0]:L2_station[1]]

L3_one_hot = one_hot_stations.iloc[:,L3_station[0]:L3_station[1]]



pd.options.display.max_columns = None

pd.options.display.max_rows = None

pd.options.display.max_colwidth = None
# Drop rows with all 0 for each station

L0_one_hot = L0_one_hot.loc[~(L0_one_hot==0).all(axis=1)]

L1_one_hot = L1_one_hot.loc[~(L1_one_hot==0).all(axis=1)]

L2_one_hot = L2_one_hot.loc[~(L2_one_hot==0).all(axis=1)]

L3_one_hot = L3_one_hot.loc[~(L3_one_hot==0).all(axis=1)]



print("Parts in L0:{}".format(len(L0_one_hot)))

print("Parts in L1:{}".format(len(L1_one_hot)))

print("Parts in L2:{}".format(len(L2_one_hot)))

print("Parts in L3:{}".format(len(L3_one_hot)))
L0_one_hot.insert(0, "Id",one_hot_stations["Id"])

L1_one_hot.insert(0, "Id",one_hot_stations["Id"])

L2_one_hot.insert(0, "Id",one_hot_stations["Id"])

L3_one_hot.insert(0, "Id",one_hot_stations["Id"])
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



        L0_one_hot.insert(0, "Id",one_hot_stations["Id"])

L0_one_hot.insert(1, 'Response', response_column['Response'])

L1_one_hot.insert(1, 'Response', response_column['Response'])

L2_one_hot.insert(1, 'Response', response_column['Response'])

L3_one_hot.insert(1, 'Response', response_column['Response'])
L0_one_hot.head()
L1_one_hot.head()
L2_one_hot.head()
L3_one_hot.head()
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
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



L1_one_hot.insert(2, "Cluster_Numbers_from_KMeans", pred_y)
L1_one_hot.loc[L1_one_hot["Cluster_Numbers_from_KMeans"] == 0].sample(5)
L1_one_hot.loc[L1_one_hot["Cluster_Numbers_from_KMeans"] == 1].sample(5)
L1_one_hot.loc[L1_one_hot["Cluster_Numbers_from_KMeans"] == 2].sample(5)
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



L2_one_hot.insert(2, "Cluster_Numbers_from_KMeans", pred_y)
L2_one_hot.sample(20)
pc_list = []

for i in range(0, len(L0_one_hot.iloc[:,2:].columns)):

    pc_list.append('PC'+str(i))

    

pca = PCA(whiten=True).fit(L0_one_hot.iloc[:,2:])

df_pca_summary = pd.DataFrame({'var': pca.explained_variance_ratio_, 'PC':pc_list})



df_pca_summary.plot.bar(x='PC', y='var', rot=0, figsize=(25,10))

plt.xlabel("Variance explained")

plt.ylabel("Principle components")

plt.show()
df_pca_summary.loc[0:8]['var'].sum()
n_components = 9

pca = PCA(n_components = n_components, whiten=True)



sampled_data = L0_one_hot.sample(len(L0_one_hot))

sampled_data_pca = pca.fit_transform(sampled_data.iloc[:,2:])



PCA_comps = pd.DataFrame({"Id":sampled_data.Id , "Response":sampled_data.Response})

for i in range(n_components):

    s = "pc"+str(i)

    PCA_comps[s] = sampled_data_pca[:,i]

    

# PCA_comps.sort_values(by=['Id'], inplace=True)
column_names = PCA_comps.columns[2:]
'''

kmeans = KMeans(n_clusters=100)

kmeans.fit(PCA_comps[column_names])

print(kmeans.inertia_)





kmeans = KMeans(n_clusters=250)

kmeans.fit(PCA_comps[column_names])

print(kmeans.inertia_)

'''
first_split = PCA_comps[0:int(len(PCA_comps)/4)]

second_split = PCA_comps[int(len(PCA_comps)/4):int(2*len(PCA_comps)/4)]

third_split = PCA_comps[int(2*len(PCA_comps)/4):int(3*len(PCA_comps)/4)]

fourth_split = PCA_comps[int(3*len(PCA_comps)/4):]
from sklearn.cluster import DBSCAN



dbscan = DBSCAN()
preds_first_split = dbscan.fit_predict(first_split[column_names])

preds_second_split = dbscan.fit_predict(second_split[column_names])

preds_third_split = dbscan.fit_predict(third_split[column_names])

preds_fourth_split = dbscan.fit_predict(fourth_split[column_names])

print("Shape of preds_first_split: {}".format(preds_first_split.shape))

print("number of clusters in dbscan: {}".format(np.max(preds_first_split)))



print("Shape of preds_first_split: {}".format(preds_second_split.shape))

print("number of clusters in dbscan: {}".format(np.max(preds_second_split)))



print("Shape of preds_first_split: {}".format(preds_third_split.shape))

print("number of clusters in dbscan: {}".format(np.max(preds_third_split)))



print("Shape of preds_first_split: {}".format(preds_third_split.shape))

print("number of clusters in dbscan: {}".format(np.max(preds_third_split)))
first_split.insert(2, "Clusters", preds_first_split)
first_split.head(100)
cluster_label_0 = first_split.loc[first_split['Clusters'] == 0]

part_station_info_c0 = L0_one_hot.loc[L0_one_hot['Id'].isin(cluster_label_0["Id"])]

part_station_info_c0.insert(1, "Cluster", 0)
part_station_info_c0.head(100)
cluster_label_68 = first_split.loc[first_split['Clusters'] == 68]

part_station_info_c68 = L0_one_hot.loc[L0_one_hot['Id'].isin(cluster_label_68["Id"])]

part_station_info_c68.insert(1, "Cluster", 68)
part_station_info_c68.head(100)
cluster_label_43 = first_split.loc[first_split['Clusters'] == 43]

part_station_info_c43 = L0_one_hot.loc[L0_one_hot['Id'].isin(cluster_label_43["Id"])]

part_station_info_c43.insert(1, "Cluster", 43)
part_station_info_c43.head(100)
'''

from sklearn.cluster import OPTICS



optics = OPTICS(max_eps=2)

preds = optics.fit_predict(PCA_comps[column_names].sample(100000))

'''
'''

pc_list = []

for i in range(0, len(L3_one_hot.iloc[:,2:].columns)):

    pc_list.append('PC'+str(i))

    

pca = PCA(whiten=True).fit(L3_one_hot.iloc[:,2:])

df_pca_summary = pd.DataFrame({'var': pca.explained_variance_ratio_, 'PC':pc_list})



df_pca_summary.plot.bar(x='PC', y='var', rot=0, figsize=(25,10))

plt.xlabel("Variance explained")

plt.ylabel("Principle components")

plt.show()

'''
'''

df_pca_summary.loc[0:10]['var'].sum()

'''
'''

n_components = 11

pca = PCA(n_components = n_components, whiten=True)



sampled_data = L3_one_hot.sample(len(L3_one_hot))

sampled_data_pca = pca.fit_transform(sampled_data.iloc[:,2:])



PCA_comps = pd.DataFrame({"Id":sampled_data.Id , "Response":sampled_data.Response})

for i in range(n_components):

    s = "pc"+str(i)

    PCA_comps[s] = sampled_data_pca[:,i]

    

PCA_comps.sort_values(by=['Id'], inplace=True)

'''