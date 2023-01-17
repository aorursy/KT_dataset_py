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
import zipfile



zf = zipfile.ZipFile('../input/bosch-production-line-performance/train_numeric.csv.zip') 

train_numeric_chunks = pd.read_csv(zf.open('train_numeric.csv'), iterator=True, chunksize=100000)



path = '../input/bosch-dataset/station_one_hot.csv'

one_hot_stations = pd.read_csv(path)



pd.options.display.max_columns = None

pd.options.display.max_rows = None

pd.options.display.max_colwidth = None
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
one_hot_stations.insert(1, '#OfStations', one_hot_stations.iloc[:,1:].isin([1]).sum(1))

one_hot_stations.insert(1, 'Response', response_column['Response'])

one_hot_stations.head()
fail_parts_one_hot_stations = one_hot_stations.loc[one_hot_stations['Response'] == 1]

print("Fail parts/All parts = {}/{}".format(len(fail_parts_one_hot_stations), len(one_hot_stations)))
fail_parts_one_hot_stations.head()
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
pc_list = []

for i in range(0, 52):

    pc_list.append('PC'+str(i))

    

only_stations = one_hot_stations.iloc[:,3:]

    

pca = PCA(whiten=True).fit(only_stations)

df_pca_summary = pd.DataFrame({'var': pca.explained_variance_ratio_, 'PC':pc_list})
df_pca_summary.plot.bar(x='PC', y='var', rot=0, figsize=(25,10))

plt.xlabel("Variance explained")

plt.ylabel("Principle components")

plt.show()
df_pca_summary.loc[0:1]['var'].sum()
# use 2 components

pca = PCA(n_components = 2, whiten=True)

pca.fit(only_stations)
number_of_samples = [[10000, 100000], [500000, 1183747]]

color = ["orange", "purple"]

label = ["working parts", "fail parts"]





fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))



for i in range(2):

    for j in range(2):

        sampled_data = one_hot_stations.sample(number_of_samples[i][j], random_state=1)

        sampled_data_pca = pca.transform(sampled_data.iloc[:,3:])

        

        PCA_components = pd.DataFrame({"Id":sampled_data.Id , "Response":sampled_data.Response, "pc1":sampled_data_pca[:,0], "pc2":sampled_data_pca[:,1]})



        for each in range(2):

            axs[i][j].scatter(PCA_components.pc1[PCA_components.Response == each], PCA_components.pc2[PCA_components.Response == each], color = color[each], label=label[each])

            axs[i][j].set_title("PCA with {} data points".format(number_of_samples[i][j]))

            axs[i][j].legend(loc = 'lower right')

            axs[i][j].set_xlabel('pc1')

            axs[i][j].set_ylabel('pc2')
PCA_components.sort_values(by=['Id'], inplace=True)
PCA_components.head(10)
inertias = []



for i in range(10, 20):

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(PCA_components[['pc1', 'pc2']])

    inertias.append(kmeans.inertia_)

    



plt.plot(range(10, 20), inertias)

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('inertia')

plt.show()
inertias = []



for i in range(50, 60):

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(PCA_components[['pc1', 'pc2']])

    inertias.append(kmeans.inertia_)

    



plt.plot(range(50, 60), inertias)

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('inertia')

plt.show()
inertias = []



for i in range(65, 75):

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(PCA_components[['pc1', 'pc2']])

    inertias.append(kmeans.inertia_)

    



plt.plot(range(65, 75), inertias)

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('inertia')

plt.show()
n_clusters = 75

kmeans = KMeans(n_clusters=n_clusters)

pred_y = kmeans.fit_predict(PCA_components[['pc1', 'pc2']])
PCA_components["Cluster_Numbers_from_KMeans"] = pred_y

PCA_components.sample(20)
#cluster_colors = ['rgb(127,219,218)', 'rgb(173,228,152)','rgb(254,191,99)']

colors = np.arange(n_clusters)

labels=[]

for i in range(n_clusters):

    labels.append("Cluster " + str(i))
plt.figure(figsize = (10, 10)) 

for each in range(n_clusters):

    plt.scatter(PCA_components.pc1[PCA_components["Cluster_Numbers_from_KMeans"] == each], 

                PCA_components.pc2[PCA_components["Cluster_Numbers_from_KMeans"] == each],

                color = (np.random.random_sample(), np.random.random_sample(), np.random.random_sample()),

                label = labels[each])



plt.title("Cluster")

#plt.legend(loc="lower right")

plt.xlabel("pc1")

plt.ylabel("pc2")
id_cluster_df = PCA_components[["Id", "Cluster_Numbers_from_KMeans"]]

id_cluster_df.to_csv("Parts_id_cluster.csv")
part_station_info_c_list = []

for i in range(n_clusters):

    parts_id_of_cluster = PCA_components.loc[PCA_components['Cluster_Numbers_from_KMeans'] == i]["Id"]

    part_station_info_c_i = one_hot_stations.loc[one_hot_stations['Id'].isin(parts_id_of_cluster)]

    part_station_info_c_i.insert(1, "Cluster", i)

    part_station_info_c_list.append(part_station_info_c_i)

    part_station_info_c_i.to_csv("Cluster{}_station.csv".format(i))
part_station_info_c_list[12].head(10)
part_station_info_c_list[29].head(10)
part_station_info_c_list[37].head(10)
part_station_info_c_list[51].head(10)
part_station_info_c_list[67].head(10)