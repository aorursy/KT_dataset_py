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
n_components = 8

df_pca_summary.loc[0:n_components-1]['var'].sum()
# use 2 components

pca = PCA(n_components = n_components, whiten=True)

pca.fit(only_stations)



sampled_data = one_hot_stations.sample(1183747, random_state=1)

sampled_data_pca = pca.transform(sampled_data.iloc[:,3:])



PCA_components = pd.DataFrame({"Id":sampled_data.Id , "Response":sampled_data.Response})



for i in range(n_components):

    s = "pc"+str(i)

    PCA_components[s] = sampled_data_pca[:,i]



PCA_components.sort_values(by=['Id'], inplace=True)
PCA_components.head()
from sklearn.manifold import TSNE



tsne = TSNE()



sampled_data = PCA_components.sample(100000)



tsne_t = tsne.fit_transform(sampled_data.iloc[:,2:])

tsne_components = pd.DataFrame({"Id":sampled_data.Id , "Response":sampled_data.Response, "t1":tsne_t[:,0], "t2":tsne_t[:,1]})
color = ["orange", "purple"]

label = ["working parts", "fail parts"]



plt.figure(figsize = (10, 10)) 



for each in range(2):

    plt.scatter(tsne_components.t1[tsne_components.Response == each],

                tsne_components.t2[tsne_components.Response == each],

                color = color[each],

                label=label[each])

plt.title("TSNE with {} data points".format("100000"))

plt.legend(loc = 'lower right')

plt.xlabel('t1')

plt.ylabel('t2')
tsne_components.sort_values(by=['Id'], inplace=True)
tsne_components.head(10)
n_clusters = 15

kmeans = KMeans(n_clusters=n_clusters)

kmeans.fit(tsne_components[['t1', 't2']])

print(kmeans.inertia_)
n_clusters = 50

kmeans = KMeans(n_clusters=n_clusters)

kmeans.fit(tsne_components[['t1', 't2']])

print(kmeans.inertia_)
n_clusters = 250

kmeans = KMeans(n_clusters=n_clusters)

kmeans.fit(tsne_components[['t1', 't2']])

print(kmeans.inertia_)
pred_y = kmeans.predict(tsne_components[['t1', 't2']])
tsne_components["Cluster_Numbers_from_KMeans"] = pred_y

tsne_components.sample(20)
labels=[]

for i in range(n_clusters):

    labels.append("Cluster " + str(i))
plt.figure(figsize = (10, 10)) 

for each in range(n_clusters):

    plt.scatter(tsne_components.t1[tsne_components["Cluster_Numbers_from_KMeans"] == each], 

                tsne_components.t2[tsne_components["Cluster_Numbers_from_KMeans"] == each],

                color = (np.random.random_sample(), np.random.random_sample(), np.random.random_sample()),

                label = labels[each])



plt.title("Cluster")

#plt.legend(loc="lower right")

plt.xlabel("t1")

plt.ylabel("t2")