# K-Means, CURE, DBSCAN

!pip install pyclustering

import csv

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from yellowbrick.cluster import KElbowVisualizer

from sklearn import preprocessing

from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from pyclustering.cluster.cure import cure

from pyclustering.cluster import cluster_visualizer

from pyclustering.samples.definitions import SIMPLE_SAMPLES

from pyclustering.samples.definitions import FCPS_SAMPLES

from pyclustering.utils import read_sample

from pyclustering.utils import timedcall

from sklearn.cluster import DBSCAN

from sklearn.datasets import make_blobs

from mpl_toolkits import mplot3d

from sklearn import metrics

from sklearn.cluster import DBSCAN

%matplotlib inline



df=pd.read_csv('../input/ecommerce-data/data.csv',encoding='iso-8859-1')

df['InvoiceDate'] =  pd.to_datetime(df['InvoiceDate'], format='%m/%d/%Y %H:%M')
df.describe()
import matplotlib.pyplot as plt



plt.figure()

df.boxplot()
df=df.dropna().reset_index()

df = df[df.Quantity <=10000]

df = df[df.Quantity >=0]

df=df.sort_values(['Quantity'],ascending=False)

df.shape
import matplotlib.pyplot as plt



plt.figure()

df.boxplot('Quantity')
sns.set(style="darkgrid")

f, ax = plt.subplots(figsize=(25, 5))

ax = sns.countplot(x="Country", data=df)
dfuk=df[df['Country']=='United Kingdom']



dfukg = (dfuk.groupby(['CustomerID','Country'],as_index=False)

          .agg({'InvoiceNo':'nunique', 'StockCode':'nunique','UnitPrice':'mean','Quantity':'sum'}))

dfukg

dfukg.reset_index()

dfukg['avgitems']=dfukg['Quantity']/dfukg['InvoiceNo']

db=dfukg[['InvoiceNo','UnitPrice','avgitems']]



import numpy as np



i=0 

while i<=len(db)-1:

    quartile_1, quartile_3 = np.percentile(db['avgitems'], [25, 75])

    iqr = quartile_3 - quartile_1

    lower_bound = quartile_1 - (iqr *1.5 )

    upper_bound = quartile_3 + (iqr *1.5)

    

    if db.loc[i,'avgitems']> upper_bound:

        db.loc[i,'outlier']=1

    elif db.loc[i,'avgitems']< lower_bound:

        db.loc[i,'outlier']=1

    else:

        db.loc[i,'outlier']=0

    i=i+1

    

db
ax = sns.countplot(x="outlier", data=db)
db=db[db['outlier']==0]

db= db.drop(columns=['outlier'])

db
model = KMeans()

visualizer = KElbowVisualizer(model, k=(2,12))

visualizer.fit(db)  

visualizer.show()        
kmeans = KMeans(5)

kmeans.fit(db)

identified_clusters = kmeans.fit_predict(db)

data_with_clusters = db.copy()

data_with_clusters['Cluster'] = identified_clusters

print(kmeans.cluster_centers_)

print(identified_clusters)



sns.set(style="darkgrid")

f, ax = plt.subplots(figsize=(25, 5))

ax = sns.countplot(x="Cluster", data=data_with_clusters)

data_with_clusters.groupby(['Cluster']).count()

fig = plt.figure()

ax = plt.axes(projection='3d')

xline=data_with_clusters['InvoiceNo']

yline=data_with_clusters['avgitems']

zline=data_with_clusters['UnitPrice']



ax.scatter3D(xline, zline,yline,c=data_with_clusters['Cluster'])

ax.view_init(60, 60)
fig = plt.figure()

ax = plt.axes(projection='3d')

xline=data_with_clusters['InvoiceNo']

yline=data_with_clusters['avgitems']

zline=data_with_clusters['UnitPrice']



ax.scatter3D(xline, zline,yline,c=data_with_clusters['Cluster'])

ax.view_init(60, 60)
data_with_clusters[data_with_clusters['Cluster']==4]
kmeans.cluster_centers_

scaler = StandardScaler()

x_scaled=scaler.fit(db)

x_scaled = scaler.fit_transform(db)

x_scaled
model = KMeans()

visualizer = KElbowVisualizer(model, k=(1,12))

visualizer.fit(x_scaled)  

visualizer.show()     
kmeans_scaled = KMeans(4)

kmeans_scaled.fit(x_scaled)

clusters_scaled = db.copy()

clusters_scaled['cluster_pred']=kmeans_scaled.fit_predict(x_scaled)

print(identified_clusters)

sns.set(style="darkgrid")

print(kmeans.cluster_centers_)

f, ax = plt.subplots(figsize=(25, 5))

ax = sns.countplot(x="cluster_pred", data=clusters_scaled)

clusters_scaled.groupby(['cluster_pred']).count()
fig = plt.figure()

ax = plt.axes(projection='3d')

xline=clusters_scaled['InvoiceNo']

yline=clusters_scaled['avgitems']

zline=clusters_scaled['UnitPrice']



ax.scatter3D(xline, zline,yline,c=clusters_scaled['cluster_pred'])

ax.view_init(35, 60)
def template_clustering(number_clusters, path, number_represent_points=1, compression=0.5, draw=True, ccore_flag=True):

    sample = read_sample(path)

    

    cure_instance = cure(sample, number_clusters, number_represent_points, compression, ccore_flag)

    (ticks, _) = timedcall(cure_instance.process)

    

    clusters = cure_instance.get_clusters()

    representors = cure_instance.get_representors()

    means = cure_instance.get_means()

    print('clusters:',means)

    print("Sample: ", path, "\t\tExecution time: ", ticks, "\n")

    print([len(cluster) for cluster in clusters])



    if draw is True:

        visualizer = cluster_visualizer()



        visualizer.append_clusters(clusters, sample)



        for cluster_index in range(len(clusters)):

            visualizer.append_cluster_attribute(0, cluster_index, representors[cluster_index], '*', 10)

            visualizer.append_cluster_attribute(0, cluster_index, [ means[cluster_index] ], 'o')



        visualizer.show()

   







        

rec = db.to_records(index=False)

db.to_csv(r'/kaggle/working/pandas.txt', header=None, index=None, sep=' ', mode='a')

path= '/kaggle/working/pandas.txt'

template_clustering(5,path)

def template_clustering(number_clusters, path, number_represent_points=1, compression=0.5, draw=True, ccore_flag=True):

    sample = read_sample(path)

    

    cure_instance = cure(sample, number_clusters, number_represent_points, compression, ccore_flag)

    (ticks, _) = timedcall(cure_instance.process)

    

    clusters = cure_instance.get_clusters()

    representors = cure_instance.get_representors()

    means = cure_instance.get_means()

    print('clusters:',means)

    print("Sample: ", path, "\t\tExecution time: ", ticks, "\n")

    print([len(cluster) for cluster in clusters])



    if draw is True:

        visualizer = cluster_visualizer()



        visualizer.append_clusters(clusters, sample)



        for cluster_index in range(len(clusters)):

            visualizer.append_cluster_attribute(0, cluster_index, representors[cluster_index], '*', 10)

            visualizer.append_cluster_attribute(0, cluster_index, [ means[cluster_index] ], 'o')



        visualizer.show()

   





dtype = [('Col1','int32'), ('Col2','float32'), ('Col3','float32')]

index = ['Row'+str(i) for i in range(1, len(x_scaled)+1)]



x_sc1 = pd.DataFrame(x_scaled, index=index)



rec = x_sc1.to_records(index=False)

x_sc1.to_csv(r'/kaggle/working/pa2ndas.txt', header=None, index=None, sep=' ', mode='a')



path= '/kaggle/working/pa2ndas.txt'

template_clustering(4,path)



stscaler = StandardScaler().fit(db)

db11 = stscaler.transform(db)

dbsc = DBSCAN(eps = .5, min_samples = 5).fit(db11)

clusters_scaled = db.copy()

clusters_scaled['cluster_pred']=dbsc.fit_predict(db11)

clusters_scaled

ax = sns.countplot(x="cluster_pred", data=clusters_scaled)

clusters_scaled.groupby(['cluster_pred']).count()

fig = plt.figure()

ax = plt.axes(projection='3d')

xline=clusters_scaled['InvoiceNo']

yline=clusters_scaled['avgitems']

zline=clusters_scaled['UnitPrice']



ax.scatter3D(xline, zline,yline,c=clusters_scaled['cluster_pred'])

ax.view_init(35, 60)
dbsc = DBSCAN(eps = .5, min_samples = 5).fit(db)

data_with_clusters = db.copy()

data_with_clusters['cluster_pred']=dbsc.fit_predict(data_with_clusters)

data_with_clusters

ax = sns.countplot(x="cluster_pred", data=data_with_clusters)

data_with_clusters.groupby(['cluster_pred']).count()

fig = plt.figure()

ax = plt.axes(projection='3d')

xline=data_with_clusters['InvoiceNo']

yline=data_with_clusters['avgitems']

zline=data_with_clusters['UnitPrice']



ax.scatter3D(xline, zline,yline,c=data_with_clusters['cluster_pred'])

ax.view_init(35, 60)
fig = plt.figure()

ax = plt.axes(projection='3d')

xline=data_with_clusters['InvoiceNo']

yline=data_with_clusters['avgitems']

zline=data_with_clusters['UnitPrice']



ax.scatter3D(xline, zline,yline,c=data_with_clusters['cluster_pred'])

ax.view_init(35, 60)