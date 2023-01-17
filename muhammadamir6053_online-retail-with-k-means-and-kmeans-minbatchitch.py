#importing modules

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline
#Load Dataset

TextFileReader = pd.read_csv("/kaggle/input/onlineretail/OnlineRetail.csv",engine='python', chunksize=100000)  # the number of rows per chunk



DS = []

for df in TextFileReader:

    DS.append(df)



df = pd.concat(DS,sort=False)
DS_2 = DS[0]

print (DS_2.columns)

DS_2.head(10)
#Some data Stats

DS_2.shape # Shape

DS_2.info() # information

DS_2.describe() #Summary Stastics
#Looking out for missing values and handling them

DS_2.isnull().sum()
#finding out unique variables

print("Description : ",DS_2.Description.unique())

print("Country  : ",DS_2.Country.unique())
# using Pandas function to_numeric to check if there is other value than number it will convert into Nan

DS_2['InvoiceNo'] = pd.to_numeric(DS_2['InvoiceNo'], errors='coerce')



DS_2.isnull().sum()
# using Pandas function to_numeric to check if there is other value than number it will convert into Nan



DS_2['StockCode'] = pd.to_numeric(DS_2['StockCode'], errors='coerce')

DS_2.isnull().sum()
# replace the matching strings 

DS_2['Description'] = DS_2['Description'].replace('lost', np.NaN)

DS_2.isnull().sum()
# Applying abs() to Quantity column which will change negative values to positive 

DS_2['Quantity'] = DS_2['Quantity'].abs() 
# Applying abs() to UnitPrice column which will change negative values to positive 

DS_2['UnitPrice'] = DS_2['UnitPrice'].abs() 
#Filling Values with mode in all Categorical columns which have NaN values in it

stringcols = DS_2.select_dtypes(exclude=np.number)

for cat in stringcols:

    DS_2[cat] = DS_2[cat].fillna(DS_2[cat].mode().values[0])

DS_2.isnull().sum()
#Changing Datatype of InvoiceDate to_datetime 

DS_2['InvoiceDate'] = pd.to_datetime(DS_2['InvoiceDate'])
#Filling Values with mean in all number columns which have NaN values in it

numcols = DS_2.select_dtypes(include=np.number)

for cat in numcols:

    DS_2[cat] = DS_2[cat].fillna(DS_2[cat].median())

DS_2.isnull().sum()
#label encoder

from numpy import array

from sklearn.preprocessing import LabelEncoder



label_encoder=LabelEncoder()

DS_2['DescriptionCode']=label_encoder.fit_transform(DS_2.Description)

DS_2.head()
sns.pairplot(DS_2)
#create correlation

corr = DS_2.corr(method = 'pearson')



#convert correlation to numpy array

mask = np.array(corr)



#to mask the repetitive value for each pair

mask[np.tril_indices_from(mask)] = False

fig, ax = plt.subplots(figsize = (15,12))

fig.set_size_inches(15,15)

sns.heatmap(corr, mask = mask, vmax = 0.9, square = True, annot = True)
DS_2_cluster = pd.DataFrame()

DS_2_cluster['Quantity'] = DS_2['Quantity']

DS_2_cluster['UnitPrice'] = DS_2['UnitPrice']

DS_2_cluster['DescriptionCode'] = DS_2['DescriptionCode']

DS_2_cluster.head()
BOLD = '\033[1m'

END = '\033[0m'

for col in numcols:



    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,3))

    sns.boxplot(DS_2[col], linewidth=1, ax = ax1)

    DS_2[col].hist(ax = ax2)



    plt.tight_layout()

    plt.show()

    print(BOLD+col.center(115)+END)
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))

sns.distplot(DS_2["Quantity"], ax=ax1)

sns.distplot(DS_2["UnitPrice"], ax=ax2)

sns.distplot(DS_2["DescriptionCode"], ax=ax3)

plt.tight_layout()

plt.legend()
Q1 = DS_2_cluster.quantile(0.25)

Q3 = DS_2_cluster.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
DS_2_clusters = DS_2_cluster[~((DS_2_cluster < (Q1 - 1.5 * IQR)) |(DS_2_cluster > (Q3 + 1.5 * IQR))).any(axis=1)]

DS_2_clusters
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))

sns.distplot(DS_2_clusters["Quantity"], ax=ax1)

sns.distplot(DS_2_clusters["UnitPrice"], ax=ax2)

sns.distplot(DS_2_clusters["DescriptionCode"], ax=ax3)

plt.tight_layout()
#Fit and transform

DS_2_clusters.head()
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

cluster_scaled = scaler.fit_transform(DS_2_clusters)
from sklearn.cluster import KMeans



Sum_of_squared_distances = []

K = range(1,15)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(cluster_scaled)

    Sum_of_squared_distances.append(km.inertia_)

plt.figure(figsize=(20,5))

plt.plot(K, Sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('Sum_of_squared_distances')

plt.title('Elbow Method For Optimal k')

plt.show()
from mpl_toolkits.mplot3d import Axes3D



model = KMeans(n_clusters=4)

model.fit(cluster_scaled)

kmeans_labels = model.labels_



fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')

ax = plt.axes(projection="3d")



ax.scatter3D(DS_2_clusters['DescriptionCode'],DS_2_clusters['Quantity'],DS_2_clusters['UnitPrice'],c=kmeans_labels, cmap='rainbow')



xLabel = ax.set_xlabel('DescriptionCode', linespacing=3.2)

yLabel = ax.set_ylabel('Quantity', linespacing=3.1)

zLabel = ax.set_zlabel('UnitPrice', linespacing=3.4)

print("K-Means")
DS2_clustered_kmeans = DS_2_clusters.assign(Cluster=kmeans_labels)

grouped_kmeans = DS2_clustered_kmeans.groupby(['Cluster']).mean().round(1)

grouped_kmeans
from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(n_clusters=4,random_state=0,batch_size=6,max_iter=10).fit(cluster_scaled)

kmeans_labels = kmeans.labels_



fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')

ax = plt.axes(projection="3d")



ax.scatter3D(DS_2_clusters['DescriptionCode'],DS_2_clusters['Quantity'],DS_2_clusters['UnitPrice'],c=kmeans_labels, cmap='rainbow')



xLabel = ax.set_xlabel('DescriptionCode', linespacing=3.2)

yLabel = ax.set_ylabel('Quantity', linespacing=3.1)

zLabel = ax.set_zlabel('UnitPrice', linespacing=3.4)

print("K-Means")
DS2_clustered_kmeans = DS_2_clusters.assign(Cluster=kmeans_labels)

grouped_kmeans = DS2_clustered_kmeans.groupby(['Cluster']).mean().round(1)

grouped_kmeans