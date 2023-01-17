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
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/ccdata/CC GENERAL.csv')
data.head(10)
data.shape
data.info()
data.describe().T
# Explore Missing Data 

data.isnull().sum().sort_values(ascending = False)
#check frequency values in MINIMUM_PAYMENTS

data['MINIMUM_PAYMENTS'].value_counts()
#copy data without cust_id beacause it is object not numeric to make visualize

df = data.copy()

df.drop(columns=['CUST_ID'] , axis=1 , inplace=True)



 
for col in df:

    df[[col]].hist()
fig = plt.figure(figsize=(20,20))

for col in range(len(df.columns)) :

    fig.add_subplot(6,3,col+1)

    sns.boxplot(x=df.iloc[ : , col])

plt.show()
fig = plt.figure(figsize=(12,10))

sns.heatmap(data.corr() , annot=True)
data['CREDIT_LIMIT']=data['CREDIT_LIMIT'].fillna(data['CREDIT_LIMIT'].mean())

data['MINIMUM_PAYMENTS'] = data['MINIMUM_PAYMENTS'].fillna(data['MINIMUM_PAYMENTS'].median())
data.isnull().sum().sort_values(ascending = False)
data = data.drop(data[data['PURCHASES'] > 4500].index)

data = data.drop(data[data['ONEOFF_PURCHASES'] > 3000].index)

data = data.drop(data[data['INSTALLMENTS_PURCHASES'] > 1800].index)

data = data.drop(data[data['CASH_ADVANCE'] > 3500].index) 

data = data.drop(data[data['CASH_ADVANCE_FREQUENCY'] > 1.3].index)

data = data.drop(data[data['CASH_ADVANCE_TRX'] > 55].index)

data = data.drop(data[data['PAYMENTS']>3500].index)

data = data.drop(data[data['MINIMUM_PAYMENTS'] > 4000].index)



'''

data.where(data['PURCHASES'] < 4500 , inplace=True)

data.where(data['ONEOFF_PURCHASES'] < 3000 , inplace=True)

data.where(data['INSTALLMENTS_PURCHASES'] < 1800 , inplace= True)

data.where(data['CASH_ADVANCE'] < 3500 ,inplace= True)

data.where(data['CASH_ADVANCE_FREQUENCY'] < 1.3 ,inplace=True)

data.where(data['CASH_ADVANCE_TRX'] < 55 , inplace=True)

data.where(data['PAYMENTS'] < 3500 , inplace=True)

data.where(data['MINIMUM_PAYMENTS'] < 4000 , inplace=True)

'''

data.drop(['CUST_ID'] , axis = 1 , inplace= True)
data.head()
data.dtypes
data.shape
data.isnull().sum().sort_values(ascending = False)
#from sklearn.preprocessing import StandardScaler

#sc = StandardScaler()

#from sklearn.preprocessing import RobustScaler

#sc = RobustScaler()

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

data = sc.fit_transform(data)
from sklearn.cluster import KMeans

wcss =[]

for i in range(1,11):

    kmeans = KMeans(n_clusters=i , init='k-means++' ,random_state = 42)

    kmeans.fit(data)

    wcss.append(kmeans.inertia_)

plt.plot(range(1,11) , wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of Cluster')

plt.ylabel('wcss')

plt.legend()

plt.show()
kmeans = KMeans(n_clusters=4 , init='k-means++' , random_state=42)

kmeans = kmeans.fit(data)

clusters = kmeans.predict(data)
data = pd.DataFrame(data)
data.head()
from sklearn.decomposition import PCA 

pca = PCA(n_components = 2)

reduced_data = pca.fit_transform(data)

explained_varience = pca.explained_variance_ratio_
reduced_data.shape  , data.shape , clusters.shape
reduced_data=pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])

red_data_2 = reduced_data.copy()

red_data_2 = pd.DataFrame(red_data_2 , columns=['PC1' , 'PC2'])

reduced_data.head()
reduced_data['clusters'] = clusters

reduced_data.head()
plt.figure(figsize=(8,6))

plt.scatter(reduced_data.loc[reduced_data['clusters'] == 0 , 'PC1'] , reduced_data.loc[reduced_data['clusters'] ==0 , 'PC2'] , c='r' , label='cluster 0')

plt.scatter(reduced_data.loc[reduced_data['clusters'] == 1 , 'PC1'] , reduced_data.loc[reduced_data['clusters'] ==1 , 'PC2'] , c='b' ,label= 'Cluster 1')

plt.scatter(reduced_data.loc[reduced_data['clusters'] == 2 , 'PC1'] , reduced_data.loc[reduced_data['clusters'] ==2 , 'PC2'] , c='g' , label='cluster 2')

plt.scatter(reduced_data.loc[reduced_data['clusters'] == 3 , 'PC1'] , reduced_data.loc[reduced_data['clusters'] ==3 , 'PC2'] , c='cyan' ,label= 'Cluster 3')

plt.title('Credit Card Segmentation')

plt.legend()

plt.show()



import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(data, method = 'ward'))

plt.title('Dendrogram')

plt.xlabel('Customers')

plt.ylabel('Euclidean distances')

plt.show()



# Fitting Hierarchical Clustering to the dataset

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')

y_hc = hc.fit_predict(data)
red_data_2['clusters'] = y_hc
plt.figure(figsize=(8,6))

plt.scatter(red_data_2.loc[red_data_2['clusters'] == 0 , 'PC1'] , red_data_2.loc[red_data_2['clusters'] ==0 , 'PC2'] , c='r' , label='cluster 0')

plt.scatter(red_data_2.loc[red_data_2['clusters'] == 1 , 'PC1'] , red_data_2.loc[red_data_2['clusters'] ==1 , 'PC2'] , c='b' ,label= 'Cluster 1')

plt.scatter(red_data_2.loc[red_data_2['clusters'] == 2 , 'PC1'] , red_data_2.loc[red_data_2['clusters'] ==2 , 'PC2'] , c='g' ,label= 'Cluster 2')

plt.title('Credit Card Segmentation')

plt.legend()

plt.show()