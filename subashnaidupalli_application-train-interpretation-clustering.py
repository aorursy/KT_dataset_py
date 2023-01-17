import numpy as np

import pandas as pd
base_dataset= pd.read_csv("../input/application_train.csv")
base_dataset.head()
### This is wrong base_dataset_var=base_dataset.var().sort_values(ascending=False).index[0:5]

base_dataset_var=base_dataset[base_dataset.var().sort_values(ascending=False).index[0:5]]
base_dataset_var.head()
### this is not requirede base_dataset_var=base_dataset[['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED','SK_ID_CURR']]
base_dataset_var
new_base_dataset=base_dataset_var.drop('SK_ID_CURR', axis=1)

##base_dataset_var.drop('SK_ID_CURR', axis=1,inplace=True) we can write in this way
new_base_dataset.head()
### Remove null values###



for i in new_base_dataset.columns:

    new_base_dataset[i].fillna(new_base_dataset[i].median(),inplace=True)
###Normalization###

from sklearn.preprocessing import MinMaxScaler

mn=MinMaxScaler()

mn.fit(new_base_dataset)

x=mn.transform(new_base_dataset)
x
x=pd.DataFrame(x)
x.head()
new_base_dataset=x
new_base_dataset.head()
### K Elbow Method ###



from sklearn.cluster import KMeans

x1 = []

for i in range(1,20):

    kmeans = KMeans(n_clusters = i)

    kmeans.fit(new_base_dataset)

    x1.append(kmeans.inertia_) 
x1
from sklearn.cluster import KMeans

km=KMeans(n_clusters=4)

x=km.fit(new_base_dataset)

len(x.labels_)
new_base_dataset1=new_base_dataset

new_base_dataset.head()
new_base_dataset1=base_dataset[['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED']]
new_base_dataset1.head()
new_base_dataset1['cluster']=x.labels_
new_base_dataset1.head()
new_base_dataset1['cluster'].value_counts()
Test
pd.DataFrame(Test)
#### Interpretations ###
## Findout better performing clusters 
#### Agglomerative ###
from sklearn.cluster import AgglomerativeClustering

Ag=AgglomerativeClustering(n_clusters=4)

Ag.fit(new_base_dataset)

Ag.labels_
new_base_dataset.head()
new_base_dataset.shape