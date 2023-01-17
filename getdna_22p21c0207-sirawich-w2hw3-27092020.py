import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline
Data = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
Data.head()
from scipy.cluster.hierarchy import dendrogram, complete

import matplotlib.pyplot as plt

%matplotlib inline
for index in Data:

    Data[index] = Data[index].fillna(Data[index].mode()[0])

Data = Data.drop(['name','host_name','neighbourhood','last_review','id','host_id'], axis = 1)



Data.neighbourhood_group = pd.Categorical(Data.neighbourhood_group,categories=pd.unique(Data.neighbourhood_group).tolist())

Data.neighbourhood_group = Data.neighbourhood_group.cat.codes



Data.room_type = pd.Categorical(Data.room_type,categories=pd.unique(Data.room_type).tolist())

Data.room_type = Data.room_type.cat.codes



fig = plt.figure(figsize=(25, 8))

dendrogram(complete(Data.sample(n=200).loc[:, [i for i in Data]].values))

plt.title('Hierarchical Clustering Dendrogram')

plt.xlabel('Rooms')

plt.show()