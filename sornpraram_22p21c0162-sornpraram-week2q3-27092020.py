#22p21c0162_Sornpraram



import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





from scipy.cluster.hierarchy import dendrogram, complete

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

for i in df:

    df[i] = df[i].fillna(df[i].mode()[0])
df = df.drop(['name','host_name','neighbourhood','last_review','id','host_id'], axis = 1)

df
neighbourhood = pd.unique(df.neighbourhood_group).tolist()

df['neighbourhood_group'] = pd.Categorical(df['neighbourhood_group'],categories=neighbourhood)

df['neighbourhood_group'] = df['neighbourhood_group'].cat.codes

room_type = pd.unique(df.room_type).tolist()

df['room_type'] = pd.Categorical(df['room_type'],categories=room_type)

df['room_type'] = df['room_type'].cat.codes
df
col = []

for i in df:

    col.append(i)

samples = df.sample(n=200)

samples = samples.loc[:, col].values

data = complete(samples)
fig = plt.figure(figsize=(25, 8))

dendrogram(data)

plt.title('Hierarchical Clustering Dendrogram')

plt.xlabel('Rooms')

plt.show()