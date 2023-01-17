import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from scipy.cluster import hierarchy
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df
df = df[['host_id','neighbourhood_group','neighbourhood','latitude','longitude','room_type','price','minimum_nights','number_of_reviews','calculated_host_listings_count','availability_365']]

df.fillna(0,inplace = True)

df.head()
col = df.columns

features = []

for i in col:

    if i == 'neighbourhood_group' or i == 'neighbourhood' or i == 'room_type':

        features.append(LabelEncoder().fit_transform(df[i]))

    else:

        features.append(df[i])

    features_array = np.array(features).T
def plot_dendrogram(n):

    link = hierarchy.linkage(features_array[:][:n],'complete')

    plt.figure(figsize = (25,15))

    dend = hierarchy.dendrogram(link)

    plt.show()
plot_dendrogram(5)
plot_dendrogram(20)
plot_dendrogram(80)