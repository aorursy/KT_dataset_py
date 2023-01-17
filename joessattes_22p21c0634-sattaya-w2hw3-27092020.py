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
import matplotlib.pyplot as plt

import seaborn as sns
df_airbnb = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

display(df_airbnb.head())

display(df_airbnb.shape)
# Data Exploration 

# check null value

display(df_airbnb.isnull().sum(axis = 0))



# info can't show null count

display(df_airbnb.info())

display(df_airbnb.shape)

display(df_airbnb.describe())
# feature filtering

df_airbnb = df_airbnb.drop(['id','host_name','last_review'], axis = 1) 
# Fill missing value: Assume the null value of reviews borning from no reviews in this month  

df_airbnb['reviews_per_month'] = df_airbnb['reviews_per_month'].fillna(value=0)
display(df_airbnb.isnull().sum(axis = 0))
# Clean null

df_airbnb = df_airbnb.dropna()

display(df_airbnb.info())

display(df_airbnb.shape)

display(df_airbnb.head())
# plot pair grid (density = room type)

sns.pairplot(df_airbnb, hue="room_type")
nbh_group = df_airbnb.neighbourhood_group.unique().tolist()

nbh_ungroup = df_airbnb.neighbourhood.unique().tolist()

room_type = df_airbnb.room_type.unique().tolist()

nbh1 = {}

nbh2 = {}

room = {}

for i,j in enumerate(nbh_group):

    nbh1[j] = i

for i,j in enumerate(nbh_ungroup):

    nbh2[j] = i

for i,j in enumerate(room_type):

    room[j] = i

df_airbnb['neighbourhood_group'].replace(nbh1,inplace=True)

df_airbnb['neighbourhood'].replace(nbh2,inplace=True)

df_airbnb['room_type'].replace(room,inplace=True)

df_airbnb.head()
df_airbnb.head()
# feature selection

display(df_airbnb.corr(method='spearman').style.background_gradient(cmap='coolwarm'))

plt.matshow(df_airbnb.corr(method='spearman'))

plt.show()
featureset_all = df_airbnb[['neighbourhood_group','neighbourhood','room_type','price','minimum_nights','number_of_reviews','reviews_per_month', 'latitude','calculated_host_listings_count','availability_365']]

featureset_corr = df_airbnb[['neighbourhood_group','room_type','price','number_of_reviews','reviews_per_month', 'latitude']]

featureset_ind = df_airbnb[['neighbourhood','minimum_nights','calculated_host_listings_count','availability_365']]

featureset_merge = df_airbnb[['neighbourhood_group','room_type','price','reviews_per_month', 'calculated_host_listings_count','availability_365', 'neighbourhood','minimum_nights']]
# feature Normalization

from sklearn.preprocessing import MinMaxScaler



def feat_norm(feat_in):

    feat_in = feat_in.values 

    min_max_scaler = MinMaxScaler()

    feature_norm = min_max_scaler.fit_transform(feat_in)

    plt.plot(feat_in[:10])

    plt.show()

    plt.plot(feature_norm[:10])

    plt.show()

    return feature_norm
# Sample Visualization from 3000 samples

featureset_norm_all = feat_norm(featureset_all.sample(3000))

featureset_norm_corr = feat_norm(featureset_corr.sample(3000))

featureset_norm_ind = feat_norm(featureset_ind.sample(3000))

featureset_norm_merge = feat_norm(featureset_merge.sample(3000))
from scipy.spatial import distance_matrix

from sklearn import datasets 

from sklearn.cluster import AgglomerativeClustering 

from sklearn.datasets import make_blobs 

from scipy.cluster.hierarchy import dendrogram

from matplotlib import pyplot as plt
# Select feature type for clustering: case 1: focusing on the correlation features 

feature_norm = featureset_norm_corr
agglom = AgglomerativeClustering(n_clusters = None, distance_threshold=0)

agglom = agglom.fit(feature_norm)
def plot_dendrogram(model, **kwargs):

    # Create linkage matrix and then plot the dendrogram



    # create the counts of samples under each node

    counts = np.zeros(model.children_.shape[0])

    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):

        current_count = 0

        for child_idx in merge:

            if child_idx < n_samples:

                current_count += 1  # leaf node

            else:

                current_count += counts[child_idx - n_samples]

        counts[i] = current_count



    linkage_matrix = np.column_stack([model.children_, model.distances_,

                                      counts]).astype(float)



    # Plot the corresponding dendrogram

    dendrogram(linkage_matrix, **kwargs)
plt.figure(figsize=(20,14))

plt.title('Hierarchical Clustering Dendrogram')

plot_dendrogram(agglom, truncate_mode='level', p=5)

plt.xlabel("Number of points in node (or index of point if no parenthesis).")

plt.show()