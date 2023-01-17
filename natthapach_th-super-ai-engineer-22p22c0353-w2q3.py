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
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df.head()
features = df.drop(['id', 'name', 'host_id', 'host_name', 'last_review'], axis=1)

features['reviews_per_month'] = features['reviews_per_month'].fillna(0)

features = pd.get_dummies(features)

features_name = features.columns

features = features.values

features
# from sklearn.preprocessing import StandardScaler



# scaler = StandardScaler()

# features_scaled = scaler.fit_transform(features)

# features_scaled
from sklearn.cluster import AgglomerativeClustering



model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(features[:50,:])
from scipy.cluster.hierarchy import dendrogram

import matplotlib.pyplot as plt



def plot_dendrogram(model, **kwargs):



    # Children of hierarchical clustering

    children = model.children_



    # Distances between each pair of children

    # Since we don't have this information, we can use a uniform one for plotting

    distance = np.arange(children.shape[0])



    # The number of observations contained in each cluster level

    no_of_observations = np.arange(2, children.shape[0]+2)



    # Create linkage matrix and then plot the dendrogram

    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)



    # Plot the corresponding dendrogram

    dendrogram(linkage_matrix, **kwargs)

    

plt.title('Hierarchical Clustering Dendrogram')

plot_dendrogram(model, labels=model.labels_)

plt.show()