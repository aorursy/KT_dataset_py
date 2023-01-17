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
%matplotlib inline

from scipy.cluster.hierarchy import dendrogram, linkage

from scipy.spatial.distance import cdist

from scipy.spatial import distance

from sklearn import preprocessing

from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import complete, fcluster, dendrogram, linkage

from scipy.spatial.distance import pdist

from matplotlib import pyplot as plt

import seaborn as sns

import matplotlib.pyplot as plt

import math
bnb = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

bnb.head()
# For making hierarchical clustering, we select significant dimentions of data.

col = ['host_id','neighbourhood_group','latitude','longitude','room_type','price','number_of_reviews','reviews_per_month']

data = bnb[col]



# clean NaN data 

data = data.fillna(0)



# For distance calculation in clustering, all dimensions need to be numeric.

# Convert categorical data to indexing number

city_list = pd.unique(bnb.neighbourhood_group).tolist()

room_list = pd.unique(bnb.room_type).tolist()



data['neighbourhood_group'] = pd.Categorical(data['neighbourhood_group'],categories=city_list)

data['neighbourhood_group'] = data['neighbourhood_group'].cat.codes



data['room_type'] = pd.Categorical(data['room_type'],categories=room_list)

data['room_type'] = data['room_type'].cat.codes

data
# ** Memory concerned **

# We randomly sampling n record 

samples = data.sample(n=100)
# min-max normalization

samples=(samples-samples.min())/(samples.max()-samples.min())
# Prepare record for distance calculation.

# Convert record samples to list.

samples = samples.loc[:, col].values
# Calculate "Hierarchical Clustering Complete-Link" (with farest Euclidean distance)

Z = complete(samples)
# ploting dendrogram of Hierarchical Clustering (Complete-Link)

fig = plt.figure(figsize=(15, 10))

dendrogram(Z)

plt.title('Dendrogram')

plt.xlabel('Rooms / Houses')

plt.ylabel('Euclidean distances')

plt.show()
# Extract sorted index from Z

n = len(Z) + 1

cache = dict()

for k in range(len(Z)):

  c1, c2 = int(Z[k][0]), int(Z[k][1])

  c1 = [c1] if c1 < n else cache.pop(c1)

  c2 = [c2] if c2 < n else cache.pop(c2)

  cache[n+k] = c1 + c2

    

index = cache[2*len(Z)]
# Sort samples by Z order

samples = samples[index]
# Plot features heatmap from sorted samples



fig, axes = plt.subplots(2, 1, figsize=(15, 15),gridspec_kw={'height_ratios': [1, 2]})



p = dendrogram(Z,ax=axes[0],no_labels=True)

axes[0].set_ylabel('Euclidean distances')



datamap = np.array(samples).T

sns.heatmap(datamap,ax=axes[1],cbar=False,xticklabels=index,yticklabels=col)

plt.xlabel("Samples")

plt.ylabel("Features")



plt.subplots_adjust(wspace=0, hspace=0)

plt.show()
