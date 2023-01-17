# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

import brewer2mpl

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

#Load datasets

train_file = '../input/train.json'

test_file = '../input/test.json'

train = pd.read_json(train_file)

test = pd.read_json(test_file)
##### Define coordinate limits of New York municipality 

long_min, long_max = (-74.2, -73.6) 

lat_min, lat_max = (40.5, 40.95)
def plot_clusters(df, n_clusters = 3, long_min=long_min, long_max=long_max, lat_min=lat_min, lat_max=lat_max):

    

    plt.figure(figsize=(12,8))



    mask_long = (df.longitude > long_min) & (df.longitude < long_max)

    mask_lat = (df.latitude > lat_min) & (df.latitude < lat_max)



    bmap = brewer2mpl.get_map('Set2', 'qualitative', n_clusters)

    colors = bmap.mpl_colors

    

    for i, cluster in enumerate(range(n_clusters)): 

        plt.scatter(df[mask_long & mask_lat].longitude[df.cluster_labels == cluster],

                    df[mask_long & mask_lat].latitude[df.cluster_labels== cluster],

                    s=10,

                    c=colors[i],

                    marker='o',

                    label='cluster '+str(cluster),

                    alpha = 0.5, 

                   )



    plt.legend()

    plt.xlim(long_min, long_max)

    plt.ylim(lat_min, lat_max)

    plt.xlabel('longitude')

    plt.ylabel('latitude')

    plt.show() 
def encode_cluster_boroughs(df):

    """Encode main NY boroughs into clusters: Manhattan, Brooklyn & Queens

    """

    if 'longitude' in df.columns :

        #Mask houses far outside NYC

        mask_long = (df.longitude > long_min) & (df.longitude < long_max)

        mask_lat = (df.latitude > lat_min) & (df.latitude < lat_max)

        

        scaled_data = StandardScaler().fit_transform(df[mask_long & mask_lat][["longitude", "latitude"]])

        

        #Did some param tweaking

        model = DBSCAN(min_samples=10, eps=0.095).fit(scaled_data)

        

        #Outside NYC cluster label as 3

        df.loc[(mask_long & mask_lat), "cluster_labels"] = model.labels_.astype(int)

        df.loc[~(mask_long & mask_lat), "cluster_labels"] = -1

        df.cluster_labels = df.cluster_labels.astype(int)



        return df 

    else:

        return None
train = encode_cluster_boroughs(train)

plot_clusters(train, n_clusters=8)
test = encode_cluster_boroughs(test)

plot_clusters(test, n_clusters=8)
plt.figure(figsize=(13,5))

ax1=plt.subplot(1,2,1)

sns.countplot(data=train[train.cluster_labels == 1], x='cluster_labels', hue='interest_level')

plt.title('Houses in Manhattan')



plt.subplot(1,2,2, sharey=ax1)

sns.countplot(data=train[train.cluster_labels == 0], x='cluster_labels', hue='interest_level')

plt.title('Houses in Brooklyn')

plt.show()
#Encode cluster labels to predictors for the training and test dataset

train['Manhattan'] = train.cluster_labels.apply(lambda s: 1 if s == 1 else 0)

test['Manhattan'] = test.cluster_labels.apply(lambda s: 1 if s == 0 else 0)



train['Brooklyn'] = train.cluster_labels.apply(lambda s: 1 if s == 0 else 0)

test['Brooklyn'] = test.cluster_labels.apply(lambda s: 1 if s == 1 else 0)



train['Queens'] = train.cluster_labels.apply(lambda s: 1 if s in [4, 6, 2] else 0)

test['Queens'] = test.cluster_labels.apply(lambda s: 1 if s in [4, 6, 2, 5] else 0)