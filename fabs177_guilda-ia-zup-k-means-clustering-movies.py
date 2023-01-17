# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

df = pd.read_csv("../input/movies_metadata.csv")
df.head(2)
df_numeric = df[['budget','popularity','revenue','runtime','vote_average','vote_count','title']]

df_numeric.head()
# handle null values

df_numeric.isnull().sum()
df_numeric = df_numeric.dropna()
df_numeric = df_numeric[df_numeric['vote_count']>30]
df_numeric.shape
from sklearn import preprocessing

minmax_processed = preprocessing.MinMaxScaler().fit_transform(df_numeric.drop('title',axis=1))

df_numeric_scaled = pd.DataFrame(minmax_processed, index=df_numeric.index, columns=df_numeric.columns[:-1])

df_numeric_scaled.head()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)

kmeans.fit(df_numeric_scaled)
len(kmeans.labels_)
df_numeric['cluster'] = kmeans.labels_
df_numeric.head()
import pylab as pl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
plt.figure(figsize=(12,7))

axis = sns.barplot(x=np.arange(0,5,1),y=df_numeric.groupby(['cluster']).count()['budget'].values)

x=axis.set_xlabel("Cluster Number")

x=axis.set_ylabel("Number of movies")
size_array = list(df_numeric.groupby(['cluster']).count()['budget'].values)

size_array
df_numeric[df_numeric['cluster']==2].head(5)
df_numeric[df_numeric['cluster']==3].tail(5)