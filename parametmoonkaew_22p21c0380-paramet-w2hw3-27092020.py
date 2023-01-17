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
#Import library



from sklearn.preprocessing  import LabelEncoder

from sklearn.preprocessing import normalize

from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.cluster.hierarchy as sch
df =pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.info()
# Sum missing values

df.isnull().sum()
#Drop unuse columns

df.drop(["id","name","host_id","host_name","last_review","room_type"],axis=1,inplace = True)
df.head()
# fill missing values in reviews_per_month with forward fill 

df["reviews_per_month"].fillna(method ='ffill', inplace = True)
df.isnull().sum()
#LabelEncoder

df['neighbourhood_group']= df[['neighbourhood_group']].apply( LabelEncoder().fit_transform)['neighbourhood_group']

df['neighbourhood']= df[['neighbourhood']].apply( LabelEncoder().fit_transform)['neighbourhood']
df.head()
#Normalize

nm = normalize(df)

nm = pd.DataFrame(nm, columns=df.columns)

nm = nm.head(n=1000)

nm = nm[["price","neighbourhood_group"]]
#Clustering

hc = AgglomerativeClustering(n_clusters = 3, affinity = "euclidean", linkage = "ward")

cluster = hc.fit_predict(nm)

nm["label"] = cluster



plt.figure(figsize = (20, 12))

plt.scatter(nm["price"][nm.label == 0], nm["neighbourhood_group"][nm.label == 0], color = "black")

plt.scatter(nm["price"][nm.label == 1], nm["neighbourhood_group"][nm.label == 1], color = "blue")

plt.scatter(nm["price"][nm.label == 2], nm["neighbourhood_group"][nm.label == 2], color = "red")

plt.xlabel("price")

plt.ylabel("availability")
#Plotting location (latitude and longtitude of airbnb host)

import seaborn as sns

sns.lmplot(data=df, x='longitude', y='latitude', hue='calculated_host_listings_count', fit_reg=False, legend=True, legend_out=True)
#Creating Cluster



import scipy.cluster.hierarchy as sch



plt.figure(figsize=(10, 8))

plt.title('Dendrograms')

plt.xlabel('Accommodation')

plt.ylabel('Euclidean distances')

dend = sch.dendrogram(sch.linkage(nm, method = 'complete'))