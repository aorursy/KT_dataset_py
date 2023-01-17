# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as exp

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/unsupervised-learning-on-country-data/Country-data.csv')

df
data_dict = pd.read_csv('/kaggle/input/unsupervised-learning-on-country-data/data-dictionary.csv')

data_dict
df.corr()
fig = plt.figure(figsize=(12,10))

ax = sns.heatmap(df.corr(),annot=True,cmap = 'viridis')

plt.show()
from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler()
scaled_data = scalar.fit_transform(df.drop('country',axis = 1))
scaled_df = pd.DataFrame(data = scaled_data,columns=df.columns[1:])

scaled_df['country'] = df['country']

scaled_df
exp.histogram(data_frame=df,x = 'gdpp',nbins=167,opacity=0.75,barmode='overlay')
exp.scatter(data_frame=df,x = 'child_mort',y = 'health',color='country')
data = scaled_df.drop('country',axis = 1)


# Calculate sum of squared distances

ssd = []

K = range(1,10)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(data)

    ssd.append(km.inertia_) 
plt.figure(figsize=(10,6))

plt.plot(K, ssd, 'bx-')

plt.xlabel('k')

plt.ylabel('ssd')

plt.title('Elbow Method For Optimal k')

plt.show()
kmean = KMeans(n_clusters=3)

kmean.fit(data)
pred = kmean.labels_

print(pred)
exp.scatter(data_frame= df,x = 'gdpp',y = 'income',color=kmean.labels_)
from sklearn.decomposition import PCA



pca = PCA(n_components=2)

pca_model = pca.fit_transform(data)

data_transform = pd.DataFrame(data = pca_model, columns = ['PCA1', 'PCA2'])

data_transform['Cluster'] = pred
data_transform.head()
plt.figure(figsize=(8,8))

g = sns.scatterplot(data=data_transform, x='PCA1', y='PCA2', palette=sns.color_palette()[:3], hue='Cluster')

title = plt.title('Countries Clusters with PCA')