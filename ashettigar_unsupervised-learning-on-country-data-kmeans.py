# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from sklearn.cluster import KMeans
%matplotlib inline
%config Completer.use_jedi = False
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/unsupervised-learning-on-country-data/Country-data.csv')
df_dict = pd.read_csv('../input/unsupervised-learning-on-country-data/data-dictionary.csv')
df.head()
pd.set_option('display.max_colwidth', None)
df_dict
for columns in df.columns:
    print(columns)
    print(np.array(df[columns].head(10)))
    print('Type: ',df[columns].dtypes)
    print('Number of Null Values: ',df[columns].isnull().sum())
    print()
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),vmin=-1,vmax=1,center=0,cmap=sns.diverging_palette(240, 10, n=9),annot=True)
inertia = []
for k in range(1,10):
    kcul = KMeans(n_clusters=k)
    kmeans = kcul.fit(df[['exports','gdpp']])
    inertia.append(kmeans.inertia_)
    
fig, ax = plt.subplots()
ax.plot(range(1,10),inertia, marker = '+',color = 'red' , ls = '--', markeredgecolor = 'blue', markersize = '10')
kcul = KMeans(n_clusters=3)
kmeans = kcul.fit(df[['exports','gdpp']])
cluster = pd.DataFrame(df)
cluster['Labels'] = kmeans.labels_
cluster
Cluster_0 = cluster[cluster['Labels'] == 0]
Cluster_1 = cluster[cluster['Labels'] == 1]
Cluster_2 = cluster[cluster['Labels'] == 2]
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(1,1,1)
ax.scatter(Cluster_0['exports'],Cluster_0['gdpp'], label = 'Cluster 0', c='r')
ax.scatter(Cluster_1['exports'],Cluster_1['gdpp'], label = 'Cluster 1', c='b')
ax.scatter(Cluster_2['exports'],Cluster_2['gdpp'], label = 'Cluster 2', c='g')
ax.set_xlabel('Exports')
ax.set_ylabel('GDPP')
ax.legend()
inertia = []
for k in range(1,10):
    kcul = KMeans(n_clusters=k)
    kmeans = kcul.fit(Cluster_0[['health','income']])
    inertia.append(kmeans.inertia_)
    
fig, ax = plt.subplots()
ax.plot(range(1,10),inertia, marker = '+',color = 'red' , ls = '--', markeredgecolor = 'blue', markersize = '10')
kcul = KMeans(n_clusters=3)
kmeans = kcul.fit(Cluster_0[['health','income']])
Cluster_0.drop('Labels',axis = 1,inplace = True)
Cluster_0['Labels'] = kmeans.labels_
Cluster_0
Cluster_0_0 = Cluster_0[Cluster_0['Labels'] == 0]
Cluster_0_1 = Cluster_0[Cluster_0['Labels'] == 1]
Cluster_0_2 = Cluster_0[Cluster_0['Labels'] == 2]
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(1,1,1)
ax.scatter(Cluster_0_0['health'],Cluster_0_0['income'], label = 'Cluster 0_0', c='r')
ax.scatter(Cluster_0_1['health'],Cluster_0_1['income'], label = 'Cluster 0_1', c='b')
ax.scatter(Cluster_0_2['health'],Cluster_0_2['income'], label = 'Cluster 0_2', c='g')
ax.set_xlabel('Health')
ax.set_ylabel('Income')
ax.legend()
inertia = []
for k in range(1,10):
    kcul = KMeans(n_clusters=k)
    kmeans = kcul.fit(Cluster_0_0[['child_mort','total_fer']])
    inertia.append(kmeans.inertia_)
    
fig, ax = plt.subplots()
ax.plot(range(1,10),inertia, marker = '+',color = 'red' , ls = '--', markeredgecolor = 'blue', markersize = '10')
kcul = KMeans(n_clusters=3)
kmeans = kcul.fit(Cluster_0_0[['child_mort','total_fer']])
Cluster_0_0.drop('Labels',axis = 1,inplace = True)
Cluster_0_0['Labels'] = kmeans.labels_
Cluster_0_0
Cluster_0_0_0 = Cluster_0_0[Cluster_0_0['Labels'] == 0]
Cluster_0_0_1 = Cluster_0_0[Cluster_0_0['Labels'] == 1]
Cluster_0_0_2 = Cluster_0_0[Cluster_0_0['Labels'] == 2]
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(1,1,1)
ax.scatter(Cluster_0_0_0['child_mort'],Cluster_0_0_0['total_fer'],c = 'r',label = 'Cluster 0_0_0')
ax.scatter(Cluster_0_0_1['child_mort'],Cluster_0_0_1['total_fer'],c = 'b',label = 'Cluster 0_0_1')
ax.scatter(Cluster_0_0_2['child_mort'],Cluster_0_0_2['total_fer'],c = 'g',label = 'Cluster 0_0_2')
ax.set_xlabel('Child Mort')
ax.set_ylabel('total fertility')
ax.legend()
Cluster_0_0_1.set_index('country').sort_values(by = 'life_expec')
