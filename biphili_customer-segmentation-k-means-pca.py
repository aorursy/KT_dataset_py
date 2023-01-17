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
import numpy as np

import pandas as pd

import scipy 

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
df_segmentation = pd.read_csv('../input/retail-dataset-analysis/segmentation-data.csv',index_col =0)

df_segmentation.head()
df_segmentation.describe()
df_segmentation.corr()
plt.figure(figsize = (12,9))

s = sns.heatmap(df_segmentation.corr(),

               annot = True,

               cmap = 'RdBu',

               vmin = -1,

               vmax = 1)

s.set_yticklabels(s.get_yticklabels(),rotation = 0,fontsize = 12)

s.set_xticklabels(s.get_xticklabels(),rotation =90,fontsize =12)

plt.title('Correlation Heatmap')

plt.show()
plt.figure(figsize = (12,9))

plt.scatter(df_segmentation.iloc[:,2],df_segmentation.iloc[:,4])

plt.xlabel('Age')

plt.ylabel('Income')

plt.title('Vizualization of raw data')

scaler = StandardScaler()

segmentation_std = scaler.fit_transform(df_segmentation)
wcss = []

for i in range(1,11):

    kmeans = KMeans(n_clusters =i,init ='k-means++',random_state=42)

    kmeans.fit(segmentation_std)

    wcss.append(kmeans.inertia_)
plt.figure(figsize = (10,8))

plt.plot(range(1,11),wcss,marker = 'o', linestyle = '--')

plt.xlabel('Number of Clusters')

plt.ylabel('WCSS')

plt.title('K-means Clustering');
kmeans = KMeans(n_clusters = 4, init = 'k-means++',random_state = 42)

kmeans.fit(segmentation_std)
df_segm_kmeans = df_segmentation.copy()

df_segm_kmeans['Segment K-means'] = kmeans.labels_ 
df_segm_analysis = df_segm_kmeans.groupby(['Segment K-means']).mean()

df_segm_analysis 
df_segm_analysis['N Obs'] = df_segm_kmeans[['Segment K-means','Sex']].groupby(['Segment K-means']).count()
df_segm_analysis['Prop Obs'] =df_segm_analysis['N Obs']/ df_segm_analysis['N Obs'].sum()
df_segm_analysis 
df_segm_analysis.rename({0:'Well Off',

                        1:'Fewer Opportunities',

                        2:'Standard',

                        3:'Career Focused'})
df_segm_kmeans['Labels'] = df_segm_kmeans['Segment K-means'].map({0:'Well Off',

                        1:'Fewer Opportunities',

                        2:'Standard',

                        3:'Career Focused'}) 

#df_segm_kmeans
x_axis = df_segm_kmeans['Age']

y_axis = df_segm_kmeans['Income']

plt.figure(figsize = (10,8))

sns.scatterplot(x_axis,y_axis,hue = df_segm_kmeans['Labels'],palette =['g','r','c','m']);
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(segmentation_std)
pca.explained_variance_ratio_
plt.figure(figsize = (12,9))

plt.plot(range(1,8),pca.explained_variance_ratio_.cumsum(),marker = 'o',linestyle = '--')

plt.title('Explained Variance by Components')

plt.xlabel('Number of Components')

plt.ylabel('Cumulative Explained Variance')
pca = PCA(n_components =3)
pca.fit(segmentation_std)
pca.components_
df_pca_comp = pd.DataFrame(data = pca.components_,

                          columns = df_segmentation.columns.values,

                          index = ['Component 1','Component 2','Component 3'])

df_pca_comp
plt.figure(figsize = (12,9))

sns.heatmap(df_pca_comp,

           vmin=-1,

           vmax=1,

           cmap='RdBu',

           annot=True)

plt.yticks([0,1,2],

          ['Component 1','Component 2','Component 3'],

           rotation =45,

          fontsize=9)
pca.transform(segmentation_std)
scores_pca = pca.transform(segmentation_std)
wcss = []

for i in range(1,11):

    kmeans_pca = KMeans(n_clusters =i,init ='k-means++',random_state=42)

    kmeans_pca.fit(scores_pca)

    wcss.append(kmeans_pca.inertia_)
plt.figure(figsize = (10,8))

plt.plot(range(1,11),wcss,marker = 'o', linestyle = '--')

plt.xlabel('Number of Clusters')

plt.ylabel('WCSS')

plt.title('K-means Clustering with PCA')

plt.show()
kmeans_pca = KMeans(n_clusters = 4,init ='k-means++',random_state=42)
kmeans_pca.fit(scores_pca)
df_segm_pca_kmeans = pd.concat([df_segmentation.reset_index(drop=True),pd.DataFrame(scores_pca)],axis = 1)

df_segm_pca_kmeans.columns.values[-3:] = ['Component 1','Component 2','Component 3']

df_segm_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_
#df_segm_pca_kmeans
df_segm_pca_kmeans_freq = df_segm_pca_kmeans.groupby(['Segment K-means PCA']).mean()

df_segm_pca_kmeans_freq
df_segm_pca_kmeans_freq['N Obs'] = df_segm_pca_kmeans[['Segment K-means PCA','Sex']].groupby(['Segment K-means PCA']).count()

df_segm_pca_kmeans_freq['Prop Obs'] =df_segm_pca_kmeans_freq['N Obs']/ df_segm_pca_kmeans_freq['N Obs'].sum()

df_segm_pca_kmeans_freq = df_segm_pca_kmeans_freq.rename({0:'Well Off',

                                                         1:'Fewer Opportunities',

                                                         2:'Standard',

                                                         3:'Career Focused'})

df_segm_pca_kmeans_freq 
df_segm_pca_kmeans['Legend'] = df_segm_pca_kmeans['Segment K-means PCA'].map({0:'Well Off',

                                                         1:'Fewer Opportunities',

                                                         2:'Standard',

                                                         3:'Career Focused'})
x_axis = df_segm_pca_kmeans['Component 2']

y_axis = df_segm_pca_kmeans['Component 1']

plt.figure(figsize = (10,8))

sns.scatterplot(x_axis,y_axis,hue = df_segm_pca_kmeans['Legend'],palette = ['g','r','c','m'])