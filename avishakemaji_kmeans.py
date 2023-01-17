# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt#For Visualization
import seaborn as sns#For Visualization
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/species-segmentation-using-iris-dataset/iris-dataset.csv')
df.head()
df1=df.copy()
df1.columns.values
df.isnull().sum()
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(df)
df=sc.transform(df)
df=pd.DataFrame(df,columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
  kmeans.fit(df)
  wcss.append(kmeans.inertia_)
wcss
plt.plot(range(1,11),wcss)
plt.title('The Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
kmeans=KMeans(n_clusters=3,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(df)
y_kmeans
plt.scatter(df['sepal_length'], df['sepal_width'], c=y_kmeans, cmap = 'rainbow')
plt.show()
# fig,a=plt.subplots(2,3)
for i in range(3,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    y_kmeans=kmeans.fit_predict(df)
#     print("Number of Clusters:-",i)
#     plt.subplots(3,3,i-2)
    plt.figure(figsize=(10,10))
#     cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
    sns.scatterplot(df['sepal_length'], df['sepal_width'],s=100,hue=y_kmeans,palette='Set2',)
    plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
    plt.title('Sepal width vs Sepal length')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.legend()
    plt.show()