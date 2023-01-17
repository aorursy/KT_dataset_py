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
file="/kaggle/input/clustering-kmeans/Mall_Customers.csv"
data=pd.read_csv(file)
data.head()
data.describe()
data.info()
#converting Gender to a categorical variable

data=pd.get_dummies(data,columns=['Genre'])
data.columns
data.drop(['Genre_Female'],inplace=True,axis=1)
data.rename(columns={'Genre_Male':'Male'},inplace=True)
data.head()
#Observe the Distributions for the Features- Age, Annual Income (k$),Spending Score (1-100), Male



import seaborn as sns

import matplotlib.pyplot as plt



sns.distplot(data['Age'])
sns.distplot(data['Annual Income (k$)'])
sns.distplot(data["Spending Score (1-100)"])
data.columns
#Standardising 

from sklearn import preprocessing

#standardized_X = preprocessing.scale(X)

data['scaled_age'] = preprocessing.scale(data['Age'])

data['scaled_annual_income'] = preprocessing.scale(data['Annual Income (k$)'])

data['scaled_spending_score'] = preprocessing.scale(data['Spending Score (1-100)'])
from sklearn.cluster import KMeans



ssd=[]



for i in range(2, 10):

    kmeans = KMeans(n_clusters=i, random_state=0)

    kmeans.fit(data[['scaled_age','scaled_annual_income','scaled_spending_score']])

    ssd.append(kmeans.inertia_)
plt.plot(range(2, 10), ssd)

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
k=4

kmeans = KMeans(n_clusters=k, random_state=0)

result=kmeans.fit(data[['scaled_age','scaled_annual_income','scaled_spending_score']])
from sklearn.decomposition import PCA



pca=PCA(n_components=2)

data_reduced=pca.fit_transform(data[['scaled_age','scaled_annual_income','scaled_spending_score']])
centroids_reduced=pca.transform(kmeans.cluster_centers_)
data['result']=result
plt.figure('4 Cluster K-Means')

plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=result.labels_)



plt.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], marker='X',s=150,c='r')





plt.show()

from sklearn.manifold import TSNE



data_reduced = TSNE(n_components=2).fit_transform(data[['scaled_age','scaled_annual_income','scaled_spending_score']])
centroids_reduced = TSNE(n_components=2).fit_transform(kmeans.cluster_centers_)
plt.figure('4 Cluster K-Means')

plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=result.labels_)



plt.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], marker='X',s=150,c='r')





plt.show()

plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=result.labels_)
