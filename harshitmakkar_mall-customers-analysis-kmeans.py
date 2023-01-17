# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.simplefilter('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Mall_Customers.csv')
data.head()
data.info()
data.isnull().values.any()
data.corr()
sns.jointplot(x='Age',y='Spending Score (1-100)',data=data,kind='kde')
sns.lmplot(x='Age',y='Spending Score (1-100)',data=data)
#plt.figure(figsize=(15,6))

sns.lineplot(x='Age',y='Spending Score (1-100)',data=data,hue='Gender')
sns.lmplot(x='Annual Income (k$)',y='Spending Score (1-100)',data=data)
data_new = data.iloc[:, [2, 4]].values
from sklearn.cluster import KMeans 
error_rate = []

for i in range(1,40):

    KM = KMeans(n_clusters=i)

    KM.fit(data_new)

    error_rate.append(KM.inertia_)

    

plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,marker='o')
km = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = km.fit_predict(data_new)
y_means
(data_new[y_means == 0, 0]).shape
plt.scatter(data_new[y_means == 0, 0], data_new[y_means == 0, 1], s = 200, c = 'pink')

plt.scatter(data_new[y_means == 1, 0], data_new[y_means == 1, 1], s = 200, c = 'yellow')

plt.scatter(data_new[y_means == 2, 0], data_new[y_means == 2, 1], s = 200, c = 'cyan')

plt.scatter(data_new[y_means == 3, 0], data_new[y_means == 3, 1], s = 200, c = 'magenta')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue')



plt.title('K Means Clustering', fontsize = 20)

plt.xlabel('Age')

plt.ylabel('Spending Score')