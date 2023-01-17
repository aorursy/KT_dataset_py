# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')



df.head()
df.describe()
df.info()
from sklearn.preprocessing import LabelEncoder



df = df.set_index('CustomerID')



le = LabelEncoder()

df['Gender'] = le.fit_transform(df['Gender']) # 1 for male, 0 for female



df.head()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set();

%matplotlib inline



fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Does Annual Income have impact on Spending Score?

sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df, ax=ax[0]).set_title('Relationship of Income Level and Spending Habit')



# Are older people wealthier?

sns.scatterplot(x='Age', y='Annual Income (k$)', data=df, ax=ax[1]).set_title('Relationship of Age and Income Level');
from scipy.cluster.hierarchy import linkage, fcluster

from scipy.cluster.vq import kmeans, vq

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

X = scaler.fit_transform(df[['Annual Income (k$)', 'Spending Score (1-100)']])



print(X[:5])
# Hierarchy Clustering

z = linkage(X, 'ward')

y1 = fcluster(z, 5, criterion='maxclust')



# Kmeans Clustering

centroids, _= kmeans(X, 5)

y2, _ = vq(X, centroids)



print(y1)

print()

print(y2)

# plot hierarchy result

fig, ax = plt.subplots(1, 2, figsize=(14,7), sharey=True)



sns.scatterplot(x=X[:,0], y=X[:,1], hue=y1, ax=ax[0]);

ax[0].set_ylabel('Scaled Spending Score', size=15);

ax[0].set_title('Clustering by hierarchy', size=15);



# plot kmeans result

sns.scatterplot(x=X[:,0], y=X[:,1], hue=y2, ax=ax[1]);

ax[1].set_title('Clustering by kmeans', size=15);

fig.text(0.5,0.05, 'Scaled Income Level', ha='center', size=15);

from scipy.cluster.hierarchy import dendrogram



plt.figure(figsize=(14,6));

dn = dendrogram(z)