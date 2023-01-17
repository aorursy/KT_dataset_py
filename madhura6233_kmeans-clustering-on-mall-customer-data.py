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

import seaborn as sns

import matplotlib as plt

%matplotlib inline
mall_df = pd.read_csv("../input/Mall_Customers.csv")
mall_df.head(5)
# Checking if there are any NULL or missing values in the dataset

mall_df.isnull().sum()
# Getting the distribution of values in the dataset

mall_df.describe()
mall_df.plot.scatter(x='Annual Income (k$)', y = 'Spending Score (1-100)')
sns.lmplot(x='Age', y = 'Spending Score (1-100)', hue='Gender', data=mall_df)
sns.pairplot(mall_df, palette='inferno')
corr = mall_df.corr()



sns.heatmap(corr, annot=True)
from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters=5)
# First, clustering based on Age and Spending Score



X1 = mall_df.loc[:,['Age', 'Spending Score (1-100)']]
X1.head(5)
mall_df['cluster_age'] = kmeans.fit_predict(X1)
for i in range(0,5):

    print(mall_df[mall_df['cluster_age'] == i].head(5))
plt.pyplot.figure(figsize=(20,10))

label = mall_df['cluster_age'].unique()

plt.pyplot.scatter(x=mall_df['Age'], y = mall_df['Spending Score (1-100)'], c = mall_df['cluster_age'], s=100, label=label)

plt.pyplot.title('Clustering based on Age v/s Spending Score')



# Now let's cluster based on Annual Income and Spending Score

X2 = mall_df.loc[:, ['Annual Income (k$)', 'Spending Score (1-100)']]
mall_df['cluster_annual'] = kmeans.fit_predict(X2)
pd.set_option('display.max_columns', 500)

for i in range(0,5):

    print(mall_df[mall_df['cluster_annual'] == i][['Age','Annual Income (k$)', 'Spending Score (1-100)',  'cluster_annual']].head(5))
plt.pyplot.figure(figsize=(20,10))

label = mall_df['cluster_annual'].unique()

plt.pyplot.scatter(x=mall_df['Annual Income (k$)'], y = mall_df['Spending Score (1-100)'], c = mall_df['cluster_annual'], s=100)

plt.pyplot.title('Clustering based on Annual v/s Spending Score')

#Now let us combine Age, Annual Income and Spending Score to form a more hollistic cluster pattern

X = mall_df.loc[:, ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

mall_df['cluster'] = kmeans.fit_predict(X)



plt.pyplot.figure(figsize=(10,10))

plt.pyplot.scatter(y= mall_df['Spending Score (1-100)'], x = mall_df['Age'], c = mall_df['cluster'], s = 100)

plt.pyplot.title('Age v/s Spending score based on overall clustering')
plt.pyplot.figure(figsize=(10,10))

plt.pyplot.scatter(y= mall_df['Spending Score (1-100)'], x = mall_df['Annual Income (k$)'], c = mall_df['cluster'], s = 100)

plt.pyplot.title('Annual Income v/s Spending score based on overall clustering')