#Importing the libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.set_style("whitegrid")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#Importing and reading the dataset



df_mall = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv') 



df_mall.head()
df_mall.describe()
df_mall.columns = ['CustomerID', 'Gender', 'Age', 'AnnualIncome', 'SpendingScore(1-100)']
#Checking if there are any missing/null values in our dataset



df_mall.isnull().any()
df_mall.info()
sns.pairplot(data= df_mall)
#Univariate distribution



fig, axs = plt.subplots(ncols = 3, figsize= (18,5))



sns.distplot(df_mall['Age'], bins=20, ax=axs[0]).set_title("Distribution of Age")



sns.distplot(df_mall['SpendingScore(1-100)'], bins=20, ax=axs[1]).set_title("Distribution of Spending Score")



sns.distplot(df_mall['AnnualIncome'], bins=20, ax=axs[2]).set_title("Distribution of Annual Income")
sns.countplot(data= df_mall, x='Gender').set_title("Gender Distribution")
fig, axs = plt.subplots(figsize= (8,5))

axs.set_xlabel("Age")

axs.set_ylabel("Count")



sns.kdeplot(df_mall.Age[df_mall.Gender=='Male'], label='men', shade=True)



sns.kdeplot(df_mall.Age[df_mall.Gender=='Female'], label='women', shade=True)
sns.jointplot(data= df_mall, x= "AnnualIncome", y= "SpendingScore(1-100)", kind='hex')



sns.jointplot(data= df_mall, x= "Age", y= "SpendingScore(1-100)", kind='hex')



sns.jointplot(data= df_mall, x= "Age", y= "AnnualIncome", kind='hex')
sns.set_style("white")

fig, ax = plt.subplots(figsize= (8,6))

sns.scatterplot(data= df_mall, x= "Age", y= "AnnualIncome", hue="Gender", alpha=.6, ax=ax)

ax.set_title("Age v/s Annual Income")
#Mean spending of females is bit more than that of males.



sns.boxplot(data= df_mall, x= 'Gender', y= 'SpendingScore(1-100)', hue='Gender')
#import KMeans



from sklearn.cluster import KMeans
X = df_mall[['AnnualIncome', 'SpendingScore(1-100)']]

X.values
sq_dist = {}

for n in range(1,11):

    cluster = KMeans(n_clusters = n, random_state= 5)

    y_KMeans = cluster.fit(X)

    sq_dist[n] = y_KMeans.inertia_

    

sns.pointplot(x = list(sq_dist.keys()), y = list(sq_dist.values()))

plt.xlabel('No. of Clusters')

plt.ylabel('Square distances sum')

plt.show()
cluster = KMeans(n_clusters = 5, random_state= 5)

y_KMeans = cluster.fit_predict(X)

print(y_KMeans)
df_mall['clusters'] = cluster.labels_

plt.figure(figsize = (10, 6))

sns.scatterplot(df_mall['AnnualIncome'], df_mall['SpendingScore(1-100)'], hue=df_mall['clusters'], palette='prism')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score(1-100)')

plt.scatter(cluster.cluster_centers_[:,0], cluster.cluster_centers_[:,1], s=40, c='black', marker = '*')