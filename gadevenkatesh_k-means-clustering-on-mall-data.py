import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

%matplotlib inline

df = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
df.head()
df.shape
df.info()
df.isnull().sum()
# Understanding Male-Female split
sns.set_style(style= "whitegrid") 
sns.countplot(x = 'Gender', data = df )
plt.title('Gender Split')
plt.show()


df.Gender.value_counts()
# identifying relevant numerical columns
numerical_col = df.select_dtypes(include = np.number).drop('CustomerID', axis = 'columns').columns
numerical_col
sns.set_style(style= "dark") 
for column in numerical_col:
  sns.distplot(df[column], color= 'green')
  plt.show()
df.corr()

# younger the customer --- higher the spending score (-ve Correlation)
sns.heatmap(df.corr())
plt.show()
 
sns.set_style(style= "white") 
df1_male = df[df.Gender == 'Male']
df1_female = df[df.Gender == 'Female']

fig = plt.figure(figsize=(10,6))

plt.scatter(df1_male.Age, df1_male['Spending Score (1-100)'], color = 'blue', label = 'male')
plt.scatter(df1_female.Age, df1_female['Spending Score (1-100)'], color = 'magenta', label = 'female')
plt.title('Age  vs  Spending Score')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

df1_male = df[df.Gender == 'Male']
df1_female = df[df.Gender == 'Female']
plt.figure(figsize=(10,6))

plt.scatter(df1_male['Annual Income (k$)'], df1_male['Spending Score (1-100)'], color = 'blue', label = 'male')
plt.scatter(df1_female['Annual Income (k$)'], df1_female['Spending Score (1-100)'], color = 'magenta', label = 'female')
plt.title('Annual Income   vs   Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

plt.show()
df1_male = df[df.Gender == 'Male']
df1_female = df[df.Gender == 'Female']
plt.figure(figsize=(10,6))

plt.scatter(df1_male['Age'], df1_male['Annual Income (k$)'], color = 'blue', label = 'male')
plt.scatter(df1_female['Age'], df1_female['Annual Income (k$)'], color = 'magenta', label = 'female')
plt.title('Age   vs   Annual Income (k$)')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.legend()

plt.show()
from sklearn.cluster import KMeans
km = KMeans(n_clusters=5)

cluster_n = []
SSE = []

for i in range(1,9):
  km1 = KMeans(n_clusters= i)
  km1.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
  SSE.append(km1.inertia_)
  cluster_n.append(i)

sns.set_style('whitegrid')
plt.plot(cluster_n, SSE)
plt.title('Elbow Method')
plt.xlabel('No. of Clusters')
plt.ylabel("SSE")
plt.show()
# K - Means Clustering in Action
from sklearn.cluster import KMeans
km = KMeans(n_clusters=5)
cluster = km.fit_predict(df[['Annual Income (k$)','Spending Score (1-100)']])
cluster
df['Cluster'] = cluster
df.head()
# plotting the clusters

c0 = df[df['Cluster'] == 0]
c1 = df[df['Cluster'] == 1]
c2 = df[df['Cluster'] == 2]
c3 = df[df['Cluster'] == 3]
c4 = df[df['Cluster'] == 4]

plt.figure(figsize=(10,6))
sns.set_style("white")

plt.scatter(c0['Annual Income (k$)'], c0['Spending Score (1-100)'], color = 'red', label = 'Cluster 0')
plt.scatter(c1['Annual Income (k$)'], c1['Spending Score (1-100)'], color = 'orange', label = 'Cluster 1')
plt.scatter(c2['Annual Income (k$)'], c2['Spending Score (1-100)'], color = 'green', label = 'Cluster 2')
plt.scatter(c3['Annual Income (k$)'], c3['Spending Score (1-100)'], color = 'magenta', label = 'Cluster 3')
plt.scatter(c4['Annual Income (k$)'], c4['Spending Score (1-100)'], color = 'blue', label = 'Cluster 4')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1], color = 'black', label = "Centroid", marker = "*")

plt.title('Annual Income   vs   Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

plt.show()
df.head()
Premium_Customers = df[df['Cluster']==2]
Premium_Customers.head()
sns.countplot(Premium_Customers['Gender'])
Premium_Customers.Gender.value_counts()
sns.distplot(Premium_Customers['Spending Score (1-100)'], color = 'green')
Premium_Customers['Spending Score (1-100)'].mean()
# Premium_Customers['Spending Score (1-100)'].median()
sns.distplot(Premium_Customers['Age'], bins = 14, color = 'green')
plt.xticks(ticks = np.arange(27,41))
sns.distplot(Premium_Customers['Annual Income (k$)'], bins = 10, color = 'green')
Premium_Customers['Annual Income (k$)'].mean()
Premium_Customers['Annual Income (k$)'].median()
print(Premium_Customers['Annual Income (k$)'].min()," ",Premium_Customers['Annual Income (k$)'].max())

