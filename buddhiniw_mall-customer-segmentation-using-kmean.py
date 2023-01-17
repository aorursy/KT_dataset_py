import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans 



# Load the dataset 

data=pd.read_csv('../input/Mall_Customers.csv')

data.head()
# check for NULLs

data.isnull().sum()
# Drop customer ID (Categorical variable)

df = data.drop('CustomerID', axis=1)

df.head()
df = df.drop_duplicates()

df.count()
# Overall summary

df.describe()
df.describe().transpose()
fig = plt.figure(figsize=(15,10))



plt.subplot(2, 2, 1)

sns.set(style = 'whitegrid')

sns.distplot(df['Annual Income (k$)'], color = 'green',kde=False)



plt.title('Distribution of Annual Income', fontsize = 20)

plt.xlabel('Annual Income (k$)')

plt.ylabel('Count')



plt.subplot(2, 2, 2)

sns.set(style = 'whitegrid')

sns.distplot(df['Age'], color = 'red',kde=False) # Turned off KDE 

plt.title('Distribution of Age', fontsize = 20)

plt.xlabel('Age (years)')

plt.ylabel('Count')





plt.subplot(2, 2, 3)

labels = ['Men', 'Women']

percentages = df['Gender'].value_counts(normalize=True) * 100



explode=(0.1,0)

plt.pie(percentages, explode=explode, labels=labels,  

       autopct='%1.0f%%', 

       shadow=False, startangle=0,   

       pctdistance=1.2,labeldistance=1.4)

plt.axis('equal')

#plt.title("Gender Ratios")

plt.legend(frameon=False, bbox_to_anchor=(1.5,0.8))



plt.tight_layout()
fig = plt.figure(figsize=(15,15))

sns.set(style = 'whitegrid')

#sns.set(style="ticks")



sns.pairplot(df, hue="Gender", palette="Set2")
from sklearn.preprocessing import StandardScaler

X = df.values[:,1:]

X = np.nan_to_num(X)



Clus_dataSet = StandardScaler().fit_transform(X)

Clus_dataSet 
sum_of_squared_distances = []

K = range(1,9)

for k in K:

    km = KMeans(n_clusters=k, init = 'k-means++', 

                max_iter = 300, n_init = 10, random_state = 0)

    km = km.fit(Clus_dataSet)

    sum_of_squared_distances.append(km.inertia_)

    

plt.plot(K, sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('sum of squared distances')

plt.title('Elbow Method')

plt.show()
Clus_dataSet1 = pd.DataFrame(Clus_dataSet[:])

Clus_dataSet1 = Clus_dataSet1[[0,2]]



sum_of_squared_distances = []

K = range(1,9)

for k in K:

    km = KMeans(n_clusters=k, init = 'k-means++', 

                max_iter = 300, n_init = 5, random_state = 0)

    km = km.fit(Clus_dataSet1)

    sum_of_squared_distances.append(km.inertia_)

    

plt.plot(K, sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('sum of squared distances')

plt.title('Elbow Method for Optimum K')

plt.show()
Clus_dataSet2 = pd.DataFrame(Clus_dataSet[:])

Clus_dataSet2 = Clus_dataSet2[[1,2]]



sum_of_squared_distances = []

K = range(1,9)

for k in K:

    km = KMeans(n_clusters=k, init = 'k-means++', 

                max_iter = 300, n_init = 5, random_state = 0)

    km = km.fit(Clus_dataSet2)

    sum_of_squared_distances.append(km.inertia_)

    

plt.plot(K, sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('sum of squared distances')

plt.title('Elbow Method for Optimum K')

plt.show()
# apply k-means on the dataset

from sklearn.cluster import KMeans 

clusterNum = 5

k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)

k_means.fit(X)

labels = k_means.labels_

print(labels)
# assign the labels to each row in dataframe.

df["cluster"] = labels

df.head(5)
# We can easily check the centroid values by averaging the features in each cluster.

df.groupby('cluster').mean()
# look at the distribution of customers based on their age and income:



# Create plot

fig = plt.figure()



sns.lmplot( x="Annual Income (k$)", y="Spending Score (1-100)", 

           data=df, fit_reg=False, hue='cluster', legend=False)

plt.legend(loc='upper right')

labels = ['Average shoppers', 'Budget shoppers', 

          'Under spending shoppers','Over spending shoppers',

          'High spending shoppers']

#plt.legend(labels)

plt.legend(labels, loc='center right', 

           bbox_to_anchor=(1.75, 0.5), ncol=1)



#ax.scatter(X[:, 1], X[:, 2], c=labels, alpha=0.5)

plt.xlabel('Annual Income', fontsize=18)

plt.ylabel('Spending Score (1-100)', fontsize=16)

plt.title('K-Mean Clusters')

plt.show()
sns.pairplot(df[df['cluster'] == 2], hue="Gender", palette="Set2")