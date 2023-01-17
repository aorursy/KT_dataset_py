import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import seaborn as sns

from sklearn.cluster import KMeans
data = pd.read_csv('../input/Mall_Customers.csv')
data.shape
data.head()
data.isnull().any()
uniq = data.CustomerID.unique()

len(uniq)
data.describe()
data[['Gender','CustomerID']].groupby('Gender').count()
gender = data['Gender'].value_counts()

labels = ['Female', 'Male']

colors = ['c', 'coral']

explode = [0, 0.05]



plt.figure(figsize=(8,8))

plt.title('Total of customers by gender', fontsize = 16, fontweight='bold') 

plt.pie(gender, colors = colors, autopct = '%1.0f%%', labels = labels, explode = explode, startangle=90, textprops={'fontsize': 16})

plt.savefig('Total of customers by gender.png', bbox_inches = 'tight')

plt.show()
gender_spending = data[['CustomerID', 'Spending Score (1-100)','Gender']].groupby('Gender').mean()

gender_spending
values = gender_spending['Spending Score (1-100)'].values

genders = ['Female', 'Male']



plt.title('Average spending score by gender', fontsize = 14, fontweight='bold')

plt.bar(genders[0], values[0], color = 'c')

plt.bar(genders[1], values[1], color = 'coral')

plt.yticks(np.arange(0, max(values)+10, 10), fontsize = 12)

plt.xticks(fontsize = 12)

for i in range(len(values)):

    plt.text(x = genders[i], y = values[i] + 2, s = round(values[i],2), size = 12)

plt.savefig('Average spending score by gender.png')

plt.show()    
ages = data[['Age']].describe()

ages
plt.figure(figsize=(10,8))

plt.title('Distribution of Age', fontsize = 16, fontweight='bold')

plt.hist(data['Age'], color = 'mediumpurple', edgecolor = 'rebeccapurple')

plt.xlabel('Age', fontsize = 13)

plt.savefig('Distribution of Age.png', bbox_inches = 'tight')

plt.grid(False)
income = data[['Annual Income (k$)']].describe()

income
plt.figure(figsize=(10,8))

plt.title('Distribution of Annual Income', fontsize = 16, fontweight='bold')

plt.hist(data['Annual Income (k$)'], color = 'gold', edgecolor = 'goldenrod')

plt.xlabel('Annual Income (k$)', fontsize = 13)

plt.savefig('Distribution of Annual Income.png', bbox_inches = 'tight')

plt.grid(False)
plt.figure(figsize=(16,6))

plt.subplot(1,2,1)

sns.distplot(data['Spending Score (1-100)'], color = 'green')

plt.title('Distribution of Spending Score')

plt.subplot(1,2,2)

sns.distplot(data['Annual Income (k$)'], color = 'green')

plt.title('Distribution of Annual Income (k$)')

plt.show()
sns.pairplot(data=data, diag_kind="kde")

plt.savefig('Distribution.png', bbox_inches = 'tight')
plt.figure(figsize=(8,6))

plt.title('Annual Income vs Spending Score', fontsize = 16, fontweight='bold')  

plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], color = 'indianred', edgecolors = 'crimson')

plt.xlabel('Annual Income', fontsize = 14)

plt.ylabel('Spending Score', fontsize = 14)

plt.savefig('Annual Income vs Spending Score.png', bbox_inches = 'tight')

plt.show()
plt.figure(figsize=(8,6))

plt.title('Age vs Spending Score', fontsize = 16, fontweight='bold')  

plt.scatter(data['Age'], data['Spending Score (1-100)'], color = 'indianred', edgecolors = 'crimson')

plt.xlabel('Age', fontsize = 14)

plt.ylabel('Spending Score', fontsize = 14)
# calculate distortion for a range of number of cluster

X = data[['Annual Income (k$)' , 'Spending Score (1-100)']].iloc[: , :].values

distortions = []



for i in range(1, 11):

    km = KMeans(n_clusters=i, random_state=0)

    km.fit(X)

    distortions.append(km.inertia_)



# plot

plt.title('Elbow Method', fontsize = 14, fontweight='bold')

plt.plot(range(1, 11), distortions, 'c', marker='o', markeredgecolor = 'b')

plt.xlabel('Number of clusters', fontsize = 14)

plt.ylabel('Distortion', fontsize = 14)

plt.annotate('k = 5', xy=(5.1, 51000), xytext=(6, 85000),fontsize = 12, arrowprops={'arrowstyle': '->', 'color': 'blue'})

plt.savefig('Elbow Method.png', bbox_inches = 'tight')

plt.show()
kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)

y_kmeans=kmeans.fit_predict(X)



plt.figure(figsize=(10,8))

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'royalblue', edgecolors = 'navy', label = 'Cluster 1')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'lightsalmon', edgecolors = 'tomato', label = 'Cluster 2')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'gold', edgecolors = 'goldenrod', label = 'Cluster 3')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'mediumorchid', edgecolors = 'purple', label = 'Cluster 4')

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'turquoise', edgecolors = 'teal', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'black', label = 'Centroids')

plt.title('Clusters of customers', fontsize = 16, fontweight='bold')

plt.xlabel('Annual Income (k$)', fontsize = 14)

plt.ylabel('Spending Score (1-100)', fontsize = 14)

plt.legend(fontsize = 14)

plt.savefig('Clusters of customers.png', bbox_inches = 'tight')

plt.show()
#beginning of  the cluster numbering with 1 instead of 0

y_kmeans1=y_kmeans

y_kmeans1=y_kmeans+1



# New Dataframe called cluster

num_cluster = pd.DataFrame(y_kmeans1)



# Adding cluster to the Dataset

data['cluster'] = num_cluster



#Mean of clusters

kmeans_mean_cluster = pd.DataFrame(round(data.groupby('cluster').mean(),1))

kmeans_mean_cluster
gender_count_cluster = data[['cluster','Gender', 'CustomerID']].groupby(['cluster','Gender']).count()

gender_count_cluster
total_cluster = data[['cluster','CustomerID']].groupby(['cluster']).count()

total_cluster
def as_perc(value, total):

    return round((value/total)*100,2)



per = gender_count_cluster.apply(as_perc, total = total_cluster['CustomerID'])

per