# Importing libraries needed
from sklearn.preprocessing import StandardScaler
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv') # Read dataset file
df.head(5) # Get first 5 rows of the dataset
df.info() # Get dataset info
# Renaming columns
df.rename(index=str, columns={'Annual Income (k$)': 'Income',
                              'Spending Score (1-100)': 'Score'}, inplace=True)
df.head()
df.shape # Get the dataset dimension
df.columns # Get column indexes
# Check if there are any missing values
df.isnull().any().any()
# Get the descriptive statistics of the dataset
df.describe(include='all')
# Create distribution plot for Annual Income, Customer's Age, and Spending Score

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (20, 5)

# Distribution plot for annual income
plt.subplot(1, 3, 1)
sns.set(style = 'whitegrid')
sns.distplot(df['Income'])
plt.title('Distribution of Annual Income', fontsize = 16)
plt.xlabel('Range of Annual Income')
plt.ylabel('Count')

# Distribution plot for customer's age
plt.subplot(1, 3, 2)
sns.set(style = 'whitegrid')
sns.distplot(df['Age'], color = 'red')
plt.title('Distribution of Customer''s Age', fontsize = 16)
plt.xlabel('Range of Age')
plt.ylabel('Count')
plt.show()

plt.subplot (1, 3, 3)
sns.set(style = 'whitegrid')
sns.distplot(df['Score'], color = 'orange')
plt.title('Distribution of Spending Score', fontsize = 16)
plt.xlabel('Range of Age')
plt.ylabel('Count')
plt.show()
# Visualization on number of customer based on gender

plt.figure(1 , figsize = (15 , 5))
sns.countplot(y = 'Gender' , data = df)
plt.show()
# Visualization on customers gender's percentage

labels = ['Female', 'Male']
size = df['Gender'].value_counts()
colors = ['lightgreen', 'orange']
explode = [0, 0.1]

# Plot pie chart
plt.rcParams['figure.figsize'] = (9, 9)
plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
plt.title('Gender', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()
# Customer's distribution based on age
plt.figure(figsize=(20,5))
sns.countplot(df['Age'])
plt.xticks(rotation=90)
plt.title('Age Distribution')
# Spending score comparison based on gender
sns.boxplot(df['Gender'],df['Score'])
plt.title('Spending Score Comparison Based on Gender')
# Customer's distribution based on annual income

plt.figure(figsize=(25,5))
sns.countplot(df['Income'])
plt.title('Annual Income Distribution')
# Heatmap correlation

plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),linewidths=.1,cmap="YlGnBu", annot=True)
plt.yticks(rotation=0);
plt.title('Heatmap Correlation on Mall Customer Segmentation')
# Pair plot visualization to see if genders has direct relation on customer segmentation
X = df.drop(['CustomerID', 'Gender'], axis=1)
sns.pairplot(df.drop('CustomerID', axis=1), hue='Gender', aspect=1.5)
plt.title('Pair Plot Visualization on Gender')
plt.show()
#Visualization of Spending score over income

plt.bar(df['Income'],df['Score'])
plt.title('Spending Score Over Income')
plt.xlabel('Income')
plt.ylabel('Score')
# Show the data that we are going to cluster

plt.scatter(df['Income'],df['Score'])
plt.title('Spending Score over Income')
plt.xlabel('Income')
plt.ylabel('Spend Score')
# Defining elbow point to determine K value
from sklearn.cluster import KMeans

# Inertia list
clusters = []
for i in range(1,11):
  km = KMeans(n_clusters=i).fit(X)
  clusters.append(km.inertia_)

# Plot inertia
fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax, marker=".", markersize=10)
ax.set_title('Elbow Method')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')
# Defining Elbow Point

fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax, marker=".", markersize=10)
ax.set_title('Elbow Method')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')
ax.annotate('Optimal Elbow Point', xy=(5, 80000), xytext=(5, 150000), xycoords='data',          
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

plt.show()
# Making K-Means object
km5 = KMeans(n_clusters=5).fit(X)

# Add column labels on dataset
X['Labels'] = km5.labels_

# Plot 5-clusters K-Means
plt.figure(figsize=(8,4))
sns.scatterplot(X['Income'], X['Score'], hue=X['Labels'],
                palette=sns.color_palette('hls', 5))
plt.title('5-Clusters K-Means')
plt.show()
# Silhouette Coefficient of K-Means Model

from sklearn import metrics
round(metrics.silhouette_score(X, X['Labels']), 2)
# Determine MinPts and Epsilon

data_c = pd.DataFrame({'Age': df['Age'], 'Income': df['Income'], 'Score': df['Score']})
from sklearn.neighbors import NearestNeighbors
neighbors = NearestNeighbors(n_neighbors=6) #n_neighbors is the MinPts
neighbors_fit = neighbors.fit(data_c)
distances, indices = neighbors_fit.kneighbors(data_c)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)

# Create DBSCAN Object

from sklearn.cluster import DBSCAN 

db = DBSCAN(eps=11, min_samples=6).fit(X)

X['Labels'] = db.labels_
plt.figure(figsize=(12, 4))
sns.scatterplot(X['Income'], X['Score'], hue=X['Labels'], 
                palette=sns.color_palette('hls', np.unique(db.labels_).shape[0]))
plt.title('DBSCAN with epsilon 11, min samples 6')
plt.show()
# Silhouette Coefficient of DBSCAN Model

round(metrics.silhouette_score(X, X['Labels']), 2)
# Create Mean Shift Object

from sklearn.cluster import MeanShift, estimate_bandwidth

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.1)
ms = MeanShift(bandwidth).fit(X)

X['Labels'] = ms.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(X['Income'], X['Score'], hue=X['Labels'], 
                palette=sns.color_palette('hls', np.unique(ms.labels_).shape[0]))
plt.plot()
plt.title('MeanShift')
plt.show()
# Silhouette Coefficient of Mean Shift Model

round(metrics.silhouette_score(X, X['Labels']), 2)
# Plot swarmplot to analyze clusters
X['Labels'] = ms.labels_

fig = plt.figure(figsize=(20,8))
ax = fig.add_subplot(121)
sns.swarmplot(x='Labels', y='Income', data=X, ax=ax)
ax.set_title('Labels According to Annual Income')

ax = fig.add_subplot(122)
sns.swarmplot(x='Labels', y='Score', data=X, ax=ax)
ax.set_title('Labels According to Scoring')

plt.show()