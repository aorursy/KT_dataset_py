# Import needed libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Read data in the csv file

df = pd.read_csv('../input/mall-customers/Mall_Customers.csv')
df.head()

df.shape
df.describe()
# Check null values
df.isnull().sum()
# Rename Columns 

df.rename(columns = {'Annual Income (k$)':'Annual_Income_(k$)' , 'Spending Score (1-100)': 'Spending_Score'}, inplace = True)
df.head()
# Dispaly the distribution of Spending_Score in a chart
plt.hist(df['Spending_Score'])
plt.show()
# Dispaly the distribution of Annual_Income_(k$) in a chart
plt.hist(df['Annual_Income_(k$)'])
plt.show()
# Dispaly the distribution of Age in a chart
plt.hist(df['Age'])
plt.show()
# Display the number of male and female in a chart

gender_df = df.groupby(['Genre']).count()['CustomerID']
gender_df.plot(kind='bar', title ='Genders Distribution');
# Spending_Score mean with respect to Genre
df[['Spending_Score',  'Genre']].groupby(['Genre']).mean()
# Dispaly the distribution of Spending_Score based on the gender
figure, ax = plt.subplots(figsize = (10, 5))
ax.hist(df[df['Genre'] =='Male']['Spending_Score'], color = 'blue', alpha = 0.7, label = 'Male')
ax.hist(df[df['Genre'] =='Female']['Spending_Score'], color ='red', alpha = 0.7, label = 'Female')
plt.xlabel('Spending Score')
plt.ylabel('Count')
plt.title('Spending Score Distribution of the Customers based on their Gender')
plt.legend()
plt.show()
# Spending_Score max with respect to Genre
df[['Spending_Score', 'Genre']].groupby(['Genre']).max()
# Spending_Score min with respect to Genre
df[['Spending_Score', 'Genre']].groupby(['Genre']).min()
# Display the relationship between the features of the dataset
sns.pairplot(df[[ 'Age', 'Annual_Income_(k$)', 'Spending_Score']])
X = df[['Annual_Income_(k$)', 'Spending_Score']]
# Average distance to the center
WSS = []
index = range(1,8)
for i in index:
    kmeans = KMeans(n_clusters = i, random_state = 42)
    kmeans.fit(X)
    WSS.append(kmeans.inertia_)
    print(kmeans.inertia_)
# Use the Elbow method to determine the optimum number of clusters
plt.plot(index, WSS)
plt.xlabel('K')
plt.ylabel('WSS')
plt.show()
# Create Kmeans model with 5 clusters
kmeans = KMeans(n_clusters = 5, random_state = 42)
kmeans.fit(X)
# The values of cluster centers
center = kmeans.cluster_centers_
center
# The clusters of the customers
kmeans.labels_
# Add the cluster column to the dataframe 
X_cls = X
X_cls['Cluster'] = kmeans.labels_
X_cls.head()
# Display the five clusters and the centroid
plt.scatter(df['Annual_Income_(k$)'], df['Spending_Score'], c = kmeans.labels_)
plt.scatter(center[:, 0], center[:, 1], c = 'red', marker = '*')
plt.xlabel( 'Annual Income (K$)')
plt.ylabel ('Spending Score')
plt.title ('Clusters and Centroid')
plt.show()
# Mearge the two dataframes based on the indexes numbers
new_df = df.merge(X_cls['Cluster'], left_index = True, right_index = True, how = 'inner')
new_df.head()
# Creating a datframe for Cluster 0. 
# Cluster 0: medium sending score and medium annual income.
df_c0 = new_df[new_df['Cluster'] == 0]
df_c0.head()
# Creating a datframe for Cluster 1. 
# Cluster 1: low spending score and high annual income.
df_c1 = new_df[new_df['Cluster'] == 1]
df_c1.head()
# Creating a datframe for Cluster 2. 
# Cluster 2: low sending score and low annual income.
df_c2 = new_df[new_df['Cluster'] == 2]
df_c2.head()
# Creating a datframe for Cluster 3. 
# Cluster 3: high sending score and low annual income.
df_c3 = new_df[new_df['Cluster'] == 3]
df_c3.head()
# Creating a datframe for Cluster 4. 
# Cluster 4: high spending score and high annual income.
df_c4 = new_df[new_df['Cluster'] == 4]
df_c4.head()
# Creating a datframe for very heigh spending score, Spending Score > 60
df_high_sp = new_df[new_df['Spending_Score'] > 60]
df_high_sp.head()
# Dispaly the distribution of Age in the Cluster 0
plt.hist(df_c0['Age'])
plt.show()
# Display the number of male and female in the Cluster 0
gender_df_c0 = df_c0.groupby(['Genre']).count()['CustomerID']
gender_df_c0.plot(kind='bar', title ='Genders Distribution');
# Dispaly the distribution of Age in the Cluster 1
plt.hist(df_c1['Age'])
plt.show()
# Display the number of male and female in the Cluster 1
gender_df_c1 = df_c1.groupby(['Genre']).count()['CustomerID']
gender_df_c1.plot(kind='bar', title ='Genders Distribution');
# Dispaly the distribution of Age in the Cluster 2
plt.hist(df_c2['Age'])
plt.show()
# Display the number of male and female in the Cluster 2
gender_df_c2 = df_c2.groupby(['Genre']).count()['CustomerID']
gender_df_c2.plot(kind='bar', title ='Genders Distribution');
# Dispaly the distribution of Age in the Cluster 3
plt.hist(df_c3['Age'])
plt.show()
# Display the number of male and female in the Cluster 3
gender_df_c3 = df_c3.groupby(['Genre']).count()['CustomerID']
gender_df_c3.plot(kind='bar', title ='Genders Distribution');
# Dispaly the distribution of Age in the Cluster 4
plt.hist(df_c4['Age'])
plt.show()
# Display the number of male and female in the Cluster 4
gender_df_c4 = df_c4.groupby(['Genre']).count()['CustomerID']
gender_df_c4.plot(kind='bar', title ='Genders Distribution');
# Dispaly the distribution of Age of the customers with very heigh spending score
plt.hist(df_high_sp['Age'])
plt.show()
# Display the number of male and female of the customers with very heigh spending score
gender_df_high_sp = df_high_sp.groupby(['Genre']).count()['CustomerID']
gender_df_high_sp.plot(kind='bar', title ='Genders Distribution');
# Dispaly the distribution of Age based on the gender in Cluster 0
figure, ax = plt.subplots(figsize = (10, 5))
ax.hist(df_c0[df_c0['Genre'] =='Male']['Age'], color = 'blue', alpha = 0.7, label = 'Male')
ax.hist(df_c0[df_c0['Genre'] =='Female']['Age'], color ='red', alpha = 0.7, label = 'Female')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution of the Customers based on their Gender')
plt.legend()
plt.show()
# Dispaly the distribution of Age based on the gender in Cluster 1
figure, ax = plt.subplots(figsize = (10, 5))
ax.hist(df_c1[df_c1['Genre'] =='Male']['Age'], color = 'blue', alpha = 0.7, label = 'Male')
ax.hist(df_c1[df_c1['Genre'] =='Female']['Age'], color ='red', alpha = 0.7, label = 'Female')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution of the Customers based on their Gender')
plt.legend()
plt.show()
# Dispaly the distribution of Age based on the gender in Cluster 2
figure, ax = plt.subplots(figsize = (10, 5))
ax.hist(df_c2[df_c2['Genre'] =='Male']['Age'], color = 'blue', alpha = 0.7, label = 'Male')
ax.hist(df_c2[df_c2['Genre'] =='Female']['Age'], color ='red', alpha = 0.7, label = 'Female')
plt.xlabel('Age')
plt.ylabel('Age Count')
plt.title(' Distribution of the Customers based on their Gender')
plt.legend()
plt.show()
# Dispaly the distribution of Age based on the gender in Cluster 3
figure, ax = plt.subplots(figsize = (10, 5))
ax.hist(df_c3[df_c3['Genre'] =='Male']['Age'], color = 'blue', alpha = 0.7, label = 'Male')
ax.hist(df_c3[df_c3['Genre'] =='Female']['Age'], color ='red', alpha = 0.7, label = 'Female')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution of the Customers based on their Gender')
plt.legend()
plt.show()
# Dispaly the distribution of Age based on the gender in Cluster 4
figure, ax = plt.subplots(figsize = (10, 5))
ax.hist(df_c4[df_c4['Genre'] =='Male']['Age'], color = 'blue', alpha = 0.7, label = 'Male')
ax.hist(df_c4[df_c4['Genre'] =='Female']['Age'], color ='red', alpha = 0.7, label = 'Female')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution of the Customers based on their Gender')
plt.legend()
plt.show()
# Dispaly the distribution of Age based on the gender of the customers with very heigh spending score
figure, ax = plt.subplots(figsize = (10, 5))
ax.hist(df_high_sp[df_high_sp['Genre'] =='Male']['Age'], color = 'blue', alpha = 0.7, label = 'Male')
ax.hist(df_high_sp[df_high_sp['Genre'] =='Female']['Age'], color ='red', alpha = 0.7, label = 'Female')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution of the Customers based on their Gender')
plt.legend()
plt.show()