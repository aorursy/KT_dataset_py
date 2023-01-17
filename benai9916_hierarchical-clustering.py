import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import datetime as dt

# import required libraries for clustering
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
df = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
# first five row
df.head()
# size of datset
df.shape
# change column name
df.rename(columns={'Annual Income (k$)': 'Annual_income', 'Spending Score (1-100)': 'Spending_score'}, inplace=True)
# statistical summary of numerical variables
df.describe()
# summary about dataset
df.info()
# check for missing values
df.isna().sum() 
plt.pie(df['Gender'].value_counts(), labels=df['Gender'].unique(), autopct='%1.1f%%')
plt.show()
cols = ['Age', 'Annual_income', 'Spending_score']

for col in cols:
    plt.hist(df[col],bins=15)
    plt.title('Histogram ' + col)
    plt.xlabel(col)
    plt.ylabel('Freq')
    plt.show()
sns.pairplot(df,kind='scatter',hue='Gender',palette=('#40a1e6','#e64040'))
plt.show()
cols = ['Age', 'Annual_income', 'Spending_score']

for col in cols:
    sns.boxplot(data=df, x=col)
    plt.title(col)
    plt.show()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['Gender'] = le.fit_transform(df['Gender'])
# copy data
new_df = df.copy()

# drope customer id
new_df = new_df.drop('CustomerID', axis=1)

# view data after scaling
new_df.head()
# figure size
plt.figure(figsize=(19,8))

# Single linkage
mergings = linkage(new_df, method="single", metric='euclidean')

# diagram
dendrogram(mergings)
plt.show()
# figure size
plt.figure(figsize=(19,8))

# Single linkage
mergings = linkage(new_df, method="complete", metric='euclidean')

# diagram
dendrogram(mergings)
plt.show()
# figure size
plt.figure(figsize=(19,8))

# Single linkage
mergings = linkage(new_df, method="average", metric='euclidean')

# diagram
dendrogram(mergings)
plt.show()
# 2 clusters

cluster_labels = cut_tree(mergings, n_clusters=2).reshape(-1, )
cluster_labels
# Assign cluster labels

new_df['Cluster_Labels'] = cluster_labels
new_df.head()
# Plot Cluster Id vs Spending_score
for cols in new_df.columns[:-1].to_list():
    sns.boxplot(y=cols, x='Cluster_Labels', data=new_df)
    plt.show()