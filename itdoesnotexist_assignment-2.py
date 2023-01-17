# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_path = './../input/credit-card-data.csv' # Path to data file
data = pd.read_csv(data_path)
data.head(15)
# What columns are in the data set ? Do they have spaces that I should consider
data.columns
data.describe()
data['tenure'].value_counts().plot(kind = 'bar')
missing_data_results = data['credit_limit'].isnull().sum()
print(missing_data_results)
# Let's view the distribution of the data, where is it possible to find groups?
# We are using boxplots of all the columns except the first (cust_id which is a string)
for col in data.columns[2:]:
    data[col].plot(kind = 'box')
    plt.title('Box plot for ' +col)
    plt.show()
cluster_data = data[['purchases', 'payments']]
cluster_data.head()
cluster_data.plot(kind='scatter', x='purchases', y='payments')
# is there any missing data?
missing_data_results = cluster_data.isnull().sum()
print(missing_data_results)

# perform imputation with median values
# not require since none missing
#cluster_data = cluster_data.fillna( data.median() )
#retrieve just the values for all columns except customer id
data_values = cluster_data.iloc[ :, :].values
data_values
# Use the Elbow method to find a good number of clusters using WCSS (within-cluster sums of squares)
wcss = []
for i in range( 1, 15 ):
    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300) 
    kmeans.fit_predict( data_values )
    wcss.append( kmeans.inertia_ )
    
plt.plot( wcss, 'ro-', label="WCSS")
plt.title("Computing WCSS for KMeans++")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()
kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10, max_iter=300) 
cluster_data["cluster"] = kmeans.fit_predict( data_values )
cluster_data
cluster_data['cluster'].value_counts()
cluster_data['cluster'].value_counts().plot(kind='bar',title='Distribution of Customers across groups')
sns.pairplot( cluster_data, hue="cluster")
grouped_cluster_data = cluster_data.groupby('cluster')
grouped_cluster_data
grouped_cluster_data.describe()
grouped_cluster_data.plot(subplots=True,)
kmeans.predict(pd.DataFrame({'purchases':[300],'payments':[40000]}))
help(kmeans.predict)
kmeans.predict([[3,2]])

kmeans.algorithm
kmeans.cluster_centers_
kmeans
limit = np.mean(data.credit_limit)
purch = np.mean(data.purchases)
comp = [limit, purch]
objects = ('Average Limit', 'Average Purchases')
y_pos = np.arange(len(objects)) 
plt.bar(y_pos, comp, align='center', color = 'orange')
plt.xticks(y_pos, objects)
plt.ylabel('Amount of Money')
plt.title('Graph showing average amount of money being spent compared to the average credit limit')
advance_y = data[data.cash_advance > 0].cash_advance.count()
advance_n = data[data.cash_advance == 0].cash_advance.count()
advance = [advance_y, advance_n]
labels = ('Cash Advance', 'No Cash Advance')
colors = ['red', 'green']
plt.pie(advance, labels=labels, colors=colors, autopct='%1.1f%%',startangle=140)
plt.title('Pie chart showing percentage of people who use cash advance')
advance = data[data.cash_advance > 0].cash_advance.sum()
purchase = data[data.purchases > 0].purchases.sum()
comp = [purchase, advance]
objects = ('Purchases', 'Cash Advances')
y_pos = np.arange(len(objects))
plt.bar(y_pos, comp, align='center')
plt.xticks(y_pos, objects)
plt.ylabel('Amount')
plt.title('Bar chart comparing total amount of cash advances and total amount of purchases')
purchases0 = np.mean(cluster_data[cluster_data.cluster == 0].purchases)
purchases1 = np.mean(cluster_data[cluster_data.cluster == 1].purchases)
purchases2 = np.mean(cluster_data[cluster_data.cluster == 2].purchases)
purchases3 = np.mean(cluster_data[cluster_data.cluster == 3].purchases)
purchases4 = np.mean(cluster_data[cluster_data.cluster == 4].purchases)
cluster_purchases = [purchases0, purchases1, purchases2, purchases3, purchases4]
objects = ('Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4')
colors = ['red', 'green', 'yellow', 'blue', 'orange']
plt.pie(cluster_purchases, labels=objects, colors=colors, autopct='%1.1f%%',startangle=140)
plt.title('Chart comparing average spending in different clusters')
payments0 = np.mean(cluster_data[cluster_data.cluster == 0].payments)
payments1 = np.mean(cluster_data[cluster_data.cluster == 1].payments)
payments2 = np.mean(cluster_data[cluster_data.cluster == 2].payments)
payments3 = np.mean(cluster_data[cluster_data.cluster == 3].payments)
payments4 = np.mean(cluster_data[cluster_data.cluster == 4].payments)
cluster_payments = [payments0, payments1, payments2, payments3, payments4]
objects = ('Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4')
y_pos = np.arange(len(objects))
plt.bar(y_pos, cluster_payments, align='center')
plt.xticks(y_pos, objects)
plt.ylabel('Average Payment Amount')
plt.title('Chart comparing average payments in different clusters')
cluster_data2 = data[['cash_advance','balance']]
cluster_data2.head()
cluster_data2.plot(kind="scatter", x="balance", y="cash_advance")
missing_data_results = cluster_data2.isnull().sum()
print(missing_data_results)
#retrieve just the values for all columns except customer id
data_values2 = cluster_data2.iloc[ :, :].values
data_values2

# Use the Elbow method to find a good number of clusters using WCSS (within-cluster sums of squares)
wcss = []
for i in range( 1, 15 ):
    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300) 
    kmeans.fit_predict( data_values2 )
    wcss.append( kmeans.inertia_ )
    
plt.plot( wcss, 'ro-', label="WCSS")
plt.title("Computing WCSS for KMeans++")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()
kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10, max_iter=300) 
cluster_data2["cluster"] = kmeans.fit_predict( data_values2 )
cluster_data2
cluster_data2['cluster'].value_counts()
cluster_data2['cluster'].value_counts().plot(kind='bar',title='Distribution of Customers across groups')
sns.pairplot( cluster_data2, hue="cluster")
grouped_cluster_data2 = cluster_data2.groupby('cluster')
grouped_cluster_data2
grouped_cluster_data2.describe()
grouped_cluster_data2.plot(subplots=True,)
kmeans.predict(pd.DataFrame({'cash_advance':[200],'balance':[6000]}))
kmeans.predict([[3,2]])
kmeans.algorithm
kmeans.cluster_centers_
kmeans
