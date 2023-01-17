# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/credit-card-data.csv")
data
data.columns
data.describe()
data.head()
data.info()
data.isnull().sum()
# Fill NAs by mean
data = data.fillna(data.mean())

data.isna().sum()
# Remove CUST_ID (not usefull)
data.drop("cust_id", axis=1, inplace=True)
data['tenure'].value_counts().plot(kind='bar')

# Let's view the distribution of the data, where is it possible to find groups?
# We are using boxplots of all the columns except the first (cust_id which is a string)
for col in data.columns[2:]:
    data[col].plot(kind='box')
    plt.title('Box Plot for '+col)
    plt.show()

cluster_data = data[['purchases','payments']]
cluster_data.head()
cluster_data.plot(kind='scatter',x='purchases',y='payments')

# Is there any missing data
missing_data_results = cluster_data.isnull().sum()
print(missing_data_results)

# perform imputation with median values
# not require since none missing
#cluster_data = cluster_data.fillna( data.median() )
from sklearn.cluster import KMeans

#retrieve just the values for all columns except customer id
data_values = cluster_data.values
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
group = data.groupby('tenure')
group.cash_advance.count()
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
group.cash_advance.count().plot(kind='pie', autopct='%1.1f%%',shadow=True)

bal = data[['credit_limit','balance']]
len(bal)
data.credit_limit.iloc[0:8950]
bal = data[['credit_limit','balance']]
over = 0
count = 0
for x in range(0,len(bal)): 
    if (data.balance.iloc[x] < data.credit_limit.iloc[x]):# comparing credit balance and credit limit for each row
        count += 1
    else:
        over += 1

        
print("the amount that spent over their credit card limit:",over)
print("the amount that spent within their credit card limit:",count)
obj = ('Spent within limit','Spent Over Limit')
y_position = np.arange(len(obj))
perform = [count, over]
plt.bar(y_position, perform, align='center')
plt.xticks(y_position, obj)
plt.ylabel('# of Customers')
plt.title('Customers and thier Credit Card Limit')
# Correlation plot
sns.heatmap(data.corr(),
            xticklabels=data.columns,
            yticklabels=data.columns
           )
cluster_data
cluster_data['cluster'].value_counts().plot(kind='pie',autopct='%1.1f%%')
cluster_data['cluster'].value_counts()
x ="purchases"
y="payments"
sns.lmplot(x, # Horizontal axis
           y, # Vertical axis
           data=cluster_data, # Data source
           fit_reg=False, # Don't fix a regression line
           hue="cluster", # Set color
           scatter_kws={"marker": "D", # Set marker style
                        "s": 100}) # S marker size
data
cluster_data = data[['cash_advance','tenure']]
cluster_data.head()
cluster_data.plot(kind='scatter',x='cash_advance',y='tenure')

# Is there any missing data
missing_data_results = cluster_data.isnull().sum()
print(missing_data_results)

# perform imputation with median values
# not require since none missing
#cluster_data = cluster_data.fillna( data.median() )
#retrieve just the values for all columns except customer id
data_values = cluster_data.values
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
kmeans = KMeans(n_clusters=4, init="k-means++", n_init=10, max_iter=300) 
cluster_data["cluster"] = kmeans.fit_predict( data_values )
cluster_data
cluster_data['cluster'].value_counts()
cluster_data['cluster'].value_counts().plot(kind='bar',title='Distribution of Customers across groups')
sns.pairplot( cluster_data, hue="cluster")
grouped_cluster_data = cluster_data.groupby('cluster')
grouped_cluster_data
grouped_cluster_data.describe()
grouped_cluster_data.plot(subplots=True,)
