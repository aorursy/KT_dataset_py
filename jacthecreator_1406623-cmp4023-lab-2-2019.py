import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import seaborn as sns

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_path = '/kaggle/input/dwdm-petrol-prices/Petrol Prices.csv' # Path to data file

data = pd.read_csv(data_path)

data
# Displaying the first 7 records

data.head(7)
# Displaying the last 4 records

data.tail(4)
# Number of records that exist for each month in the data set

monthly_count = pd.to_datetime(data['Date'], errors='coerce').dt.strftime('%B').value_counts()

monthly_count
monthly_count.plot(kind='bar', title ="Number of records that exist for each month in the data set", figsize=(15, 10))

plt.show()
data['Month'] = pd.to_datetime(data['Date'], errors='coerce').dt.strftime('%B')

data['Month']
data['Day'] = pd.to_datetime(data['Date'], errors='coerce').dt.strftime('%d')

data['Day']
data['Year'] = pd.to_datetime(data['Date'], errors='coerce').dt.strftime('%Y')

data['Year']
data
# data['Timestamp'] = pd.to_datetime(data['Date'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

data['Timestamp'] = pd.to_datetime(data['Date'], errors='coerce').dt.strftime('%Y-%m-%d')

data['Timestamp']
data2 = data[['Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery', 'Timestamp', 'Day', 'Month', 'Year']].copy()

data2
data2.plot(kind='line', x='Timestamp', title="Comparsion of Gas Prices", figsize=(15, 10))

plt.show()
data2['Butane'].pct_change(periods= 4,fill_method='ffill')
data2['Butane'].pct_change(periods= 4,fill_method='ffill').plot( title="Percentage Change of Butane", figsize=(15, 10))
data3 = data2[['Butane', 'Propane', 'Year']]

data3
cluster_data = data2[['Butane', 'Propane']]

cluster_data
missing_data_results = cluster_data.isnull().sum()

print(missing_data_results)



# perform imputation with median values

cluster_data = cluster_data.fillna( data.median() )

print(cluster_data)
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
kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, max_iter=300) 

cluster_data["cluster"] = kmeans.fit_predict( data_values )

cluster_data
cluster_data['cluster'].value_counts()
cluster_data['cluster'].value_counts().plot(kind='bar',title='Distribution of Customers across groups')
sns.pairplot( cluster_data, hue="cluster")
grouped_cluster_data = cluster_data.groupby('cluster')

grouped_cluster_data
grouped_cluster_data.describe()
grouped_cluster_data.plot(subplots=True,)
# What is the average price per year of each gas type (“interesting column”) before you clustered the data?

data3.groupby('Year').mean()