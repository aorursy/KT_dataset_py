import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



# K-Means to cluster the users

from sklearn.cluster import KMeans



# Yellowbrick for Model visualization

from yellowbrick.cluster import KElbowVisualizer
# Define the functions

def order_clusters(cluster, target, df, ascending):

    new_cluster = 'new' + cluster

    

    temp = df.groupby(cluster)[target].mean().reset_index()

    temp = temp.sort_values(by=target, ascending=ascending).reset_index(drop=True)

    temp['index'] = temp.index

    

    cluster_df = pd.merge(df, temp[[cluster, 'index']], on=cluster)

    cluster_df = cluster_df.drop([cluster], axis=1)

    cluster_df = cluster_df.rename(columns={'index':cluster})

    

    return cluster_df



def rfm_cluster(df, cluster_variable, n_clusters, ascending):

    

    # Create and fit the k-means 

    model = KMeans(n_clusters=n_clusters)

    model.fit(df[[cluster_variable]])

    

    # predict the cluster and pass it to the dataframe

    df[cluster_variable + 'Cluster'] = model.predict(df[[cluster_variable]])

    

    # order the cluster numbers

    df = order_clusters(cluster_variable + 'Cluster', cluster_variable, df, ascending)

    

    return df
# Import the data

df = pd.read_csv(r"../input/onlineretail/OnlineRetail.csv", encoding='cp1252', parse_dates=['InvoiceDate'])



# I'll only keep UK sales

df = df[df.Country == 'United Kingdom']
df.head()
df.info()
maxdate = df['InvoiceDate'].dt.date.max()

mindate = df['InvoiceDate'].dt.date.min()

customers = df['CustomerID'].nunique()

stock = df['StockCode'].nunique()

quantity = df['Quantity'].sum()



print(f'Transactions timeframe: {mindate} to {maxdate}.')

print(f'Unique customers: {customers}.')

print(f'Unique items sold: {stock}.')

print(f'Quantity sold in period {quantity}')
# Create a users dataframe to segment

users = pd.DataFrame(df['CustomerID'].unique())

users.columns = ['CustomerID']
# Get the latest purchase date for each customer and pass it to a df

max_purchase = df.groupby('CustomerID').InvoiceDate.max().reset_index()

max_purchase.columns = ['CustomerID', 'MaxPurchaseDate']

max_purchase['MaxPurchaseDate'] = max_purchase['MaxPurchaseDate'].dt.date



# Calculate Recency

max_purchase['Recency'] = (

    max_purchase['MaxPurchaseDate'].max() - max_purchase['MaxPurchaseDate']).dt.days 



# Merge the dataframe with the users to get the Recency value for each customer

users = pd.merge(users, max_purchase[['CustomerID', 'Recency']], on='CustomerID')

users.head()
# plot a histogram of Recency

fig = plt.figure(figsize=(10, 6))

plt.hist(users['Recency']);
# Describe Recency

users['Recency'].describe()
# calculate the frequency score, that is how 

# frequently the customer buy from the store

frequency_score = df.groupby('CustomerID')['InvoiceDate'].count().reset_index()

frequency_score.columns = ['CustomerID', 'Frequency']
users = pd.merge(users, frequency_score, on='CustomerID')

users.head()
# Plot the distribution 

plt.hist(users['Frequency']);
plt.hist(df['Quantity']);
df.drop(df[df['Quantity']<0].index, axis=0, inplace=True)
# Calculate revenue for each individual customer

df['Revenue'] = df['UnitPrice']*df['Quantity']
# Calculate revenue for each individual customer

df['Revenue'] = df['UnitPrice']*df['Quantity']

revenue = df.groupby('CustomerID')['Revenue'].mean().reset_index()



# Merge the revenue with users dataframe

users = pd.merge(users, revenue, on='CustomerID')
# Plot the data

plt.hist(users['Revenue']);
model = KMeans()



recency_score = users[['Recency']]



visualizer = KElbowVisualizer(model, k=(1, 11))



visualizer.fit(recency_score)

visualizer.show();
# Create the Recency cluster, smaller recency is 

# better, so we set ascending to False

users = rfm_cluster(users, 'Recency', 3, False)



# Check the df with the clusters

users.head()
# Now every customer has been assigned to a cluster based on their Recency

# and the clusters are ordered from lowest to highest

users.groupby('RecencyCluster')['Recency'].describe()
rfm_cluster(users, 'Revenue', 4, True)
users.groupby('RevenueCluster')['Revenue'].describe()
# Create the Frequency Clusters

users = rfm_cluster(users,'Frequency', 3, True)



# describe the clusters

users.groupby('FrequencyCluster')['Frequency'].describe()
# Calculate OverallScore

users['OverallScore'] = users['FrequencyCluster'] + users['RevenueCluster'] - users['RecencyCluster'] 



# Show the mean of the features for each OverallScore value

users.groupby('OverallScore')[['Recency', 'Frequency', 'Revenue']].mean()
# Create the Segment variable based on the OverallScore

x = users['OverallScore']

conditions = [x<0, x<2]

choices = ['Low', 'Medium']



users['Segment'] = np.select(conditions, choices, default='High')
fig = plt.figure(figsize=(10, 6))

plt.title('Segments')

plt.xlabel('Revenue')

plt.ylabel('Frequency')

    

plt.scatter(x=users.query("Segment == 'Low'")['Revenue'],

            y= users.query("Segment == 'Low'")['Frequency'],

            c='green', alpha=0.7, label='Low')



plt.scatter(x=users.query("Segment == 'Medium'")['Revenue'],

            y=users.query("Segment == 'Medium'")['Frequency'],

            c='red', alpha=0.6, label='Medium')



plt.scatter(x=users.query("Segment == 'High'")['Revenue'],

            y=users.query("Segment == 'High'")['Frequency'],

            c='blue', alpha=0.5, label='High')



plt.legend();
fig = plt.figure(figsize=(10, 6))

plt.title('Segments')

plt.xlabel('Frequency')

plt.ylabel('Revenue')

    

plt.scatter(x=users.query("Segment == 'Low'")['Frequency'],

            y= users.query("Segment == 'Low'")['Revenue'],

            c='green', alpha=0.7, label='Low')



plt.scatter(x=users.query("Segment == 'Medium'")['Frequency'],

            y=users.query("Segment == 'Medium'")['Revenue'],

            c='red', alpha=0.6, label='Medium')



plt.scatter(x=users.query("Segment == 'High'")['Frequency'],

            y=users.query("Segment == 'High'")['Revenue'],

            c='blue', alpha=0.5, label='High')



plt.legend();
fig = plt.figure(figsize=(10, 6))

plt.title('Segments')

plt.xlabel('Recency')

plt.ylabel('Revenue')



plt.scatter(x=users.query("Segment == 'Low'")['Recency'],

            y= users.query("Segment == 'Low'")['Revenue'],

            c='green', alpha=0.7, label='Low')



plt.scatter(x=users.query("Segment == 'Medium'")['Recency'],

            y=users.query("Segment == 'Medium'")['Revenue'],

            c='red', alpha=0.6, label='Medium')



plt.scatter(x=users.query("Segment == 'High'")['Recency'],

            y=users.query("Segment == 'High'")['Revenue'],

            c='blue', alpha=0.5, label='High')



plt.legend();

fig = plt.figure(figsize=(10, 6))

plt.title('Segments')

plt.xlabel('Recency')

plt.ylabel('Frequency')

    

plt.scatter(x=users.query("Segment == 'Low'")['Recency'],

            y= users.query("Segment == 'Low'")['Frequency'],

            c='green', alpha=0.7, label='Low')



plt.scatter(x=users.query("Segment == 'Medium'")['Recency'],

            y=users.query("Segment == 'Medium'")['Frequency'],

            c='red', alpha=0.6, label='Medium')



plt.scatter(x=users.query("Segment == 'High'")['Recency'],

            y=users.query("Segment == 'High'")['Frequency'],

            c='blue', alpha=0.5, label='High')



plt.legend();
users.head()