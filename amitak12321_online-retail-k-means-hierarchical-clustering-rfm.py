#importing the requird liberaries for dataframe and the data visiulazition

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from IPython.display import Image

#importing ML libraries for clustering

import sklearn
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
#reading the input data
df = pd.read_csv('../input/onlineretail/OnlineRetail.csv',sep = ",",encoding="ISO-8859-1",header = 0)
df.head()
#describe the dataframe.....so here we can seee the df(dataframe) has 8 rows and 541909 rows (entries)
# And will get the full information about the dataframe, like:- count of values in all the columns, their data type
df.info()
#describing the dataframe
#with the help of this describe() function will get breakdown of the dataset.
#here we can see that the Quantity is showing -80995(min) and the max(80995) so it seems there is something wrong with the dataframe so we have
##perform data cleansing to clean the data.
df.describe()
#as we saw above that we have null values in Description and customer id columns in Describe() function.
#now we are going to see the percentage of null values in our dataset

df_null = round(100*(df.isnull().sum())/len(df),2)
df_null
#droping rows that has the missing values because we can not replace the null in Customer ID with any value.
df = df.dropna()
df.shape
#now we can see that there is no null values in our data
df.info()
#we have to change the data type of the customerid as per Business Understanding
df['CustomerID'] = df['CustomerID'].astype(str)
#we have unit price and the qty in our data so we are going to create anothere variable ammount
df['Amount'] = df['UnitPrice']*df['Quantity']
#now grouping the data based on customer id
rfm_m = df.groupby('CustomerID')['Amount'].sum()
rfm_m = rfm_m.reset_index()
rfm_m.head()
#now we are going to count the InvoiceNo variable to create another variable FREQUENCY 
rfm_f= df.groupby('CustomerID')['InvoiceNo'].count()
rfm_f = rfm_f.reset_index()
rfm_f.columns = ['CustomerID', 'Frequency']
rfm_f.head()
#now we are going to merge these to df what we have just created
rfm = pd.merge(rfm_m,rfm_f,on='CustomerID', how='inner')
rfm.head()
#to calculate the receancy we need to calculate the latest date in our dataset
#and before that we need to convert the date in appropriate  format.
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'],format='%d-%m-%Y %H:%M')
l_date = max(df['InvoiceDate'])
print(l_date)
#now we are going to check the difference between the max date and the invoice date.
df['Difference'] = l_date - df['InvoiceDate']
df.head()
#now we are calculating the minimum date gap per cuctomer to get the recency.
rfm_r = df.groupby('CustomerID')['Difference'].min()
rfm_r =rfm_r.reset_index()
rfm_r.head()
#now we are going to extract number of days only from the difference variable
#run this code block once otherwise it will throw error because the data will get change once you will run it.
rfm_r['Difference'] = rfm_r['Difference'].dt.days

rfm_r.head()
#Now we are going to merge this rfm_r to the main rfm dataset
rfm = pd.merge(rfm,rfm_r,on='CustomerID', how='inner')
rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']
rfm.head()
#Outlier analysis of Amount, Frequency & Recency Features
attributes = ['Amount','Frequency','Recency']
plt.rcParams['figure.figsize'] = [10,8]
sns.boxplot(data = rfm[attributes], orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)
plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')
plt.ylabel("Range", fontweight = 'bold')
plt.xlabel("Attributes", fontweight = 'bold')
#treating outliers
#as you can see there are outliers in above figure...so we need to treat them.
# Removing (statistical) outliers for Amount
Q1 = rfm.Amount.quantile(0.05)
Q3 = rfm.Amount.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Amount >= Q1 - 1.5*IQR) & (rfm.Amount <= Q3 + 1.5*IQR)]

# Removing (statistical) outliers for Recency
Q1 = rfm.Recency.quantile(0.05)
Q3 = rfm.Recency.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Recency >= Q1 - 1.5*IQR) & (rfm.Recency <= Q3 + 1.5*IQR)]

# Removing (statistical) outliers for Frequency
Q1 = rfm.Frequency.quantile(0.05)
Q3 = rfm.Frequency.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Frequency >= Q1 - 1.5*IQR) & (rfm.Frequency <= Q3 + 1.5*IQR)]
# Rescaling the attributes

rfm_df = rfm[['Amount', 'Frequency', 'Recency']]

# Instantiate
scaler = StandardScaler()

# fit_transform
rfm_df_scaled = scaler.fit_transform(rfm_df)
rfm_df_scaled.shape
#converting it to dataframe and adding colummn names
rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']
rfm_df_scaled.head()
# k-means with some arbitrary k

kmeans = KMeans(n_clusters=4, max_iter=50)
kmeans.fit(rfm_df_scaled)
kmeans.labels_
# Elbow-curve/SSD

ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)
    
    ssd.append(kmeans.inertia_)
    
# plot the SSDs for each n_clusters
plt.plot(ssd)
# Silhouette Analysis

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(rfm_df_scaled, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
#so from this output we are going to buildthe model on 3 clusters
# Final model with k=3
kmeans = KMeans(n_clusters=3, max_iter=50)
kmeans.fit(rfm_df_scaled)
kmeans.labels_
#adding clusters to our df
# assign the label
rfm['Cluster_Id'] = kmeans.labels_
rfm.head()
# Box plot to visualize Cluster Id vs Amount

sns.boxplot(x='Cluster_Id', y='Amount', data=rfm)
# Box plot to visualize Cluster Id vs Frequency

sns.boxplot(x='Cluster_Id', y='Frequency', data=rfm)
# Box plot to visualize Cluster Id vs Recency

sns.boxplot(x='Cluster_Id', y='Recency', data=rfm)
#single linkage
Image("../input/linkage/single.png")
# Single linkage: 

mergings = linkage(rfm_df_scaled, method="single", metric='euclidean')
dendrogram(mergings)
plt.show()
#Complete linkage
Image("../input/linkage/mplete.png")
# Complete linkage

mergings = linkage(rfm_df_scaled, method="complete", metric='euclidean')
dendrogram(mergings)
plt.show()
# average linkage
Image("../input/linkage/average.png")
# Average linkage

mergings = linkage(rfm_df_scaled, method="average", metric='euclidean')
dendrogram(mergings)
plt.show()
# 3 clusters
cluster_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )
cluster_labels
# Assign cluster labels

rfm['Cluster_Labels'] = cluster_labels
rfm.head()
#Plot Cluster labels vs Amount

sns.boxplot(x='Cluster_Labels', y='Amount', data=rfm)
#Plot Cluster labels vs Frequency

sns.boxplot(x='Cluster_Labels', y='Frequency', data=rfm)
#Cluster labels vs Recency
sns.boxplot(x='Cluster_Labels', y='Recency',data=rfm)
