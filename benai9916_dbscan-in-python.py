import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# import required libraries for clustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('/kaggle/input/online-retail-customer-clustering/OnlineRetail.csv', sep=",", encoding="ISO-8859-1", header=0)
# first five row
df.head()
# size of datset
df.shape
# statistical summary of numerical variables
df.describe()
# summary about dataset
df.info()
# check for missing values
df.isna().sum() / df.shape[0] * 100
# Droping rows having missing values

df = df.dropna()
df.shape
# New Attribute : Monetary

df['Amount'] = df['Quantity']*df['UnitPrice']

rfm_m = df.groupby('CustomerID')['Amount'].sum()
rfm_m = rfm_m.reset_index()
rfm_m.head()
# New Attribute : Frequency

rfm_f = df.groupby('CustomerID')['InvoiceNo'].count()

rfm_f = rfm_f.reset_index()
rfm_f.columns = ['CustomerID', 'Frequency']
rfm_f.head()
# Merging the two dfs

rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
rfm.head()
# New Attribute : Recency

# Convert to datetime to proper datatype

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'],format='%d-%m-%Y %H:%M')
# Compute the maximum date to know the last transaction date

max_date = max(df['InvoiceDate'])
max_date
# Compute the difference between max date and transaction date

df['Diff'] = max_date - df['InvoiceDate']
df.head()
# Compute last transaction date to get the recency of customers

rfm_p = df.groupby('CustomerID')['Diff'].min()

rfm_p = rfm_p.reset_index()
rfm_p.head()
# Extract number of days only

rfm_p['Diff'] = rfm_p['Diff'].dt.days
rfm_p.head()
# Merge tha dataframes to get the final RFM dataframe

rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')

rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']
rfm.head()
# Outlier Analysis of Amount Frequency and Recency

attributes = ['Amount','Frequency','Recency']
plt.rcParams['figure.figsize'] = [10,8]
sns.boxplot(data = rfm[attributes], orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)
plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')
plt.ylabel("Range", fontweight = 'bold')
plt.xlabel("Attributes", fontweight = 'bold')
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
rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']

rfm_df_scaled.head()
# create an object
db = DBSCAN(eps=0.8, min_samples=7, metric='euclidean')

# fit the model
db.fit(rfm_df_scaled)
# Cluster labled
db.labels_
from sklearn.metrics import silhouette_score

cluster_labels = db.labels_   

# silhouette score
silhouette_avg = silhouette_score(rfm_df_scaled, cluster_labels)
print("The silhouette score is", format(silhouette_avg))

rfm_df_scaled['label']=db.labels_

rfm_df_scaled.head()
for c in rfm_df_scaled.columns[:-1]:
    plt.figure(figsize=(6,4))
    sns.boxplot(data=rfm_df_scaled, y=c, x='label')
    plt.show()