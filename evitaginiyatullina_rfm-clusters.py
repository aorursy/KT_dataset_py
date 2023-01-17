# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv(r'../input/customer-sales-transactions-of-retail-store/Customer_Sales_Transactional_Data_CSV.txt', encoding='cp1251', sep =',')
df.head()
df.rename(columns={'SALES_dATE': 'sale_date', 'CUSTOMER_ID': 'customer_id', 'SALES_AMOUNT':'sale_amount'}, inplace=True)
df.head()
df.drop_duplicates(inplace=True)
df.isnull().sum()
df['sale_date']= pd.to_datetime(df['sale_date'])
print(df.sale_date.max())
print(df.sale_date.min())
cus_df=pd.DataFrame(df['customer_id'].unique())
cus_df.columns = ['customer_id']
cus_df
max_purchase = df.groupby('customer_id').sale_date.max().reset_index()
max_purchase.columns = ['customer_id','MaxPurchaseDate']
max_purchase
max_purchase['Recency'] = (max_purchase['MaxPurchaseDate'].max() - max_purchase['MaxPurchaseDate']).dt.days
max_purchase
cus_df = pd.merge(cus_df, max_purchase[['customer_id','Recency']], on='customer_id')

cus_df.head(10)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
sse={}
recency = cus_df[['Recency']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(recency)
    recency["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()), marker='o')
plt.xlabel("Number of cluster")
plt.show()
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

frequency = df.groupby('customer_id').sale_date.count().reset_index()
frequency.columns = ['customer_id','Frequency']

#add this data to our main dataframe
cus_df = pd.merge(cus_df, frequency, on='customer_id')
cus_df
monetary = df.groupby('customer_id').sale_amount.sum().reset_index()
monetary.columns = ['customer_id','Monetary']
cus_df = pd.merge(cus_df, monetary, on='customer_id')
cus_df
kmeans = KMeans(n_clusters=4)
kmeans.fit(cus_df[['Recency']])
cus_df['RecencyCluster'] = kmeans.predict(cus_df[['Recency']])
cus_df = order_cluster('RecencyCluster', 'Recency',cus_df,True)
cus_df['FrequencyCluster'] = kmeans.predict(cus_df[['Frequency']])
cus_df = order_cluster('FrequencyCluster', 'Frequency',cus_df,False)
cus_df['MonetaryCluster'] = kmeans.predict(cus_df[['Monetary']])
cus_df = order_cluster('MonetaryCluster', 'Monetary',cus_df,False)
cus_df
print(cus_df.groupby('RecencyCluster')['Recency'].describe())
print(cus_df.groupby('FrequencyCluster')['Frequency'].describe())
print(cus_df.groupby('MonetaryCluster')['Monetary'].describe())
cus_df['OverallScore'] = cus_df['RecencyCluster'] + cus_df['FrequencyCluster'] + cus_df['MonetaryCluster']
cus_df.sort_values('OverallScore')
cus_df.groupby('OverallScore')['Recency','Frequency','Monetary'].mean()
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(cus_df['Recency'], cus_df['Frequency'], cus_df['Monetary'], c=cus_df['OverallScore'], cmap="plasma" )
pyplot.legend()
pyplot.show()