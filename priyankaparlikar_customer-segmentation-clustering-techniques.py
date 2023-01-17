# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from scipy.stats import zscore

from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.cluster import AgglomerativeClustering
pd.set_option('display.max_columns',None)

pd.set_option('display.max_rows',200)
df=pd.read_csv('../input/ecommerce-data/data.csv',encoding = 'ISO-8859-1')

df.head()
df.shape
df.info()
df.isna().sum()
df=df.drop(df[df['CustomerID'].isna()==True].index,axis=0)
df.shape
df.isna().sum()  #rechecking missing values
df['Country'].value_counts()

# Maximum orders are coming from UK
print(df['Country'].unique())

print('Total no. of countries from where customers belong: ',df['Country'].nunique())
print('Total no. of customers: ',df['CustomerID'].nunique())

print('Total transactions done: ',df['InvoiceNo'].nunique())

print('Products sold are : ',df['StockCode'].nunique())
# Need to check the cancelled orders as well as they are of not use for customer segmentation,

#'C'mentioned before the Invoiceno indicates that the order is cancelled

df[df['InvoiceNo'].apply(lambda x: x[0]=='C')]
percent_transaction_cancelled = round((df[df['InvoiceNo'].apply(lambda x: x[0]=='C')]['InvoiceNo'].nunique()/ df['InvoiceNo'].nunique())*100,2)

print('Percentage of Transactions cancelled are : ',percent_transaction_cancelled)
df=df.drop(df[df['InvoiceNo'].apply(lambda x: x[0]=='C')].index,axis=0)

df.shape
df['Amount'] = df['Quantity'] * df['UnitPrice']
a=df.groupby('CustomerID').sum()['Amount']

a= a.reset_index()
a.head()
b= df.groupby('CustomerID')['InvoiceNo'].count()

b= b.reset_index()

b.columns = ['CustomerID', 'Frequency']
b.head()
df1= pd.merge(a, b, on='CustomerID', how='inner')

df1.head()
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

max_date = df['InvoiceDate'].max()

max_date
df['days_diff'] = max_date - df['InvoiceDate']

df.head()
c = df.groupby('CustomerID')['days_diff'].min()

c = c.reset_index()

c.head()
c['days_diff'] = c['days_diff'].dt.days

c.head()
df1 = pd.merge(df1, c, on='CustomerID', how='inner')

df1.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']

df1.head()
df2=df1[['Amount','Frequency','Recency']]   # using only Amount, Frequency and Recency to find the customer segments.
df1_scaled = df2.apply(zscore)    # scaling of data is required as all the calculations is based on distance

df1_scaled.head()
kmeans = KMeans(random_state=2)

kmeans.fit(df1_scaled)
cluster_range = range( 1, 15 )

cluster_errors = []

for num_clusters in cluster_range:

    clusters = KMeans( num_clusters, n_init = 10 )

    clusters.fit(df1_scaled)

    cluster_errors.append( clusters.inertia_ )

clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

clusters_df[0:15]
plt.figure(figsize=(12,6))

plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
kmeans = KMeans(n_clusters=3, n_init = 15, random_state=2)

kmeans.fit(df1_scaled)

centroids = kmeans.cluster_centers_

centroid_df = pd.DataFrame(centroids, columns = list(df1_scaled) )

centroid_df
df_labels = pd.DataFrame(kmeans.labels_ , columns = list(['labels']))

df_labels['labels'] = df_labels['labels'].astype('category')
df_kmeans = df1.join(df_labels)

df_kmeans.head()
sns.pairplot(df_kmeans,diag_kind='kde',hue='labels')
Z = linkage(df1_scaled, method='ward',metric='euclidean')

plt.figure(figsize=(25, 10))

plt.title('Hierarchical Clustering Dendrogram')

plt.xlabel('sample index')

plt.ylabel('distance')

dendrogram(

    Z,

    leaf_rotation=90.,  # rotates the x axis labels

    leaf_font_size=8.,  # font size for the x axis labels

)

plt.show()
plt.title('Hierarchical Clustering Dendrogram (truncated)')

plt.xlabel('sample index')

plt.ylabel('distance')

dendrogram(

    Z,

    truncate_mode='lastp',  # show only the last p merged clusters

    p=12,  # show only the last p merged clusters

    show_leaf_counts=False,  # otherwise numbers in brackets are counts

    leaf_rotation=90.,

    leaf_font_size=12.,

    show_contracted=True,  # to get a distribution impression in truncated branches

)

plt.show()
hie_clus = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

labels = hie_clus.fit_predict(df1_scaled)



df_h = df1.copy(deep=True)

df_h['label'] = labels

df_h['label']=df_h['label'].astype('category')

df_h.head()
sns.pairplot(df_h,diag_kind='kde',hue='label')