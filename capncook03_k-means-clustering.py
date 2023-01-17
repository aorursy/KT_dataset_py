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
# reading the dataset



retail_df = pd.read_csv("/kaggle/input/online-retail/OnlineRetail.csv",sep = ",",encoding = "ISO-8859-1",header=0)

retail_df
retail_df.shape
retail_df.info()
retail_df.describe()
# column-wise percentages of missing values



round((retail_df.isnull().sum()/len(retail_df))*100,2)
# dropping the rows having missing values



retail_df.dropna(inplace = True)

retail_df.shape
# creating a new feature Revenue



retail_df['Revenue'] = retail_df['Quantity']*retail_df['UnitPrice']

retail_df
# grouping the dataframe for the monetary metric



grouped = retail_df.groupby(by = 'CustomerID')['Revenue'].sum()

grouped = grouped.reset_index()

grouped
# grouping the dataframe for the frequency metric



frequency = retail_df.groupby(by = 'CustomerID')['InvoiceNo'].count()

frequency = frequency.reset_index()

frequency.columns = ['CustomerID','Frequency']

frequency
grouped = pd.merge(grouped,frequency,on = 'CustomerID',how = 'inner')

grouped
# converting InvoiceDate to date time format



retail_df['InvoiceDate'] = pd.to_datetime(retail_df['InvoiceDate'],format = '%d-%m-%Y %H:%M')

retail_df.info()
max_date = max(retail_df['InvoiceDate'])

max_date
retail_df['Diff'] = max_date - retail_df['InvoiceDate']

retail_df
import datetime as dt

retail_df['Diff'] = retail_df['Diff'].dt.days

retail_df
Recency = retail_df.groupby(by = 'CustomerID')['Diff'].min()

Recency = Recency.reset_index()

Recency.columns = ['CustomerID','Recency']

Recency
grouped = pd.merge(grouped,Recency,on = 'CustomerID',how = 'inner')

grouped
sns.boxplot(y = grouped['Revenue'])
sns.boxplot(y = grouped['Frequency'])
sns.boxplot(y = grouped['Recency'])
# outlier treatment for Revenue



Q1 = grouped.Revenue.quantile(0.05)

Q3 = grouped.Revenue.quantile(0.95)

IQR = Q3 - Q1

grouped = grouped[(grouped.Revenue >= Q1 - 1.5*IQR) & (grouped.Revenue <= Q3 + 1.5*IQR)]



# outlier treatment for Recency



Q1 = grouped.Recency.quantile(0.05)

Q3 = grouped.Recency.quantile(0.95)

IQR = Q3 - Q1

grouped = grouped[(grouped.Recency >= Q1 - 1.5*IQR) & (grouped.Recency <= Q3 + 1.5*IQR)]



# outlier treatment for Frequency



Q1 = grouped.Frequency.quantile(0.05)

Q3 = grouped.Frequency.quantile(0.95)

IQR = Q3 - Q1

grouped = grouped[(grouped.Frequency >= Q1 - 1.5*IQR) & (grouped.Frequency <= Q3 + 1.5*IQR)]
rfm_df = grouped.drop('CustomerID',axis = 1)

rfm_df
import sklearn

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

rfm_df = scaler.fit_transform(rfm_df)

rfm_df
rfm_df = pd.DataFrame(rfm_df)

rfm_df.columns = ['Revenue','Frequency','Recency']
from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

import numpy as np

from math import isnan

 

def hopkins(X):

    d = X.shape[1]

    #d = len(vars) # columns

    n = len(X) # rows

    m = int(0.1 * n) 

    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

 

    rand_X = sample(range(0, n, 1), m)

 

    ujd = []

    wjd = []

    for j in range(0, m):

        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)

        ujd.append(u_dist[0][1])

        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)

        wjd.append(w_dist[0][1])

 

    H = sum(ujd) / (sum(ujd) + sum(wjd))

    if isnan(H):

        print(ujd, wjd)

        H = 0

 

    return H
hopkins(rfm_df)
from sklearn.cluster import KMeans
# clustering with some arbitrary K



kmeans = KMeans(n_clusters = 4,max_iter = 50)

kmeans.fit(rfm_df)
# labels of the data points



kmeans.labels_
# elbow curve



ssd = []

mylist = [2,3,4,5,6,7,8]

for i in mylist:

    kmeans = KMeans(n_clusters = i,max_iter=50)

    kmeans.fit(rfm_df)

    ssd.append({'K':i,'SSD':kmeans.inertia_})



ssd_df = pd.DataFrame(ssd)



plt.plot(ssd_df['K'],ssd_df['SSD'])
from sklearn.metrics import silhouette_score
for i in mylist:

    

    # intialise kmeans

    kmeans = KMeans(n_clusters = i,max_iter = 50)

    kmeans.fit(rfm_df)

    cluster_labels = kmeans.labels_

    

    # silhouette score

    silhouette_avg = silhouette_score(rfm_df,cluster_labels)

    print("For n_clusters={0}, the silhouette score is {1}".format(i,silhouette_avg))
# moving ahead with K = 3



kmeans = KMeans(n_clusters = 3,max_iter = 50)

kmeans.fit(rfm_df)
kmeans.labels_
# assigning the labels



grouped['Cluster'] = kmeans.labels_

grouped
sns.boxplot(y = 'Revenue',x = 'Cluster',data = grouped)
sns.boxplot(y = 'Frequency',x = 'Cluster',data = grouped)
sns.boxplot(y = 'Recency',x = 'Cluster',data = grouped)