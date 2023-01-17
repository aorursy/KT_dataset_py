# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/CreditCardUsage.csv")
df.head()
df.shape
df.columns
df.dtypes
df.describe()


import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

%matplotlib inline





plt.figure(figsize=(8,5))

plt.title("PURCHASES",fontsize=16)

plt.xlabel ("BALANCE",fontsize=14)

plt.grid(True)

plt.hist(df['BALANCE'],color='orange',edgecolor='k')

plt.show()
plt.figure(figsize=(8,5))

plt.title("Balance and purchase correlation",fontsize=18)

plt.xlabel ("BALANCE",fontsize=14)

plt.ylabel ("PURCHASES_TRX",fontsize=14)

plt.grid(True)

plt.scatter(df['BALANCE'],df['PURCHASES_TRX'],color='red',edgecolor='k',alpha=0.6, s=100)

plt.show()
import scipy.cluster.hierarchy as sch



plt.figure(figsize=(15,6))

plt.title('Dendrogram')

plt.xlabel('BALANCE')

plt.ylabel('PURCHASES_TRX')

#plt.grid(True)

dendrogram = sch.dendrogram(sch.linkage(BALANCE, method = 'ward'))

plt.show()
#Remove Unneccasary column

df.drop('CUST_ID', axis = 1, inplace = True)
sns.heatmap(df.corr(), xticklabels=df.columns, yticklabels=df.columns)
import seaborn as sns
sns.heatmap(df.corr(), xticklabels=df.columns, yticklabels=df.columns)
import matplotlib.pyplot as plt
missing = df.isna().sum()

print(missing)
df = df.fillna( df.median() )

#We use standardScaler() to normalize our dataset.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

Scaled_df = scaler.fit_transform(df)

df_scaled = pd.DataFrame(Scaled_df,columns=df.columns)

df_scaled.head()
#df = df.fillna( df.median() )

# Let's assume we use all cols except CustomerID

vals = df_scaled.iloc[ :, :].values

from sklearn.cluster import KMeans

# Use the Elbow method to find a good number of clusters using WCSS



wcss = []

for i in range( 1, 30 ):

    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300) 

    kmeans.fit_predict( vals )

    wcss.append( kmeans.inertia_ )

   

plt.plot( wcss, 'ro-', label="WCSS")

plt.title("Computing WCSS for KMeans++")

plt.xlabel("Number of clusters")

plt.ylabel("WCSS")

plt.show()
kmeans = KMeans(n_clusters=8, init="k-means++", n_init=10, max_iter=300) 

y_pred = kmeans.fit_predict( vals )

labels = kmeans.labels_

df_scaled["Clus_km"] = labels



# As it's difficult to visualise clusters when the data is high-dimensional - we'll see

# if Seaborn's pairplot can help us see how the clusters are separating out the samples.   

import seaborn as sns

df_scaled["cluster"] = y_pred

cols = list(df_scaled.columns)





sns.lmplot(data=df_scaled,x='BALANCE',y='PURCHASES',hue='Clus_km')







#plt.scatter(X[:,0], X[:,2], c=labels.astype(np.float), alpha=0.5)

#plt.xlabel('BALANCE', fontsize=18)

#plt.ylabel('PURCHASES', fontsize=16)



#sns.pairplot( df[ cols ], hue="cluster")
#Let's choose n=8 clusters. As it's difficult to visualize clusters when we have more than 2-dimensions, we'll see if Seaborn's pairplot can show how the clusters are segmenting the samples.
#using best cols  :



best_cols = ["BALANCE","PURCHASES","CASH_ADVANCE","CREDIT_LIMIT","PAYMENTS","MINIMUM_PAYMENTS"]

kmeans = KMeans(n_clusters=8, init="k-means++", n_init=10, max_iter=300) 

best_vals = df_scaled[best_cols].iloc[ :, :].values

y_pred = kmeans.fit_predict( best_vals )

wcss = []

for i in range( 1, 30 ):

    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300) 

    kmeans.fit_predict( best_vals )

    wcss.append( kmeans.inertia_ )
sns.set_palette('Set2')

sns.scatterplot(df_scaled['BALANCE'],df_scaled['PURCHASES'],hue=labels,palette='Set1')