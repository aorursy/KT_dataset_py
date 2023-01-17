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
import seaborn as sns
df=pd.read_csv("../input/CreditCardUsage.csv")
df.shape
df.info()
df.columns
df.head()
df.describe().T
#Remove Unneccasary column

df.drop('CUST_ID', axis = 1, inplace = True)
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