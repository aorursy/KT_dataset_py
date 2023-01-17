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
credit_df=pd.read_csv("../input/CreditCardUsage.csv")

credit_df.shape
credit_df.describe().transpose()
credit_df.info()
credit_df.head(10)
credit_df.isna().sum()
credit_df["MINIMUM_PAYMENTS"]=credit_df.MINIMUM_PAYMENTS.fillna(np.mean(credit_df.MINIMUM_PAYMENTS))

credit_df["CREDIT_LIMIT"]=credit_df.CREDIT_LIMIT.fillna(np.mean(credit_df.CREDIT_LIMIT))

credit_df.isna().sum()
credit_df.cov()
credit_corr=credit_df.corr()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
plt.figure(figsize=(20,10))

sns.heatmap(credit_corr,vmin=-1,vmax=1,center=0,annot=True)
df = credit_df.drop('CUST_ID', axis=1)

df.head()
from sklearn.preprocessing import StandardScaler

X = df.values[:]

X = np.nan_to_num(X)

Clus_dataSet = StandardScaler().fit_transform(X)

Clus_dataSet
from sklearn.cluster import KMeans 

from sklearn.datasets.samples_generator import make_blobs 
clusterNum = 7

k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)

k_means.fit(X)

labels = k_means.labels_

centroids = k_means.cluster_centers_

print(labels)

print(centroids)
Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in Nc]

kmeans

score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]

score

plt.plot(Nc,score)

plt.xlabel('Number of Clusters')

plt.ylabel('Score')

plt.title('Elbow Curve')

plt.show()
Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in Nc]

kmeans

score = [kmeans[i].fit(X).inertia_ for i in range(len(kmeans))]

score

plt.plot(Nc,score)

plt.xlabel('Number of Clusters')

plt.ylabel('Sum of within sum square')

plt.title('Elbow Curve')

plt.show()
df["Clus_km"] = labels

df.head(5)
df.columns

#df.iloc[:,:-1]

#df.iloc[:,-1]
plt.figure(figsize=(10,50))

key_col=['BALANCE',  'PURCHASES', 'ONEOFF_PURCHASES',

       'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',  'CASH_ADVANCE_TRX', 'PURCHASES_TRX',

       'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS']

sns.pairplot(data=df,vars=key_col,hue='Clus_km')
#fig, ax = plt.subplots(15, 15,figsize=(30, 40))

#for subplot,tuple_col in zip(ax,col_combination):

    #print(tuple_col[0],tuple_col[1])

    #area = np.pi * ( X[:, 1])**2 

 #   plt.scatter(x=tuple_col[0],y=tuple_col[1], label='Clus_km', alpha=0.5,ax=subplot)

  #  plt.show()

sns.lmplot(data=df,x='BALANCE',y='CASH_ADVANCE',hue='Clus_km')

sns.lmplot(data=df,x='BALANCE',y='CREDIT_LIMIT',hue='Clus_km')

sns.lmplot(data=df,x='BALANCE',y='PAYMENTS',hue='Clus_km')



sns.lmplot(data=df,x='PURCHASES',y='ONEOFF_PURCHASES',hue='Clus_km')

sns.lmplot(data=df,x='PURCHASES',y='INSTALLMENTS_PURCHASES',hue='Clus_km')

sns.lmplot(data=df,x='PURCHASES',y='PAYMENTS',hue='Clus_km')



sns.lmplot(data=df,x='ONEOFF_PURCHASES',y='ONEOFF_PURCHASES_FREQUENCY',hue='Clus_km')

sns.lmplot(data=df,x='ONEOFF_PURCHASES',y='PURCHASES_TRX',hue='Clus_km')

sns.lmplot(data=df,x='ONEOFF_PURCHASES',y='PAYMENTS',hue='Clus_km')



sns.lmplot(data=df,x='INSTALLMENTS_PURCHASES',y='PURCHASES_INSTALLMENTS_FREQUENCY',hue='Clus_km')

sns.lmplot(data=df,x='INSTALLMENTS_PURCHASES',y='PURCHASES_TRX',hue='Clus_km')



sns.lmplot(data=df,x='CASH_ADVANCE',y='CASH_ADVANCE_FREQUENCY',hue='Clus_km')

sns.lmplot(data=df,x='CASH_ADVANCE',y='CASH_ADVANCE_TRX',hue='Clus_km')



sns.lmplot(data=df,x='PURCHASES_FREQUENCY',y='PURCHASES_INSTALLMENTS_FREQUENCY',hue='Clus_km')

sns.lmplot(data=df,x='PURCHASES_FREQUENCY',y='PURCHASES_TRX',hue='Clus_km')





sns.lmplot(data=df,x='ONEOFF_PURCHASES_FREQUENCY',y='PURCHASES_TRX',hue='Clus_km')

sns.lmplot(data=df,x='PURCHASES_INSTALLMENTS_FREQUENCY',y='PURCHASES_TRX',hue='Clus_km')

#area = np.pi * ( X[:, 13])**2  

#colors = ['b', 'c', 'y', 'm', 'r']



#lo = plt.scatter(X[:,0], X[:,2], marker='x', c=labels.astype(np.float))

#ll = plt.scatter(X[:,0], X[:,2], marker='o', c=labels.astype(np.float))

#l  = plt.scatter(X[:,0], X[:,2], marker='o', c=labels.astype(np.float))

#a  = plt.scatter(X[:,0], X[:,2], marker='o', c=labels.astype(np.float))

#h  = plt.scatter(X[:,0], X[:,2], marker='o', c=labels.astype(np.float))

#hh = plt.scatter(X[:,0], X[:,2], marker='o', c=labels.astype(np.float))

#ho = plt.scatter(X[:,0], X[:,2], marker='x', c=labels.astype(np.float))



#plt.legend((lo, ll, l, a, h, hh, ho),

#           ('Low Outlier', 'LoLo', 'Lo', 'Average', 'Hi', 'HiHi', 'High Outlier'),

#           scatterpoints=1,

#           loc='upper right',

#           ncol=3,

#           fontsize=8)



plt.scatter(X[:,0], X[:,2], c=labels.astype(np.float), alpha=0.5)

plt.xlabel('BALANCE', fontsize=18)

plt.ylabel('PURCHASES', fontsize=16)







plt.show()
#area = np.pi * ( X[:, 13])**2  

plt.scatter(X[:,0], X[:,4], c=labels.astype(np.float), alpha=0.5)

plt.xlabel('BALANCE', fontsize=18)

plt.ylabel('INSTALLEMENT_PURCHASES', fontsize=16)



plt.show()
#area = np.pi * ( X[:, 13])**2  

plt.scatter(X[:,0], X[:,12], c=labels.astype(np.float), alpha=0.5)

plt.xlabel('BALANCE', fontsize=18)

plt.ylabel('CREDIT_LIMIT', fontsize=16)



plt.show()
#area = np.pi * ( X[:, 13])**2  

plt.scatter(X[:,0], X[:,13], c=labels.astype(np.float), alpha=0.5)

plt.xlabel('BALANCE', fontsize=18)

plt.ylabel('PAYMENT', fontsize=16)



plt.show()
plt.scatter(X[:,2], X[:,12], c=labels.astype(np.float), alpha=0.5)

plt.xlabel('PURCHASES', fontsize=18)

plt.ylabel('CREDIT_LIMIT', fontsize=16)



plt.show()
#area = np.pi * ( X[:, 13])**2  

plt.scatter(X[:,2], X[:,13], c=labels.astype(np.float), alpha=0.5)

plt.xlabel('PURCHASES', fontsize=18)

plt.ylabel('PAYMENT', fontsize=16)



plt.show()
plt.scatter(X[:,3], X[:,12], c=labels.astype(np.float), alpha=0.5)

plt.xlabel('ONEOFF_PURCHASES', fontsize=18)

plt.ylabel('CREDIT_LIMIT', fontsize=16)



plt.show()
#area = np.pi * ( X[:, 13])**2  

plt.scatter(X[:,3], X[:,13], c=labels.astype(np.float), alpha=0.5)

plt.xlabel('ONEOFF_PURCHASES', fontsize=18)

plt.ylabel('PAYMENT', fontsize=16)



plt.show()
#area = np.pi * ( X[:, 13])**2  

plt.scatter(X[:,4], X[:,12], c=labels.astype(np.float), alpha=0.5)

plt.xlabel('INSTALLEMENT_PURCHASES', fontsize=18)

plt.ylabel('CREDIT_LIMIT', fontsize=16)



plt.show()
#area = np.pi * ( X[:, 13])**2  

plt.scatter(X[:,4], X[:,13], c=labels.astype(np.float), alpha=0.5)

plt.xlabel('INSTALLEMENT_PURCHASES', fontsize=18)

plt.ylabel('PAYMENT', fontsize=16)



plt.show()
plt.scatter(X[:,5], X[:,12], c=labels.astype(np.float), alpha=0.5)

plt.xlabel('CASH_ADVANCE', fontsize=18)

plt.ylabel('CREDIT_LIMIT', fontsize=16)



plt.show()
plt.scatter(X[:,12], X[:,13], c=labels.astype(np.float), alpha=0.5)

plt.xlabel('CREDIT_LIMIT', fontsize=18)

plt.ylabel('PAYMENT', fontsize=16)



plt.show()
df.groupby('Clus_km').mean().sort_values(by='BALANCE')