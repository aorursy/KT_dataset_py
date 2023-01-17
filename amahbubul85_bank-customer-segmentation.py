import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler,normalize

from sklearn.decomposition import PCA
creditcard_df=pd.read_csv('../input/bank-customer/marketing_data.csv')



# CUSTID: Identification of Credit Card holder 

# BALANCE: Balance amount left in customer's account to make purchases

# BALANCE_FREQUENCY: How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)

# PURCHASES: Amount of purchases made from account

# ONEOFFPURCHASES: Maximum purchase amount done in one-go

# INSTALLMENTS_PURCHASES: Amount of purchase done in installment

# CASH_ADVANCE: Cash in advance given by the user

# PURCHASES_FREQUENCY: How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)

# ONEOFF_PURCHASES_FREQUENCY: How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)

# PURCHASES_INSTALLMENTS_FREQUENCY: How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)

# CASH_ADVANCE_FREQUENCY: How frequently the cash in advance being paid

# CASH_ADVANCE_TRX: Number of Transactions made with "Cash in Advance"

# PURCHASES_TRX: Number of purchase transactions made

# CREDIT_LIMIT: Limit of Credit Card for user

# PAYMENTS: Amount of Payment done by user

# MINIMUM_PAYMENTS: Minimum amount of payments made by user  

# PRC_FULL_PAYMENT: Percent of full payment paid by user

# TENURE: Tenure of credit card service for user
creditcard_df.head()
creditcard_df.info()
creditcard_df.describe()
sns.heatmap(creditcard_df.isnull(),yticklabels = False,cbar=True,cmap='Blues')
creditcard_df.isnull().sum()
creditcard_df.MINIMUM_PAYMENTS.fillna(value=creditcard_df.MINIMUM_PAYMENTS.mean(),inplace=True)
creditcard_df.isnull().sum()
creditcard_df.CREDIT_LIMIT.fillna(value=creditcard_df.CREDIT_LIMIT.mean(),inplace=True)
creditcard_df.isnull().sum()
creditcard_df.duplicated().sum()
creditcard_df.drop('CUST_ID',axis=1,inplace=True)
creditcard_df.columns
# distplot combines the matplotlib.hist function with seaborn kdeplot()

# KDE Plot represents the Kernel Density Estimate

# KDE is used for visualizing the Probability Density of a continuous variable. 

# KDE demonstrates the probability density at different values in a continuous variable. 



# Mean of balance is $1500

# 'Balance_Frequency' for most customers is updated frequently ~1

# For 'PURCHASES_FREQUENCY', there are two distinct group of customers

# For 'ONEOFF_PURCHASES_FREQUENCY' and 'PURCHASES_INSTALLMENT_FREQUENCY' most users don't do one off puchases or installment purchases frequently 

# Very small number of customers pay their balance in full 'PRC_FULL_PAYMENT'~0

# Credit limit average is around $4500

# Most customers are ~11 years tenure



plt.figure(figsize=(10,50))

for ii in range(len(creditcard_df.columns)):

  plt.subplot(17,1,ii+1)

  sns.distplot(creditcard_df.iloc[:,ii],kde_kws={"color": "b", "lw": 3, "label": "KDE"}, hist_kws={"color": "g"})

plt.tight_layout()

plt.figure(figsize=(14,6))

sns.heatmap(creditcard_df.corr(),annot=True,)
scale=StandardScaler()

creditcard_df_scaled=scale.fit_transform(creditcard_df)
creditcard_df_scaled
creditcard_df_scaled.shape
model=KMeans(n_clusters=3)

model.fit_transform(creditcard_df_scaled)
model.inertia_
scores=[]

for ii in range(1,20):

  model=KMeans(n_clusters=ii)

  model.fit_transform(creditcard_df_scaled)

  scores.append(model.inertia_)



plt.figure(figsize=(10,15))

plt.plot(scores,'bx-')
model_opt=KMeans(n_clusters=7)

model_opt.fit(creditcard_df_scaled)
model_opt.labels_
cluster_centres=pd.DataFrame(data=model_opt.cluster_centers_,columns=creditcard_df.columns)

cluster_centres
cluster_centres=scale.inverse_transform(cluster_centres)

cluster_centres
cluster_centres=pd.DataFrame(data=cluster_centres,columns=creditcard_df.columns)

cluster_centres
labels=pd.DataFrame(data=model_opt.labels_,columns=['label'])
labels
labels=model_opt.fit_predict(creditcard_df_scaled)
labels
labels_df=pd.DataFrame(data=labels,columns=['labels'])

labels_df
creditcard_df_concated=pd.concat([creditcard_df,labels_df],axis=1)

creditcard_df_concated
for ii in creditcard_df_concated.columns:

  cluster=creditcard_df_concated.loc[creditcard_df_concated['labels']==0,creditcard_df_concated.columns[ii]]

  sns.distplot(cluster,kde=False)
for ii in creditcard_df_concated.columns:

  plt.figure(figsize=(35,5))

  for cluster_no in range(7):

    plt.subplot(1,7,cluster_no+1)

    cluster=creditcard_df_concated.loc[creditcard_df_concated['labels']==cluster_no,ii]

    cluster.hist(bins=20)

    plt.title('{}    \nCluster {} '.format(ii,cluster_no))

  

  plt.show()
pca=PCA(n_components=2)

principal_comp=pca.fit_transform(creditcard_df_scaled)
pca_df = pd.DataFrame(data = principal_comp, columns =['pca1','pca2'])

pca_df.shape

pca_df
pca_df=pd.concat([pca_df,labels_df],axis=1)

pca_df.head()
plt.figure(figsize=(10,10))

sns.scatterplot(x="pca1",y="pca2",data=pca_df,hue=labels,palette =['red','green','blue','pink','yellow','gray','purple'])

plt.show()