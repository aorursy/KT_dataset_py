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
X_BAL_PUR = df.iloc[:,[0,2]].values

X_BAL_INSTALL_PUR=df.iloc[:,[0,4]].values

X_PUR_CASH_ADVANCE=df.iloc[:,[2,5]].values

X_PUR_PAYMENT=df.iloc[:,[2,13]].values

X_BAL_PYMT=df.iloc[:,[0,13]].values

import scipy.cluster.hierarchy as sch



plt.figure(figsize=(15,6))

plt.title('Dendrogram')

plt.xlabel('BALANCE AND PURCHASES')

plt.ylabel('Euclidean distances')

#plt.grid(True)

dendrogram = sch.dendrogram(sch.linkage(X_BAL_PUR, method = 'ward'))

plt.show()
plt.figure(figsize=(15,6))

plt.title('Dendrogram')

plt.xlabel('BALANCE AND PURCHASES')

plt.ylabel('Euclidean distances')

plt.hlines(y=80000,xmin=0,xmax=2000000,lw=3,linestyles='--')

plt.text(x=1000,y=100000,s='Horizontal line crossing 6 vertical lines so k=7(6+1)',fontsize=20)

#plt.grid(True)

dendrogram = sch.dendrogram(sch.linkage(X_BAL_PUR, method = 'ward'))

plt.show()
plt.figure(figsize=(15,6))

plt.title('Dendrogram')

plt.xlabel('BALANCE AND PYMT')

plt.ylabel('Euclidean distances')

#plt.grid(True)

dendrogram = sch.dendrogram(sch.linkage(X_BAL_PYMT, method = 'ward'))

plt.show()
plt.figure(figsize=(15,6))

plt.title('Dendrogram')

plt.xlabel('BALANCE AND PYMT')

plt.ylabel('Euclidean distances')

plt.hlines(y=100000,xmin=0,xmax=2000000,lw=3,linestyles='--')

plt.text(x=1000,y=120000,s='Horizontal line crossing 5 vertical lines so k=5',fontsize=20)

#plt.grid(True)

dendrogram = sch.dendrogram(sch.linkage(X_BAL_PYMT, method = 'ward'))

plt.show()




plt.figure(figsize=(15,6))

plt.title('Dendrogram')

plt.xlabel('BALANCE AND INSTALLMENT')

plt.ylabel('Euclidean distances')

#plt.grid(True)

dendrogram = sch.dendrogram(sch.linkage(X_BAL_INSTALL_PUR, method = 'ward'))

plt.show()
plt.figure(figsize=(15,6))

plt.title('Dendrogram')

plt.xlabel('BALANCE AND INSTALLMENT')

plt.ylabel('Euclidean distances')

plt.hlines(y=80000,xmin=0,xmax=2000000,lw=3,linestyles='--')

plt.text(x=1000,y=100000,s='Horizontal line crossing 3 vertical lines so k=3',fontsize=20)

#plt.grid(True)

dendrogram = sch.dendrogram(sch.linkage(X_BAL_INSTALL_PUR, method = 'ward'))

plt.show()




plt.figure(figsize=(15,6))

plt.title('Dendrogram')

plt.xlabel('PURCHASE AND CASH ADVANCE')

plt.ylabel('Euclidean distances')

#plt.grid(True)

dendrogram = sch.dendrogram(sch.linkage(X_PUR_CASH_ADVANCE, method = 'ward'))

plt.show()
plt.figure(figsize=(15,6))

plt.title('Dendrogram')

plt.xlabel('PURCHASE AND CASH ADVANCE')

plt.ylabel('Euclidean distances')

plt.hlines(y=80000,xmin=0,xmax=2000000,lw=3,linestyles='--')

plt.text(x=1000,y=100000,s='Horizontal line crossing 7 vertical lines so k=7',fontsize=20)

#plt.grid(True)

dendrogram = sch.dendrogram(sch.linkage(X_PUR_CASH_ADVANCE, method = 'ward'))

plt.show()




plt.figure(figsize=(15,6))

plt.title('Dendrogram')

plt.xlabel('PURCHASE AND PAYMENT')

plt.ylabel('Euclidean distances')

#plt.grid(True)

dendrogram = sch.dendrogram(sch.linkage(X_PUR_PAYMENT, method = 'ward'))

plt.show()
plt.figure(figsize=(15,6))

plt.title('Dendrogram')

plt.xlabel('PURCHASE AND PAYMENT')

plt.ylabel('Euclidean distances')

plt.hlines(y=100000,xmin=0,xmax=2000000,lw=3,linestyles='--')

plt.text(x=1000,y=120000,s='Horizontal line crossing 5 vertical lines so k=5',fontsize=20)

#plt.grid(True)

dendrogram = sch.dendrogram(sch.linkage(X_PUR_PAYMENT, method = 'ward'))

plt.show()
from sklearn.cluster import AgglomerativeClustering

hc_7 = AgglomerativeClustering(n_clusters = 7, affinity = 'euclidean', linkage = 'ward')

hc_5 = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

y_hc_bal_pur = hc_7.fit_predict(X_BAL_PUR)

y_hc_bal_pymt = hc_5.fit_predict(X_BAL_PYMT)

y_hc_pur_pymt = hc_7.fit_predict(X_PUR_PAYMENT)

plt.figure(figsize=(12,7))

plt.scatter(X_BAL_PUR[y_hc_bal_pur == 0, 0], X_BAL_PUR[y_hc_bal_pur == 0, 1], s = 100, c = 'red', label = 'VV Low Balance,VV Low Purchase')

plt.scatter(X_BAL_PUR[y_hc_bal_pur == 1, 0], X_BAL_PUR[y_hc_bal_pur == 1, 1], s = 100, c = 'blue', label = 'High Balance, High Purchase ')

plt.scatter(X_BAL_PUR[y_hc_bal_pur == 2, 0], X_BAL_PUR[y_hc_bal_pur == 2, 1], s = 100, c = 'green', label = 'Medium Balance, Medium Purchase')

plt.scatter(X_BAL_PUR[y_hc_bal_pur == 3, 0], X_BAL_PUR[y_hc_bal_pur == 3, 1], s = 100, c = 'orange', label = 'V High Balance, Low Purchase')

plt.scatter(X_BAL_PUR[y_hc_bal_pur == 4, 0], X_BAL_PUR[y_hc_bal_pur == 4, 1], s = 100, c = 'magenta', label = 'Low Balance,Medium Purchase')

plt.scatter(X_BAL_PUR[y_hc_bal_pur == 5, 0], X_BAL_PUR[y_hc_bal_pur == 5, 1], s = 100, c = 'yellow', label = 'Medium Balance,Medium Purchase')

plt.scatter(X_BAL_PUR[y_hc_bal_pur == 6, 0], X_BAL_PUR[y_hc_bal_pur == 6, 1], s = 100, c = 'pink', label = 'V Low Balance,V low Purchase')

plt.title('Clustering of Balance Vs Purchase',fontsize=20)

plt.xlabel('Balance',fontsize=16)

plt.ylabel('Purchase',fontsize=16)

plt.legend(fontsize=16)

plt.grid(True)

plt.axhspan(ymin=20,ymax=25,xmin=0.4,xmax=0.96,alpha=0.3,color='yellow')

plt.show()
plt.figure(figsize=(12,7))

plt.scatter(X_BAL_PYMT[y_hc_bal_pymt == 0, 0], X_BAL_PYMT[y_hc_bal_pymt == 0, 1], s = 100, c = 'red', label = 'Low Balance average payment')

plt.scatter(X_BAL_PYMT[y_hc_bal_pymt == 1, 0], X_BAL_PYMT[y_hc_bal_pymt == 1, 1], s = 100, c = 'blue', label = 'High Balance,low payment')

plt.scatter(X_BAL_PYMT[y_hc_bal_pymt == 2, 0], X_BAL_PYMT[y_hc_bal_pymt == 2, 1], s = 100, c = 'green', label = 'High Balance high payment')

plt.scatter(X_BAL_PYMT[y_hc_bal_pymt == 3, 0], X_BAL_PYMT[y_hc_bal_pymt == 3, 1], s = 100, c = 'orange', label = 'Average Balance,average payment')

plt.scatter(X_BAL_PYMT[y_hc_bal_pymt == 4, 0], X_BAL_PYMT[y_hc_bal_pymt == 4, 1], s = 100, c = 'magenta', label = 'Low Balance,Average payment')

plt.title('Clustering of Balance vs Payments',fontsize=20)

plt.xlabel('Balance',fontsize=16)

plt.ylabel('Payments',fontsize=16)

plt.legend(fontsize=16)

plt.grid(True)

plt.axhspan(ymin=20,ymax=25,xmin=0.4,xmax=0.96,alpha=0.3,color='yellow')

plt.show()
plt.figure(figsize=(12,7))

plt.scatter(X_PUR_PAYMENT[y_hc_pur_pymt == 0, 0], X_PUR_PAYMENT[y_hc_pur_pymt == 0, 1], s = 100, c = 'red', label = 'Medium purchases average payment')

plt.scatter(X_PUR_PAYMENT[y_hc_pur_pymt == 1, 0], X_PUR_PAYMENT[y_hc_pur_pymt == 1, 1], s = 100, c = 'blue', label = 'average puchase,medium payment')

plt.scatter(X_PUR_PAYMENT[y_hc_pur_pymt == 2, 0], X_PUR_PAYMENT[y_hc_pur_pymt == 2, 1], s = 100, c = 'green', label = 'V.High Puchases V.high payment')

plt.scatter(X_PUR_PAYMENT[y_hc_pur_pymt == 3, 0], X_PUR_PAYMENT[y_hc_pur_pymt == 3, 1], s = 100, c = 'orange', label = 'Low purchase,V.high payment')

plt.scatter(X_PUR_PAYMENT[y_hc_pur_pymt == 4, 0], X_PUR_PAYMENT[y_hc_pur_pymt == 4, 1], s = 100, c = 'magenta', label = 'Low purchase,Low payment')

plt.scatter(X_PUR_PAYMENT[y_hc_pur_pymt == 5, 0], X_PUR_PAYMENT[y_hc_pur_pymt == 5, 1], s = 100, c = 'yellow', label = 'Average purchases,Low payment')

plt.scatter(X_PUR_PAYMENT[y_hc_pur_pymt == 6, 0], X_PUR_PAYMENT[y_hc_pur_pymt == 6, 1], s = 100, c = 'pink', label = 'Low Purchase,high payment')

plt.title('Clustering of Purchases vs Payments',fontsize=20)

plt.xlabel('Purchases',fontsize=16)

plt.ylabel('Payments',fontsize=16)

plt.legend(fontsize=16)

plt.grid(True)

plt.axhspan(ymin=20,ymax=25,xmin=0.4,xmax=0.96,alpha=0.3,color='yellow')

plt.show()