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
df = pd.read_csv(r'/kaggle/input/ccdata/CC GENERAL.csv')
df.head()
df.drop('CUST_ID',axis=1,inplace=True)
df.shape
df.info()
df.isnull().sum()
df['MINIMUM_PAYMENTS'].describe()
df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].mean(),inplace=True)
df['CREDIT_LIMIT'].describe()
df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].mean(),inplace=True)
df.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns

corr=df.corr()

top_features=corr.index

plt.figure(figsize=(20,20))

sns.heatmap(df[top_features].corr(),annot=True)
df.drop(['PURCHASES','PURCHASES_FREQUENCY','CASH_ADVANCE_FREQUENCY'],axis=1,inplace=True)
fig, ax = plt.subplots(figsize=(15,10))

sns.boxplot(data=df, width= 0.5,ax=ax,  fliersize=3)

df.shape
from scipy import stats

import numpy as np

z = np.abs(stats.zscore(df))

print(z)
threshold = 3

print(np.where(z > 3))
df1 = df[(z < 3).all(axis=1)]
df1.shape
fig, ax = plt.subplots(figsize=(15,10))

sns.boxplot(data=df1, width= 0.5,ax=ax,  fliersize=3)
plt.figure(figsize=(20,25), facecolor='white')

plotnumber = 1



for column in df1:

    if plotnumber<=14 :

        ax = plt.subplot(7,2,plotnumber)

        sns.kdeplot(df1[column],bw=1.5)

        plt.xlabel(column,fontsize=20)

    plotnumber+=1

plt.show()
df1.head()
df1['log_balance']=np.log(1+df1['BALANCE'])

df1['log_oneoff_purchases']=np.log(1+df1['ONEOFF_PURCHASES'])

df1['log_installments_purchases']=np.log(1+df1['INSTALLMENTS_PURCHASES'])

df1['log_cash_advance']=np.log(1+df1['CASH_ADVANCE'])

df1['log_cash_advance_trx']=np.log(1+df1['CASH_ADVANCE_TRX'])

df1['log_purchases_trx']=np.log(1+df1['PURCHASES_TRX'])

df1['log_credit_limit']=np.log(1+df1['CREDIT_LIMIT'])

df1['log_payments']=np.log(1+df1['PAYMENTS'])

df1['log_minimum_payments']=np.log(1+df1['MINIMUM_PAYMENTS'])
df1.drop(['BALANCE','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES','CASH_ADVANCE','CASH_ADVANCE_TRX','PURCHASES_TRX','CREDIT_LIMIT','PAYMENTS','MINIMUM_PAYMENTS'],axis=1,inplace=True)
df1.head()
plt.figure(figsize=(20,25), facecolor='white')

plotnumber = 1



for column in df1:

    if plotnumber<=14 :

        ax = plt.subplot(7,2,plotnumber)

        sns.kdeplot(df1[column],bw=1.5)

        plt.xlabel(column,fontsize=20)

    plotnumber+=1

plt.show()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(df1)

X.shape
from sklearn.cluster import KMeans

wcss=[]

for i in range (1,12):

    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=40)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.plot(range(1,12),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 50)

y_kmeans = kmeans.fit_predict(X)

print(y_kmeans)
labels = kmeans.labels_

labels
from sklearn.decomposition import PCA

pca = PCA(2)

principalComponents = pca.fit_transform(X)

x, y = principalComponents[:, 0], principalComponents[:, 1]

print(principalComponents.shape)



colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'purple'}
final_df = pd.DataFrame({'x': x, 'y':y, 'label':labels}) 

groups = final_df.groupby(labels)
fig, ax = plt.subplots(figsize=(15, 10)) 



for name, group in groups:

    ax.plot(group.x, group.y, marker='o', linestyle='', ms=5, color=colors[name], mec='none')

    ax.set_aspect('auto')

    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')

    

ax.set_title("Customer Segmentation based on Credit Card usage")

plt.show()