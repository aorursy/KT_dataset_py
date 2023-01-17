#Loading data

import pandas as pd

transaction_data_aug_2018=pd.read_csv('../input/RFM201808.csv')

transaction_data_sep_2018=pd.read_csv('../input/RFM201809.csv')

transaction_data_oct_2018=pd.read_csv('../input/RFM201810.csv')

transaction_data=pd.concat([transaction_data_aug_2018,transaction_data_sep_2018,transaction_data_oct_2018],ignore_index=True)

transaction_data.head()
#checking data types of the columns

transaction_data.info()
#checking shape of the dataset

transaction_data.shape
#checking duplicate

transaction_data.duplicated().value_counts()
#checking for missing values

transaction_data.isnull().sum()
# Summary Statistics of numeric feature

transaction_data.describe()
#checking return invoice

return_invoice_number=transaction_data.query('Amount<0')['Amount'].count()

print("Return Invoice No: %d"%return_invoice_number)
#plotting distribution of Invoice Amount

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

plt.figure(figsize=(12,5))

sns.boxplot(transaction_data['Amount'],orient='v')

plt.title('Distribution of Invoice Amount')

plt.show()
#plotting distribution of Invoice Amount

plt.figure(figsize=(12,5))

sns.distplot(transaction_data['Amount'])

plt.title('Distribution of Invoice Amount')

plt.show()
#removing outliers by z-score method



import numpy as np

transaction_data=transaction_data[np.abs(transaction_data.Amount-transaction_data.Amount.mean()) <= (3*transaction_data.Amount.std())]
#plotting distribution of Amount after outliers removal

plt.figure(figsize=(12,5))

sns.boxplot(transaction_data['Amount'],orient='v')

plt.title('Distribution of Invoice Amount after outliers removing')

plt.show()
#plotting distribution of Invoice Amount after outliers removal

plt.figure(figsize=(12,5))

sns.distplot(transaction_data['Amount'],color='red')

plt.title('Distribution of Invoice Amount after outliers detection')

plt.show()
#converting to invoiceDate to date time type

transaction_data['InvoiceDate']=transaction_data['InvoiceDate'].apply(pd.to_datetime)
#Building recency feature

import datetime

reference_date = transaction_data.InvoiceDate.max()

reference_date = reference_date + datetime.timedelta(days = 1)

transaction_data['days_since_last_purchase'] = reference_date - transaction_data.InvoiceDate

transaction_data['days_since_last_purchase_num'] = transaction_data['days_since_last_purchase'].astype('timedelta64[D]')
transaction_data.head()
customer_history_df = transaction_data.groupby("CustomerCode").min().reset_index()[['CustomerCode', 'days_since_last_purchase_num']]

customer_history_df.rename(columns={'days_since_last_purchase_num':'recency'}, inplace=True)

customer_history_df.head()
#building frequency and monetary feaures

customer_monetary_val = transaction_data[['CustomerCode', 'Amount']].groupby("CustomerCode").sum().reset_index()

customer_history_df = customer_history_df.merge(customer_monetary_val, how='outer')

customer_history_df.Amount = customer_history_df.Amount+0.001

customer_freq = transaction_data[['CustomerCode', 'Amount']].groupby("CustomerCode").count().reset_index()

customer_freq.rename(columns={'Amount':'frequency'},inplace=True)

customer_history_df = customer_history_df.merge(customer_freq, how='outer')

customer_history_df.head()
#plotting recncy Distribution

plt.figure(figsize=(12,5))

sns.distplot(customer_history_df['recency'])

plt.title('Recency Distrubtion')

plt.show()
#plotting recncy Distribution

plt.figure(figsize=(12,5))

sns.boxplot(customer_history_df['recency'],orient='v')

plt.title('Recency Distrubtion')

plt.show()
#plotting Amount Distribution

plt.figure(figsize=(12,5))

sns.distplot(customer_history_df['Amount'])

plt.title('Distrubtion Of Monetary Value')

plt.show()
#plotting Amount Distribution

plt.figure(figsize=(12,5))

sns.boxplot(customer_history_df['Amount'],orient='v')

plt.title('Distrubtion Of Monetary Value')

plt.show()
#plotting frequency Distribution

plt.figure(figsize=(12,5))

sns.distplot(customer_history_df['frequency'])

plt.title('Distrubtion Of Frequency Value')

plt.show()
#plotting Amount Distribution

plt.figure(figsize=(12,5))

sns.boxplot(customer_history_df['frequency'],orient='v')

plt.title('Distrubtion Of Frequency')

plt.show()
#removing outliers 

customer_history_df=customer_history_df[np.abs(customer_history_df.Amount-customer_history_df.Amount.mean()) <= (3*customer_history_df.Amount.std())]

customer_history_df=customer_history_df[np.abs(customer_history_df.frequency-customer_history_df.frequency.mean()) <= (3*customer_history_df.frequency.std())]
#plotting Amount Distribution after outliers removal

plt.figure(figsize=(12,5))

sns.boxplot(customer_history_df['Amount'],orient='v')

plt.title('Distrubtion Of Monetary Value after outliers removal')

plt.show()
#plotting frequency Distribution

plt.figure(figsize=(12,5))

sns.boxplot(customer_history_df['frequency'],orient='v')

plt.title('Distrubtion Of Frequency after outliers removal')

plt.show()
#taking log of RFM Feature

customer_history_df['recency_log'] = customer_history_df['recency'].apply(np.log)

customer_history_df['frequency_log'] = customer_history_df['frequency'].apply(np.log)

customer_history_df['amount_log'] = customer_history_df['Amount'].apply(np.log)
#3d scatter plot of recency, frequency, monetary value

from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111, projection='3d')



xs =customer_history_df.recency_log

ys = customer_history_df.frequency_log

zs = customer_history_df.amount_log

ax.scatter(xs, ys, zs, s=15)



ax.set_xlabel('Recency')

ax.set_ylabel('Frequency')

ax.set_zlabel('Monetary')

plt.title('3d scatter plot of recency, frequency, monetary value')

plt.show()
#checking for null values

customer_history_df.isnull().sum()
#removing null

customer_history_df=customer_history_df.dropna()
#summary statidtics

customer_history_df.describe()
#selecting feature

feature=['recency_log','frequency_log','amount_log']

X=customer_history_df[feature].values
#scaling the feature

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

scaler.fit(X)

X=scaler.transform(X)
#checking feature array

X
#elbow method to select optimum cluster number

from sklearn.cluster import KMeans

distortions = []

for i in range(1, 11):

  km = KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)

  km.fit(X)

  distortions.append(km.inertia_)

plt.plot(range(1,11), distortions, marker='o')

plt.xlabel('Number of clusters')

plt.ylabel('Distortion')

plt.show()
#KMeans Clustering

from sklearn.cluster import KMeans

clustering=KMeans(n_clusters=7,init='k-means++',n_init=10,max_iter=1000,random_state=0).fit(X)

print("Clusters:",np.unique(clustering.labels_))

s=pd.Series(clustering.labels_)

print(s.value_counts())

print(len(s))
#Assigning Cluster Label

customer_history_df['Cluster_Label']=clustering.labels_
#Difference in recency among clusters

plt.figure(figsize=(12,7))

sns.boxplot(x='Cluster_Label',y='recency',data=customer_history_df)

plt.title('Difference in recency among clusters')

plt.show()
#Difference in monetary value among clusters

plt.figure(figsize=(12,7))

sns.boxplot(x='Cluster_Label',y='Amount',data=customer_history_df)

plt.title('Difference in monetary value among clusters')

plt.show()
#Difference in frequency among clusters

plt.figure(figsize=(12,7))

sns.boxplot(x='Cluster_Label',y='frequency',data=customer_history_df)

plt.title('Difference in frequency among clusters')

plt.show()
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D,axes3d





figure=plt.figure(figsize=(12,7))

ax=Axes3D(figure,elev=30,azim=150)



figure.set_size_inches(10,10)

for i,colors in zip(range(0,7),['green','blue','red','cyan','magenta','yellow','black']):

        ax.scatter(customer_history_df.loc[:,['recency_log']][customer_history_df.Cluster_Label==i], customer_history_df.loc[:,['frequency']][customer_history_df.Cluster_Label==i], customer_history_df.loc[:,['amount_log']][customer_history_df.Cluster_Label==i], s=5,color=colors)



ax.set_xlabel('Recency')

ax.set_ylabel('Frequency')

ax.set_zlabel('Monetary')

plt.legend(map(lambda x:str(x),range(0,7)))

plt.title('3d scatter plot of recency, frequency, monetary value of clusters')



plt.show()
#Cluster 0

figure=plt.figure()

ax=Axes3D(figure,elev=20,azim=160)



figure.set_size_inches(10,10)



ax.scatter(customer_history_df.loc[:,['recency']][customer_history_df.Cluster_Label==0], customer_history_df.loc[:,['frequency']][customer_history_df.Cluster_Label==0], customer_history_df.loc[:,['Amount']][customer_history_df.Cluster_Label==0], s=10,color='green',marker='^')



ax.set_xlabel('Recency')

ax.set_ylabel('Frequency')

ax.set_zlabel('Monetary')

plt.title("Cluster0")

plt.show()
#Cluster 1

figure=plt.figure()

ax=Axes3D(figure,elev=20,azim=160)



figure.set_size_inches(10,10)



ax.scatter(customer_history_df.loc[:,['recency']][customer_history_df.Cluster_Label==1], customer_history_df.loc[:,['frequency']][customer_history_df.Cluster_Label==1], customer_history_df.loc[:,['Amount']][customer_history_df.Cluster_Label==1], s=10,color='green',marker='^')



ax.set_xlabel('Recency')

ax.set_ylabel('Frequency')

ax.set_zlabel('Monetary')

plt.title("Cluster1")

plt.show()
#Cluster 2

figure=plt.figure()

ax=Axes3D(figure,elev=20,azim=160)



figure.set_size_inches(10,10)



ax.scatter(customer_history_df.loc[:,['recency']][customer_history_df.Cluster_Label==2], customer_history_df.loc[:,['frequency']][customer_history_df.Cluster_Label==2], customer_history_df.loc[:,['Amount']][customer_history_df.Cluster_Label==2], s=10,color='green',marker='^')







ax.set_xlabel('Recency')

ax.set_ylabel('Frequency')

ax.set_zlabel('Monetary')

plt.title("Cluster2")

plt.show()
#Cluster 3

figure=plt.figure()

ax=Axes3D(figure,elev=10,azim=160)



figure.set_size_inches(10,10)



ax.scatter(customer_history_df.loc[:,['recency']][customer_history_df.Cluster_Label==3], customer_history_df.loc[:,['frequency']][customer_history_df.Cluster_Label==3], customer_history_df.loc[:,['Amount']][customer_history_df.Cluster_Label==3], s=10,color='green',marker='^')



ax.set_xlabel('Recency')

ax.set_ylabel('Frequency')

ax.set_zlabel('Monetary')

plt.title("Cluster3")

plt.show()
#cluster 4

figure=plt.figure()

ax=Axes3D(figure,elev=10,azim=160)



figure.set_size_inches(10,10)



ax.scatter(customer_history_df.loc[:,['recency']][customer_history_df.Cluster_Label==4], customer_history_df.loc[:,['frequency']][customer_history_df.Cluster_Label==4], customer_history_df.loc[:,['Amount']][customer_history_df.Cluster_Label==4], s=10,color='green',marker='^')



ax.set_xlabel('Recency')

ax.set_ylabel('Frequency')

ax.set_zlabel('Monetary')

plt.title("Cluster4")

plt.show()
#cluster 5

figure=plt.figure()

ax=Axes3D(figure,elev=10,azim=160)



figure.set_size_inches(10,10)



ax.scatter(customer_history_df.loc[:,['recency']][customer_history_df.Cluster_Label==5], customer_history_df.loc[:,['frequency']][customer_history_df.Cluster_Label==5], customer_history_df.loc[:,['Amount']][customer_history_df.Cluster_Label==5], s=10,color='green',marker='^')



ax.set_xlabel('Recency')

ax.set_ylabel('Frequency')

ax.set_zlabel('Monetary')

plt.title("Cluster5")

plt.show()
#cluster six

figure=plt.figure()

ax=Axes3D(figure,elev=10,azim=160)



figure.set_size_inches(10,10)



ax.scatter(customer_history_df.loc[:,['recency']][customer_history_df.Cluster_Label==6], customer_history_df.loc[:,['frequency']][customer_history_df.Cluster_Label==6], customer_history_df.loc[:,['Amount']][customer_history_df.Cluster_Label==6], s=10,color='green',marker='^')



ax.set_xlabel('Recency')

ax.set_ylabel('Frequency')

ax.set_zlabel('Monetary')

plt.title("Cluster6")