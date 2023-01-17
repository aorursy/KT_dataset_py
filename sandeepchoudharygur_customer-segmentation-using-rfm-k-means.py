#Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime as dt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

%matplotlib inline
#Importing Online retail data containing transactions from 1/12/10 to 9/12/11

retail_data = pd.read_csv('../input/online-retail-data/OnlineRetail.csv', encoding = 'unicode_escape')

retail_data.head()
#Customer distribution by country

country_data = retail_data[['Country', 'CustomerID']].drop_duplicates()

country_data.groupby(['Country'])['CustomerID'].aggregate('count').reset_index().sort_values('CustomerID',ascending=False)
#keep only united kingdom data

uk_data = retail_data.query("Country == 'United Kingdom'").reset_index(drop=True)
uk_data.isnull().sum()
#remove missing values 

uk_data = uk_data[pd.notnull(uk_data['CustomerID'])]
#removing negative value from quantity

uk_data = uk_data[(uk_data['Quantity']>0)]
#convert the string date field to datetime

uk_data['InvoiceDate'] = pd.to_datetime(uk_data['InvoiceDate'])
#add new column total amt

uk_data['TotalAmount'] = uk_data['Quantity']*uk_data['UnitPrice']
uk_data.head()
uk_data.shape
#set latest data as 10-12-2011

LatestDate = dt.datetime(2011,12,10)



#create RFM Modelling scores for each customer

RFMScores = uk_data.groupby('CustomerID').agg({'InvoiceDate': lambda x: (LatestDate - x.max()).days, 'InvoiceNo': lambda x: len(x), 'TotalAmount': lambda x: x.sum()})



#convert invoice date into int for calc purpose

RFMScores['InvoiceDate'] = RFMScores['InvoiceDate'].astype(int)



#Rename column names to Recency, Frequency and Monetary

RFMScores.rename(columns={'InvoiceDate': 'Recency',

                         'InvoiceNo': 'Frequency',

                         'TotalAmount': 'Monetary'}, inplace = True)



RFMScores.head()
RFMScores.Recency.describe()
x = RFMScores['Recency']

ax = sns.distplot(x)
RFMScores.Frequency.describe()
x = RFMScores.query('Frequency < 1000')['Frequency']

ax = sns.distplot(x)
RFMScores.Monetary.describe()
x = RFMScores.query('Monetary < 1000')['Monetary']

ax = sns.distplot(x)
#split into four segments using quantiles

quantiles = RFMScores.quantile(q=[0.25,0.50,0.75])

quantiles = quantiles.to_dict()
quantiles

#Functions to crete R, F and M segments

def RScore(x,p,d):

    if x<= d[p][0.25]:

        return 1

    elif x <= d[p][0.50]:

        return 2

    elif x<= d[p][0.75]:

        return 3

    else:

        return 4

    

def FMScore(x,p,d):

    if x<= d[p][0.25]:

        return 4

    elif x <= d[p][0.50]:

        return 3

    elif x<= d[p][0.75]:

        return 2

    else:

        return 1
#calculate R, F and M values

RFMScores['R'] = RFMScores['Recency'].apply(RScore, args=('Recency',quantiles))

RFMScores['F'] = RFMScores['Frequency'].apply(FMScore, args=('Frequency',quantiles))

RFMScores['M'] = RFMScores['Monetary'].apply(FMScore, args=('Monetary',quantiles))



RFMScores.head()
#concatenate score of RFM

RFMScores['RFMGroup'] = RFMScores.R.map(str) + RFMScores.F.map(str) + RFMScores.M.map(str)



#Add the RFM scores

RFMScores['RFMScore'] = RFMScores[['R', 'F', 'M']].sum(axis=1)

RFMScores.head()
#Assign Loyalty level to each customer

Loyalty_Level = ['Platinum', 'Gold', 'Silver', 'Bronze']

Scorecut = pd.qcut(RFMScores.RFMScore, q=4, labels = Loyalty_Level)

RFMScores['RFM_Loyalty_Level'] = Scorecut.values

RFMScores.head()
#validate the data for RFMGroup = '111'

RFMScores[RFMScores['RFMGroup'] == '111'].sort_values('Monetary', ascending = False).head(10)
#deal with negative numbers

def neg_zero(num):

    if num <= 0:

        return 1

    else:

        return num



RFMScores['Recency'] = [neg_zero(x) for x in RFMScores.Recency]

RFMScores['Monetary'] = [neg_zero(x) for x in RFMScores.Monetary]



#normalization using log

log_data = RFMScores[['Recency', 'Frequency', 'Monetary']].apply(np.log,axis=1).round(3)
#After normalization for Recency

sns.distplot(log_data['Recency'])
#After normalization for Frequency

sns.distplot(log_data['Frequency'])
#After normalization for Monetary

sns.distplot(log_data['Monetary'])
#Scaling

sobj = StandardScaler()

Scaled_Data = sobj.fit_transform(log_data)



#convert it to dataframe

Scaled_Data = pd.DataFrame(Scaled_Data, index = RFMScores.index, columns = log_data.columns)
sum_sq_dist = {}

for k in range(1,15):

    km = KMeans(n_clusters=k, init= 'k-means++', max_iter=1000)

    km = km.fit(Scaled_Data)

    sum_sq_dist[k] = km.inertia_



#plot the graph

sns.pointplot(x= list(sum_sq_dist.keys()),y=list(sum_sq_dist.values()))

plt.xlabel('Number of Clusters(k)')

plt.ylabel('Sum of square distances')

plt.title('Elbow method using Inertia')

plt.show()
#K-Means model

KMean_clust = KMeans(n_clusters=3, init='k-means++', max_iter=1000)

KMean_clust.fit(Scaled_Data)



#finding cluster for the label observation

RFMScores['Cluster'] = KMean_clust.labels_

RFMScores.head(10)
plt.figure(figsize=(7,7))

#Scatter Plot Frequency Vs Recency

Colors = ["red", "green", "blue"]

RFMScores['Color'] = RFMScores['Cluster'].map(lambda p: Colors[p])

ax = RFMScores.plot(    

    kind="scatter", 

    x="Recency", y="Frequency",

    figsize=(10,8),

    c = RFMScores['Color']

)
RFMScores.head(10)