#importing libraries:

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline
#importing datasets.

bar=pd.read_csv('../input/stockmarket-dataset/bar-S.csv')

quote=pd.read_csv('../input/stockmarket-dataset/quote-S.csv')
#bar.reset_index(drop=True,inplace=True)

bar.head()
quote.head()
bar_N1=bar[['symbol','average_price']]

bar_N1.head()
bar_N1.describe()
from sklearn.preprocessing import LabelEncoder

LE=LabelEncoder()     #Label Encoding

label=LE.fit_transform(bar_N1['symbol'])
#Dropping the unuseful columns

bar_N1=bar_N1.drop("symbol", axis='columns')
#appending the transformed column 

bar_N1["symbol"]=label

bar_N1.head()
#Using Elbow Method to find the optimum number of clusters

from sklearn.cluster import KMeans

wcss=[]

K_rng=10



for i in range(1,K_rng):

    K=KMeans(i)

    K.fit(bar_N1)

    w=K.inertia_

    wcss.append(w)

    

Clusters=range(1,K_rng)

plt.figure(figsize=(12,8))

plt.plot(Clusters,wcss)

plt.xlabel('Clusters')

plt.ylabel('WCSS Values') #Within Cluster Sum of Squares

plt.title('Elbow Method Visualisation')
#Fitting the model

K2= KMeans(5)

K2.fit(bar_N1)
#Prediction using the model

N1_pred=bar_N1.copy()

N1_pred['Predicted']=K2.fit_predict(bar_N1)
#Visualise the clusters after prediction

plt.figure(figsize=(8,5))

plt.scatter(N1_pred['average_price'], N1_pred['symbol'], c=N1_pred['Predicted'], label='average_price', cmap = 'rainbow')

plt.xlabel('Average Price')

plt.ylabel('Symbol')

plt.title('Average Price vs Symbol(K=4)')
quote_df = quote.copy()

quote_df = quote_df [['time', 'ask_price','ticker']]

quote_df.head()
quote_df.rename(columns = {'time' : 'date', 'ask_price' : 'price_t'}, inplace = True)
#Return calculation manually



quote_df['price_t-1'] = quote_df['price_t'].shift(1)
#Return calculation using formula



quote_df['return'] = quote_df['price_t'].pct_change(1)
quote_df.head()
df_q=quote_df[['ticker','return']]
df_q.isnull().sum()
df_q=df_q.dropna()

df_q.head()
label2=LE.fit_transform(df_q['ticker'])

df_q=df_q.drop("ticker", axis='columns')

df_q["ticker"]=label2

df_q.head()
#Using Elbow Method to find the optimum number of clusters

from sklearn.cluster import KMeans

wcss=[]

K_rng=10



for i in range(1,K_rng):

    K=KMeans(i)

    K.fit(df_q)

    w=K.inertia_

    wcss.append(w)

    

Clusters=range(1,K_rng)

plt.figure(figsize=(12,8))

plt.plot(Clusters,wcss)

plt.xlabel('Clusters')

plt.ylabel('WCSS Values') #Within Cluster Sum of Squares

plt.title('Elbow Method Visualisation')
N=N1_pred['Predicted'].unique()
bar_N1['Predicted']=N1_pred['Predicted']

bar_N1['volume']=bar['volume']
bar_N1.head()
#listing out the stocks of different clusters

for i in N:

    print("For N1=",i)

    stock = bar_N1['volume'].loc[bar_N1['Predicted'] == i]

    print(stock)

    print("======================================================")
quote.head()
x=quote['time'].str.split(" ", n = 1, expand = True)

quote['Day']= x[0] # Date

quote['Time']=x[1] # Time

quote=quote.drop("time", axis='columns')

quote.head()
quote_N3=quote[['ticker','bid_size']]
label1=LE.fit_transform(quote['ticker'])
#Dropping the unuseful columns

quote_N3=quote_N3.drop("ticker", axis='columns')
#appending the transformed column 

quote_N3["ticker"]=label1

quote_N3.head()
#Using Elbow Method to find the optimum number of clusters

from sklearn.cluster import KMeans

wcss=[]

K_rng=10



for i in range(1,K_rng):

    K=KMeans(i)

    K.fit(quote_N3)

    w=K.inertia_

    wcss.append(w)

    

Clusters=range(1,K_rng)

plt.figure(figsize=(12,8))

plt.plot(Clusters,wcss)

plt.xlabel('Clusters')

plt.ylabel('WCSS Values') #Within Cluster Sum of Squares

plt.title('Elbow Method Visualisation')
#Fitting the model

K3= KMeans(4)

K3.fit(quote_N3)
#Prediction using the model

N3_pred=quote_N3.copy()

N3_pred['Predicted']=K3.fit_predict(quote_N3)
N3_pred['Day']=quote['Day']

N3_pred.head()
Days=N3_pred['Day'].unique()

n=N3_pred['Predicted'].unique()
#Distribution of bid_size in each cluster in a day

for i in n:

    for j in Days:

        print("In Day=",j)

        print("For N1=",i)

        final= N3_pred[(N3_pred['Predicted'] == i) & (N3_pred['Day'] == j)]

        print("Total bid size=",final['bid_size'].sum())

        print("================================")

    
#Visualise the clusters after prediction

plt.figure(figsize=(12,8))

plt.scatter(N3_pred['bid_size'], N3_pred['ticker'], c=N3_pred['Predicted'], cmap = 'rainbow')

plt.xlabel('bid_size')

plt.ylabel('ticker')

plt.title('bid_size vs ticker(K=4)')
data = quote.copy()

data=data [[ 'bid_price','bid_size']]

data.head()
data.rename(columns = {'bid_price' : 'price_T'}, inplace = True)
data['price_T-1'] = data['price_T'].shift(1)

data['price_change'] = (data['price_T']/data['price_T-1'])-1

data.head()
data.isnull().sum()
data=data.dropna()
#Showing the distribution of Bid Size and the Price change

plt.figure()

dataframe=data[['bid_size','price_change']].plot(figsize=(14,9),marker='o',grid=True,markersize=5)#line plot

plt.title("Distribution of Bid Size and Price Change in a day")

plt.xlabel('Time')

plt.ylabel('Distribution');
y=bar['time'].str.split(" ", n = 1, expand = True)

bar['Day']= x[0] # Date

bar['Time']=x[1] # Time

bar=bar.drop("time", axis='columns')

bar.head()
grouped_quote= quote.groupby("Day")
agg_bid_size=grouped_quote['bid_size'].agg(np.sum).sort_values(ascending=False).reset_index()
agg_bid_size.head()
grouped_bar=bar.groupby("Day")

agg_vol=grouped_bar['volume'].agg(np.sum).sort_values(ascending=False).reset_index()
agg_vol.head()
data_merge=pd.merge(agg_bid_size,agg_vol,on='Day')
data_merge.head()
plt.figure()

dataframe=data_merge.plot(x='Day',y=['bid_size','volume'],figsize=(14,9),marker='o',grid=True,markersize=5)

plt.title("Comparative Line plot between bid_size and volume per day ")

plt.xlabel('Day')

plt.ylabel('Comparative_Features');
Grouped_N1=bar_N1.groupby("Predicted")
avg_stock=Grouped_N1['volume'].agg(np.mean).sort_values(ascending=False).reset_index()
#Average stocks of all clusters of N1

avg_stock
new_quote=quote[['Day','bid_price','bid_size']]

new_quote.head()
new_quote['mean_price']=new_quote['bid_price'].agg(np.mean)

new_quote['std_price']=np.subtract(new_quote['bid_price'],new_quote['mean_price'])

new_quote['Volatility'] = np.square(new_quote['std_price'])

new_quote=new_quote.drop(['mean_price','std_price'], axis='columns')

new_quote.head()
#How bid_size and volatility is distributed on a day

rslt_df = new_quote[new_quote['Day'] == '2020-09-11'] 

rslt_df.head()
#bid_size and volatility on 2020-09-11

plt.figure()

dataframe=rslt_df[['bid_size','Volatility']].plot(figsize=(14,9),marker='o',grid=True,markersize=5)#line plot

plt.title("Distribution of Bid Size and Volatility in a day")

plt.xlabel('Time')

plt.ylabel('Distribution');
plt.figure(figsize=[20,10])

ax = sns.barplot(x="Day", y="bid_size", data=new_quote, palette="Blues")

plt.xticks(rotation=90, fontsize=16)

plt.yticks(fontsize=15)

plt.title("Distribution of Bid Size",fontsize=24)

plt.xlabel("Days",fontsize=20)

plt.ylabel("Bid Size",fontsize=20)

plt.tight_layout()
plt.figure(figsize=[20,10])

ax = sns.barplot(x="Day", y="Volatility", data=new_quote, palette="Greens")

plt.xticks(rotation=90, fontsize=16)

plt.yticks(fontsize=15)

plt.title("Distribution of Volatility",fontsize=24)

plt.xlabel("Days",fontsize=20)

plt.ylabel("Volatility",fontsize=20)

plt.tight_layout()
import math

from statsmodels.tsa.stattools import acf, pacf

import statsmodels.tsa.stattools as ts

from statsmodels.tsa.arima_model import ARIMA
new_N1=bar[['Day','average_price']]

new_N1.head()