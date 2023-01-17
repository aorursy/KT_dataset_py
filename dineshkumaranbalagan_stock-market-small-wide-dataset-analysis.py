#importing libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
print('Imported')
#importing datasets.
bar=pd.read_csv('../input/stock-market-small-wide-dataset/bar.csv')
quote=pd.read_csv('../input/stock-market-small-wide-dataset/quote.csv')
target=pd.read_csv('../input/stock-market-small-wide-dataset/target.csv')
event=pd.read_csv('../input/stock-market-small-wide-dataset/event.csv')
rating=pd.read_csv('../input/stock-market-small-wide-dataset/rating.csv')
news=pd.read_csv('../input/stock-market-small-wide-dataset/news.csv')
bar.head()
#importing files that needs to be analyzed:
quote.head()
#Filtering the table based on time column 2020-08-05:
Bar_30=bar[bar['time'].str.lower().str.contains('2020-08-05')]
Quote_30=quote[quote['time'].str.lower().str.contains('2020-08-05')]

#Merging Two dataframes into one based on time:
Dataframe1=pd.merge(Bar_30,Quote_30,on='time')

Dataframe1.info() 
#column names of this dataframe
print(Dataframe1.columns)
Dataframe1=Dataframe1.head(30)
Dataframe1.shape
#Visualizing bid_price, ask_price, average_price based on time:
plt.figure()
df=Dataframe1[['bid_price','ask_price','average_price']].plot(figsize=(8,8),marker='o',grid=True,markersize=5)#line plot
plt.xlabel('Time')
plt.ylabel('Price');
rating.head()
df1=rating[['symbol','ratingBuy','ratingScaleMark','consensusStartDate','consensusEndDate']]
df2=bar[['symbol','average_price']]
Table=pd.merge(df1,df2,on='symbol')
Table=Table[['symbol','ratingBuy','ratingScaleMark','average_price','consensusStartDate','consensusEndDate']]
Table.head()
Table.shape
plt.figure(figsize=(30,20))
plt.subplots_adjust(hspace = 0.5)
plt.subplot(2,1,1)
sns.lineplot(data=Table, x="consensusStartDate", y="average_price",marker='o',c='#4b0082',linestyle=':')
plt.xticks(rotation=90)
plt.xlabel('consensusStartDate',fontsize=17)
plt.ylabel('average_price',fontsize=20)
plt.subplot(2,1,2)
sns.lineplot(data=Table, x='consensusEndDate',y='average_price',marker='^',c='red',markersize=8,linestyle='dashed')
plt.xticks(rotation=90)
plt.xlabel('consensusEndDate',fontsize=17)
plt.ylabel('average_price',fontsize=20)
target.head()
#pulling price target average from Target data set 
target1=target[target['updatedDate'].str.lower().str.contains('2020-08-31')]
Target1=target1[['updatedDate','priceTargetAverage']]


#Renaming column label to perform merging process:
Target2=Target1.rename(columns={'updatedDate':'reportDate'})
Target2.head()
#Merging 'priceTargetAverage' column from Target dataframe with event dataframe:
Dataframe1=pd.merge(Target2,event, on='reportDate')
print(Dataframe1)
#Splitting datetime into date and time for further processing:
bar2=bar[bar['time'].str.lower().str.contains('2020-08-31')]
x=bar2['time'].str.split(" ", n = 1, expand = True)
bar2['updatedDate']= x[0] # Date
bar2['time2']=x[1] # Time
bar2.head()
#looking for a common column to merge the dataframe:
bar3=bar2[['updatedDate','average_price']]
print(bar3.head(),'\n',Target1.head())
#Merging dataframes by on updatedDate
Dataframe2=pd.merge(bar3,Target1,on='updatedDate')
Dataframe2.drop('priceTargetAverage',axis=1,inplace=True)
print(Dataframe2)
Dataframe2.plot(figsize=(18,5), grid=True)
plt.xlabel('Time', fontsize=15)
plt.ylabel('average_price', fontsize=15)
bar.columns
news.columns
news['datetime'].head(2)
bar2['time'].head(2)
bar5=bar2
bar5=bar5.rename(columns={'updatedDate':'date','time':'datetime','time2':'time'})
bar5.columns
# Selecting columns we are gonna merge from preprocessed bar dataframe: 
bar5=bar5[['datetime','average_price','time']]

#Merging dataframes for final Dataframe:
Dataframe3=pd.merge(news,bar5,on='datetime')
Dataframe3
Dataframe3.dtypes
plt.figure()
Dataframe3.plot(figsize=(15,5),c='purple')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Average Price',fontsize=12)
plt.title('Average Price Distribution', fontsize=15)