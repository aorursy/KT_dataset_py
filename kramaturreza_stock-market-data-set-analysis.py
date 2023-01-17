#importing libraries:

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns

print('Dataset_Imported')
#importing datasets.

bar=pd.read_csv('../input/stockmarket-dataset/bar-S.csv')

quote=pd.read_csv('../input/stockmarket-dataset/quote-S.csv')

target=pd.read_csv('../input/stockmarket-dataset/target.csv')

event=pd.read_csv('../input/stockmarket-dataset/event.csv')

rating=pd.read_csv('../input/stockmarket-dataset/rating.csv')

news=pd.read_csv('../input/stockmarket-dataset/news.csv')

bar.head(5)
#importing files that needs to be analyzed:

quote.head()
#Filtering the table based on time column 2020-08-05:

Bar_30=bar[bar['time'].str.lower().str.contains('2020-08-05')]

Quote_30=quote[quote['time'].str.lower().str.contains('2020-08-05')]



#Merging Two dataframes into one based on time:

Merged_df=pd.merge(Bar_30,Quote_30,on='time')



Merged_df.info() 
#column names of this dataframe

print(Merged_df.columns)
Merged_df=Merged_df.head(30)

Merged_df.shape
#Visualizing bid_price, ask_price, average_price based on time:

plt.figure()

dataframe=Merged_df[['bid_price','ask_price','average_price']].plot(figsize=(14,9),marker='o',grid=True,markersize=5)#line plot

plt.xlabel('Time')

plt.ylabel('Price');
rating.head()
df_rating=rating[['symbol','ratingBuy','ratingScaleMark','consensusStartDate','consensusEndDate']]

df_bar=bar[['symbol','average_price']]

Table=pd.merge(df_rating,df_bar,on='symbol')

Table=Table[['symbol','ratingBuy','ratingScaleMark','average_price','consensusStartDate','consensusEndDate']]

Table.head()
Table.shape
plt.figure(figsize=(30,20))

plt.subplots_adjust(hspace = 0.5)

plt.subplot(2,1,1)

sns.lineplot(data=Table, x="consensusStartDate", y="average_price",marker='o',c='#4b0082',linestyle=':')

plt.xticks(rotation=90)

plt.xlabel('consensusStartDate',fontsize=19)

plt.ylabel('average_price',fontsize=23)

plt.subplot(2,1,2)

sns.lineplot(data=Table, x='consensusEndDate',y='average_price',marker='^',c='green',markersize=8,linestyle='dashed')

plt.xticks(rotation=90)

plt.xlabel('consensusEndDate',fontsize=19)

plt.ylabel('average_price',fontsize=23)
target.head()
#pulling price target average from Target data set 

target_df=target[target['updatedDate'].str.lower().str.contains('2020-08-31')]

Target_df=target_df[['updatedDate','priceTargetAverage']]





#Renaming column label to perform merging process:

Target_df_final=Target_df.rename(columns={'updatedDate':'reportDate'})

Target_df_final.head()
#Merging 'priceTargetAverage' column from Target dataframe with event dataframe:

Merged_Dataframe=pd.merge(Target_df_final,event, on='reportDate')

print(Merged_Dataframe)
#Splitting datetime into date and time for further processing:

bar_split=bar[bar['time'].str.lower().str.contains('2020-08-31')]

x=bar_split['time'].str.split(" ", n = 1, expand = True)

bar_split['updatedDate']= x[0] # Date

bar_split['time2']=x[1] # Time

bar_split.head()
#looking for a common column to merge the dataframe:

bar_common=bar_split[['updatedDate','average_price']]

print(bar_common.head(),'\n',Target_df.head())
#Merging dataframes by on updatedDate

UpdatedDate_df_merged=pd.merge(bar_common,Target_df,on='updatedDate')

UpdatedDate_df_merged.drop('priceTargetAverage',axis=1,inplace=True)

print(UpdatedDate_df_merged)
UpdatedDate_df_merged.plot(figsize=(19,9), grid=True)

plt.xlabel('Time', fontsize=19)

plt.ylabel('average_price', fontsize=19)
bar.columns
news.columns
news['datetime'].head(2)
bar_split['time'].head(2)
bar_final=bar_split

bar_final=bar_final.rename(columns={'updatedDate':'date','time':'datetime','time2':'time'})

bar_final.columns
# Selecting columns we are gonna merge from preprocessed bar dataframe: 

bar_final=bar_final[['datetime','average_price','time']]



#Merging dataframes for final Dataframe:

Dataframe_final=pd.merge(news,bar_final,on='datetime')

Dataframe_final
Dataframe_final.dtypes
plt.figure()

Dataframe_final.plot(figsize=(19,5),c='red')

plt.xlabel('Time', fontsize=14)

plt.ylabel('Average Price',fontsize=14)

plt.title('Average Price Distribution', fontsize=19)