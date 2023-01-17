# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

auction=pd.read_csv("../input/auction.csv")

auction.head()
#How hot are they

print("price item sold for",auction.groupby('item').price.mean())
auction.isnull().sum()
import seaborn as sns

sns.boxplot(x=auction.item,y=auction.price)

(auction.auction_type.value_counts()*100/auction.shape[0]).plot(kind='bar',title='Perc of auctions by auction type')
auction.item.value_counts().plot(kind='bar',title="Items being sold")
pd.crosstab(auction.auction_type,auction.item).apply(lambda x:100*x/sum(x),axis=1).plot(kind='bar').set_title("Number of items sold by auction")
auction.pivot_table(index='auction_type',columns='item',values='price',aggfunc=np.median ).plot(kind='bar').set_title("Median price of items auctioned by type")

auction.pivot_table(index='auction_type',columns='item',values='openbid',aggfunc=np.median).plot(kind='bar').set_title("Median Open bid of items by type")

auction.pivot_table(index='auction_type',columns='item',values='bidtime',aggfunc=np.median).plot(kind='bar').set_title("Median Bid time of items by type")

sns.distplot(auction.loc[auction.item=='Cartier wristwatch'].price)

#Trend is lagging 
sns.distplot(auction.loc[auction.item=='Palm Pilot M515 PDA'].price)

sns.distplot(auction.loc[auction.item=='Xbox game console'].price)

print("Mean",auction.groupby('item').price.mean())

print("Median",auction.groupby('item').price.median())

#How long does it take to sell an item for each auction_type

print("Average time to sell",auction.bidtime.mean())

#ie close to 4th day bids start to come in
auction.groupby('auction_type').bidtime.mean().plot(kind='bar',title='How long into an auction does an item sell').axhline(y=auction.bidtime.median(),xmin=0,xmax=3,c="red",linewidth=3,zorder=0)



auction.groupby('item').bidtime.mean().plot(kind='bar',title='How long does an item take to sell').axhline(y=auction.bidtime.median(),xmin=0,xmax=3,c="red",linewidth=3,zorder=0)



one=auction.loc[auction.auctionid==1638893549]

#Thats odd the engine accepts lower bids???? after a higher bid has been put forth

one
binned=[2,5,10,15,20,30,40,50,60,70,90,100,200,300,400,500,600,700,800,1000,2000,3000,4000,5000]

auction['binned_openbid']=np.digitize(auction.openbid,binned)

auction.groupby('binned_openbid').price.count().plot(kind='bar').set_title("Do lower bids result in more people participating")
auction.groupby('binned_openbid').price.median().plot(kind='bar').set_title("Do lower bids result in better prices")
from ggplot import *

df=auction.groupby(['binned_openbid','item'],as_index=False).price.median()

ggplot(aes(x='binned_openbid',y='price',color='item'),data=df)+geom_line()+ggtitle("Do lower bids result in better prices")