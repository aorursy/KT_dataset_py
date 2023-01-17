import matplotlib as plt
import sklearn as ski
import pandas as pd
import numpy as np
import seaborn as sea
dfquote_s = pd.read_csv("../input/stock-market-small-wide-dataset/quote-S.csv")
dfbar_s = pd.read_csv("../input/stock-market-small-wide-dataset/bar-S.csv")
dfbar = pd.read_csv("../input/stock-market-small-wide-dataset/bar.csv")
dfevent = pd.read_csv("../input/stock-market-small-wide-dataset/event.csv")
dfnews = pd.read_csv("../input/stock-market-small-wide-dataset/news.csv")
dfquote = pd.read_csv("../input/stock-market-small-wide-dataset/quote.csv")
dfrating = pd.read_csv("../input/stock-market-small-wide-dataset/rating.csv")
dftarget = pd.read_csv("../input/stock-market-small-wide-dataset/target.csv")

dfquote.head()
dfquote.tail()
dfquote.shape
dfquote.describe(include='all')
dfquote.columns.values
dfquote.info()
dfquote.dtypes
dfquote.describe()
dfquote['bid_price'].unique()
dfquote['bid_size'].unique()
dfquote['ask_price'].unique()
dfquote['ask_size'].unique()
dfquote.isnull().sum()
dfquote.isnull().sum()>0
#total = dfquote.isnull().sum().sort_values(ascending=False)
#percent = (dfquote.isnull().sum()/dfquote.isnull().count()).sort_values(ascending=False)
#missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#missing_data
#dfquote['time'] = dfquote['time'].replace(np.nan, 0)
#dfquote['ticker'] = dfquote['ticker'].replace(np.nan, 0)
#dfquote['bid_price'] = dfquote['bid_price'].replace(np.nan, 0)
#dfquote['bid_size'] = dfquote['bid_size'].replace(np.nan, 0)
#dfquote['ask_price'] = dfquote['ask_price'].replace(np.nan, 0)
dfquote['time'] = pd.to_datetime(dfquote['time'])
dfquote.head()
dfquote['time'] = dfquote['time'].dt.hour
dfquote.head()
sea.distplot(dfquote['bid_price'])
sea.distplot(dfquote['bid_size'])
sea.distplot(dfquote['bid_price'])
sea.distplot(dfquote['ask_price'])
sea.distplot(dfquote['ask_size'])
sea.distplot(dfquote['time'])
f, ax = plt.pyplot.subplots(figsize=(8, 6))
fig = sea.boxplot(x='time', y="ask_size", data=dfquote)
fig.axis(ymin=0, ymax=800000)
f, ax = plt.pyplot.subplots(figsize=(8, 6))
fig = sea.boxplot(x='time', y="bid_price", data=dfquote)
fig.axis(ymin=0, ymax=800000)
f, ax = plt.pyplot.subplots(figsize=(8, 6))
fig = sea.boxplot(x='time', y="bid_size", data=dfquote)
fig.axis(ymin=0, ymax=800000)
f, ax = plt.pyplot.subplots(figsize=(8, 6))
fig = sea.boxplot(x='time', y="ask_size", data=dfquote)
fig.axis(ymin=0, ymax=800000)
sea.pairplot(dfquote)
sea.set()
cols = ['bid_price', 'bid_size', 'ask_price', 'ask_size']
sea.pairplot(dfquote[cols], size = 2.5)
plt.pyplot.show()
hourly_bid_price= dfquote.groupby("time")["bid_price"].sum().sort_values(ascending=False).to_frame()
hourly_bid_price.head()
plt.pyplot.figure(figsize=(20,10))
x = dfquote['bid_price']
y = dfquote['time']
plt.pyplot.scatter(x,y)
hourly_bid_size= dfquote.groupby("time")["bid_size"].sum().sort_values(ascending=False).to_frame()
hourly_bid_size.head()
plt.pyplot.figure(figsize=(20,10))
x = dfquote['bid_size']
y = dfquote['time']
plt.pyplot.scatter(x,y)
hourly_ask_price= dfquote.groupby("time")["ask_price"].sum().sort_values(ascending=False).to_frame()
hourly_ask_price.head()
plt.pyplot.figure(figsize=(20,10))
x = dfquote['ask_price']
y = dfquote['time']
plt.pyplot.scatter(x,y)
pearson_coefficeint = dfquote.corr(method='pearson')
pearson_coefficeint
sea.heatmap(pearson_coefficeint, cmap='RdBu_r',annot=True)
sea.regplot(x= "time",y="bid_price", data = dfquote)
plt.pyplot.ylim(0)
sea.regplot(x= "time",y="bid_size", data = dfquote)
plt.pyplot.ylim(0)
sea.regplot(x= "time",y="ask_price", data = dfquote)
plt.pyplot.ylim(0)
sea.regplot(x= "time",y="ask_price", data = dfquote)
plt.pyplot.ylim(0)
sea.regplot(x= "bid_price",y="ask_size", data = dfquote)
plt.pyplot.ylim(0)
sea.regplot(x= "bid_price",y="ask_price", data = dfquote)
plt.pyplot.ylim(0)
sea.regplot(x= "bid_price",y="ask_size", data = dfquote)
plt.pyplot.ylim(0)
sea.regplot(x= "bid_price",y="ask_size", data = dfquote)
plt.pyplot.ylim(0)
sea.regplot(x= "bid_price",y="ask_size", data = dfquote)
plt.pyplot.ylim(0)
sea.regplot(x= "bid_price",y="ask_size", data = dfquote)
plt.pyplot.ylim(0)
plt.pyplot.figure(figsize=(20,10))
x = dfquote['bid_price']
y = dfquote['ask_price']
plt.pyplot.scatter(x,y)
plt.pyplot.title("A Plot Between bid_price and ask_price")
plt.pyplot.xlabel("bid_price")
plt.pyplot.ylabel("ask_price")
plt.pyplot.figure(figsize=(20,10))
x = dfquote['bid_price']
y = dfquote['ask_size']
plt.pyplot.scatter(x,y)
plt.pyplot.title("A Plot Between bid_price and ask_size")
plt.pyplot.xlabel("bid_price")
plt.pyplot.ylabel("ask_size")
plt.pyplot.figure(figsize=(20,10))
x = dfquote['bid_price']
y = dfquote['ask_price']
plt.pyplot.scatter(x,y)
plt.pyplot.title("A Plot Between bid_price and ask_price")
plt.pyplot.xlabel("bid_price")
plt.pyplot.ylabel("ask_price")
plt.pyplot.figure(figsize=(20,10))
x = dfquote['bid_price']
y = dfquote['time']
plt.pyplot.scatter(x,y)
plt.pyplot.title("A Plot Between bid_price and time")
plt.pyplot.xlabel("bid_price")
plt.pyplot.ylabel("time")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x = dfquote[['bid_price','ask_size']]
y = dfquote['bid_size']

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
x_train.head(20)
y_train.head(20)
x_test.head(20)
y_test.head(20)
LR=LinearRegression()
LR.fit(x_train, y_train)
LR.predict(x_test)
LR.score(x_test, y_test)
dfbar.head()

dfevent.head()

dfnews.head()

dfquote_s.head()

dfquote.head()

dfrating.head()

dftarget.head()