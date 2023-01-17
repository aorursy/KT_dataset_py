import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print('Setup Complete')
stock_file_path = '../input/f.us.txt'

stock_data_ford = pd.read_csv(stock_file_path)

stock_data_ford.describe()
stock_file_path = '../input/ibm.us.txt'

stock_data_ibm = pd.read_csv(stock_file_path)

stock_data_ibm.describe()
stock_file_path = '../input/aapl.us.txt'

stock_data_aapl = pd.read_csv(stock_file_path, index_col='Date', parse_dates=True)

stock_data_aapl.describe()
stock_data_aapl.head()

stock_data_aapl.tail(9)

#type(stock_data_aapl)
plt.figure(figsize=(24,12))

plt.title('apple stock price last 50 days')





sns.lineplot(data=stock_data_aapl.tail(50))

sns.set_style("darkgrid")
list(stock_data_aapl.columns)
plt.figure(figsize=(14, 6))

plt.title('apple stock price last 100 days')



#plt.show()



sns.lineplot(data=stock_data_aapl['High'].tail(100), label="High")

sns.lineplot(data=stock_data_aapl["Low"].tail(100), label='Low')



plt.xlabel("Date")

plt.ylabel("Value($)")

sns.set_style("white")
short_data_aapl = pd.read_csv(stock_file_path, index_col='Date')

short_data_aapl = short_data_aapl.tail(50)
plt.figure(figsize=(16,16))

#plt.title("Words")

sns.barplot(x=short_data_aapl['High'], y=short_data_aapl.index)
short_data_aapl.describe
#short_data_aapl.columns()
#['Open', 'High', 'Low', 'Close', 'Volume']

hot_data_aapl = short_data_aapl

hot_data_aapl = hot_data_aapl.drop(['Volume', 'OpenInt'], axis=1)
plt.figure(figsize=(24,12))



sns.heatmap(data=hot_data_aapl, annot=True)
#plt.figure(figsize=(16,16))

sns.scatterplot(x=short_data_aapl.index,y=short_data_aapl["Close"], hue=short_data_aapl["Volume"])

sns.lmplot(x="Open",y="Close", hue="High", data=short_data_aapl)

#sns.regplot(x=stock_data_aapl.index,y=stock_data_aapl["Close"])

#can't plot regression line for date as there is no mean for that value, will revisit later and correct

sns.set_style("ticks")
plt.figure(figsize=(16,16))

sns.swarmplot(x=short_data_aapl["Open"], y=short_data_aapl["Close"])

#not a good graph for this sort of data, meant to show different groupings

sns.set_style("ticks")
plt.figure(figsize=(10,10))

sns.distplot(a=short_data_aapl["Close"], kde=False)
plt.figure(figsize=(10,10))

sns.kdeplot(data=short_data_aapl["High"], shade=True)
plt.figure(figsize=(16,16))

sns.jointplot(x=short_data_aapl["Close"], y=short_data_aapl["Volume"], kind="kde")

#The volume and close features should be scaled for better comparison

#meaning we need to make the min value represented 0 and the max 1 for both features to make a more visible corrilation
temp1 = stock_data_aapl.tail(1000)

temp2 = stock_data_ford.tail(1000)
sns.jointplot(temp1["High"], temp2["Low"], kind='resid' )

sns.jointplot(temp1["High"], temp2["Low"], kind='hex' )

sns.jointplot(temp1["High"], temp2["Low"], kind='reg' )

sns.jointplot(temp1["High"], temp2["Low"], kind='kde' )
plt.figure(figsize=(16,16))



sns.distplot(a=stock_data_aapl["Close"], label="Apple", kde=False)

sns.distplot(a=stock_data_ford["Close"], label="Ford", kde=False)

sns.distplot(a=stock_data_ibm["Close"], label="IBM", kde=False)



plt.legend()