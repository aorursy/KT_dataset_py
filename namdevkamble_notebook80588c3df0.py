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
#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from scipy import stats
df=pd.read_csv("/kaggle/input/stock-market-small-wide-dataset/bar-S.csv") #loading dataset
df.head(5)
df1=df.head(30) #Taking subset of total data
df1.head()
df1.info()
df1.describe()
df1[['symbol','time','average_price']]
df1[['time','average_price']].plot(kind='line',figsize=(12,5)) #line plot for average price
plt.title('average_price for date and time')# title for the plot
plt.xlabel('date and time')
plt.ylabel('average_price')
plt.show();
df1['average_price'].mean()
df1[['time','average_price']].plot(kind='bar',figsize=(12,5)) #bar plot for average price
plt.title('average_price for date and time')# title for the plot
plt.xlabel('date and time')
plt.ylabel('average_price')
plt.show();
df1[['time','average_price']].plot(kind='box',figsize=(12,5)) #box plot 
plt.title('average_price for date and time')# title for the plot
plt.xlabel('date and time')
plt.ylabel('average_price')
plt.show();
df1[['time','average_price','volume']].plot(kind='line',figsize=(12,5)) # plot of average_price,and volume 
df1.groupby('symbol')['average_price'].count().plot(kind='bar',figsize=(12,5))# bar plot,grouping symbol and count
#of average_price.

df1.groupby('symbol')['average_price'].sum().plot(kind='bar',figsize=(12,7))
dfq=pd.read_csv("/kaggle/input/stock-market-small-wide-dataset/quote.csv")
data=dfq.head(30)# taking subset of 30 datapoints 
data
data[['time','bid_price']].plot(kind='line',figsize=(10,5))#line plot of date time and bid price
plt.title('bid_price and datetime')
plt.xlabel('date and time')
plt.ylabel('bid_price')
plt.show();
data[['time','ask_price','bid_price']].plot(kind='line',figsize=(10,5))#line plot of date time and bid price
plt.title('bid_price and datetime')
plt.xlabel('date and time')
plt.ylabel('bid_price')
plt.show();
data[['time','ask_price','bid_price']].plot(kind='bar',figsize=(12,6))
plt.xlabel('date and time')
plt.ylabel('ask_price and bid_price')
plt.show();
sn.heatmap(data.corr(),annot=True)# heatmap of correlation matrix
data.groupby('ticker')['bid_price'].sum().plot(kind='bar',figsize=(6,4))
# looking for is there any correlation between ask_price and bid price
# H0(Null hypothesis): Nothing is going on,it means no correlation
#H11(Alternate):There is correlation
alpha=0.05 #significance level
pearson_coef, p_value = stats.pearsonr(data['bid_price'], data['ask_price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " \nwith a P-value of P =", p_value) 
if p_value<alpha:
    print("There is correlation")
else:
    print("No correlation")
alpha=0.05
pearson_coef, p_value = stats.pearsonr(data['ask_size'], data['ask_price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " \nwith a P-value of P =", p_value) 
if p_value<alpha:
    print("There is correlation")
else:
    print("No correlation")
group_ticker=data[['ticker', 'bid_price','ask_price']].groupby('ticker').head().plot()
df_rating=pd.read_csv("/kaggle/input/stock-market-small-wide-dataset/rating.csv")
df_rating.head()# rating dataset
df_rating.shape#shape of dataset
df_rating.describe()#statistical summary
df4=df_rating.drop(['ratingOverweight','ratingHold','ratingUnderweight','ratingSell','ratingNone'],axis=1)#getting required columns for creating table

df4.head()
df5=df4.head(150) #selecting first 150 records for our analysis
df5.head()#bar dataset overlook
df1.head()  #'bar' dataset
#Merging 'bar' and 'rating' dataset to create table
df6=pd.merge(df,df4,how='inner', on='symbol', left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True,
         suffixes=('_x', '_y'), copy=True, indicator=False,
         validate=None)
df6.shape
df6.head()
df7=df6[['symbol','average_price','ratingBuy','ratingScaleMark','consensusStartDate','consensusEndDate']]
df7.head()
df8=df7[['consensusStartDate','consensusEndDate','average_price']]
df8.head()
df8[['consensusStartDate','average_price']].plot(figsize=(12,6))
plt.legend('average_price variation with consensusStartDate')
plt.xlabel("consensusStartDate")
plt.ylabel("average_price")
plt.show();

df8[['consensusStartDate','average_price']].plot(kind='hist',figsize=(12,6))
plt.legend('average_price variation with consensusStartDate')
plt.xlabel("average_price")
plt.ylabel("consensusStartDate")
plt.show();
df8[['consensusEndDate','average_price']].plot(figsize=(12,6))
plt.legend('average_price variation with consensusEndDate')
plt.xlabel("consensusEndDate")
plt.ylabel("average_price")
plt.show();
df8[['consensusEndDate','average_price']].plot(kind='hist',figsize=(12,6))
plt.legend('average_price variation with consensusEndDate')
plt.xlabel("average_price")
plt.ylabel("consensusEndDate")
plt.show();
df9=df[df['time']=='2020-08-31 19:46:00+00:00']
df9.index[0]
df10=df[0:df9.index[0]]
df10.head()
df11=df[df10.index[-1]:1074858]
df11.head()
df10[['time','average_price']].plot(kind='box', figsize=(8, 5))
plt.title('average_price for time') # add a title to the histogram
plt.ylabel('time') # add y-label
plt.xlabel('average_price') # add x-label

plt.show()
df10[['time','average_price']].plot(kind='hist', figsize=(8, 5))
plt.title('average_price for time') # add a title to the histogram
plt.ylabel('time') # add y-label
plt.xlabel('average_price') # add x-label

plt.show()
df11[['time','average_price']].plot(kind='hist', figsize=(8, 5))
plt.title('average_price for time') # add a title to the histogram
plt.ylabel('time') # add y-label
plt.xlabel('average_price') # add x-label

plt.show()
df11[['time','average_price']].plot(kind='box', figsize=(8, 5))
plt.title('average_price for time') # add a title to the histogram
plt.ylabel('time') # add y-label
plt.xlabel('average_price') # add x-label

plt.show()
#reading target dataset
df12=pd.read_csv("/kaggle/input/stock-market-small-wide-dataset/target.csv")
df12.head()
df12.info()
df13=pd.read_csv("/kaggle/input/stock-market-small-wide-dataset/event.csv")  #event dataset
df13
df13.info()
df14=df12[df12['updatedDate']=='2020-08-31']
df14.head()
df14[['updatedDate','priceTargetAverage']].plot(kind='line',figsize=(14,6))
plt.legend("Changes of priceTargetAverage with updatedDate")
plt.xlabel('updatedDate')
plt.ylabel("priceTargetAverage")
plt.show();
df14[['updatedDate','priceTargetAverage']].plot(kind='bar',figsize=(14,6))
plt.legend("Changes of priceTargetAverage with updatedDate")
plt.xlabel('updatedDate')
plt.ylabel("priceTargetAverage")
plt.show();
df15=df[df9.index[0]:df9.index[-1]]
df15.head()
df16=pd.merge(df14, df15, how='inner', on='symbol', left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True,
         suffixes=('_x', '_y'), copy=True, indicator=False,
         validate=None)
df16.head()
df16[['symbol','updatedDate','priceTargetAverage','priceTargetHigh',
      'priceTargetLow','numberOfAnalysts','average_price']].head()
df17=df16[['time','average_price']]
df17.head()
df18=df[0:df9.index[0]]
df19=df18[['time','average_price']]
df19.head(30)
result = [df18, df17]

final_r = pd.concat(result)
final_r.sort_values(by=['average_price'],ascending=False).head(10)
df.head(510098)
f=df[df['time']=='2020-08-12 13:45:00+00:00']
df20=df[f.index[0]:df.index[-1]]
df20
df21=pd.read_csv("/kaggle/input/stock-market-small-wide-dataset/news.csv")  #reading 'news' dataset  
df21.head()
df21.info()
df22=df21.rename(columns={'datetime':'time'})
df22.head()
#Merge two dataframe on common column name 'time'
df24=pd.merge(df, df22, how='inner', on='time', left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True,
         suffixes=('_x', '_y'), copy=True, indicator=False,
         validate=None)
df24.head()
df25=df24[['time','stock','summary','average_price']]
df25.head() #Final Dataset containing time,stock,summary and average price as attributes.
df25['stock'].value_counts().plot()