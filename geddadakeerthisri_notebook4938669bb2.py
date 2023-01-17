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
import pandas as pd
df = pd.read_csv("/kaggle/input/stock-market-small-wide-dataset/bar.csv")
df1=df.head(30)
df1
df1.dtypes
df1.describe()
df1.describe(include=['object'])
%matplotlib inline 

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.style.use('ggplot') # optional: for ggplot-like style

# check for latest version of Matplotlib
print ('Matplotlib version: ', mpl.__version__) # >= 2.0.0
df1[['time','average_price']]
df1[['time','average_price']].plot(kind='line', figsize=(8, 5))
plt.title('average_price for date and time') # add a title to the histogram
plt.ylabel('date and time') # add y-label
plt.xlabel('average price') # add x-label

plt.show()

df1.plot(kind='scatter', x='time', y='average_price', figsize=(20,10), color='darkblue')
plt.title('average_price for date and time')
df1[['time','average_price']].plot(kind='hist', figsize=(8, 5))
plt.title('average_price for date and time') # add a title to the histogram
plt.ylabel('date and time') # add y-label
plt.xlabel('average price') # add x-label

plt.show()
df1[['time','average_price']].plot(kind='box', figsize=(8, 5))
plt.title('average_price for date and time') # add a title to the histogram
plt.ylabel('date and time') # add y-label
plt.xlabel('average price') # add x-label

plt.show()




df1[['time','average_price']].plot(kind='bar', figsize=(8, 5))
plt.title('average_price for date and time') # add a title to the histogram
plt.ylabel('date and time') # add y-label
plt.xlabel('average price') # add x-label

plt.show()
df1.corr()
from scipy import stats
pearson_coef, p_value = stats.pearsonr(df1['open_price'], df1['average_price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
pearson_coef, p_value = stats.pearsonr(df1['VWAP'], df1['average_price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  
df1[['symbol','average_price']]
grouped_test=df1[['symbol', 'average_price']].groupby(['symbol'])
grouped_test.head(2)
df4=pd.read_csv("/kaggle/input/stock-market-small-wide-dataset/quote.csv")

df5=df4.head(30)
df5
df5.dtypes
df5.describe()
df5[['time','bid_price','ask_price']]
df5.describe(include=['object'])
df5[['time','bid_price']].plot(kind='line', figsize=(8, 5))
plt.title('bid_price for date and time') # add a title to the histogram
plt.ylabel('date and time') # add y-label
plt.xlabel('bid_price') # add x-label

plt.show()
df5.plot(kind='scatter', x='time', y='bid_price', figsize=(20,10), color='darkblue')
plt.title('bid_price for date and time')
df5[['time','bid_price']].plot(kind='hist', figsize=(8, 5))
plt.title('bid_price for date and time') # add a title to the histogram
plt.ylabel('date and time') # add y-label
plt.xlabel('bid_price') # add x-label

plt.show()
df5[['time','bid_price']].plot(kind='box', figsize=(8, 5))
plt.title('bid_price for date and time') # add a title to the histogram
plt.ylabel('date and time') # add y-label
plt.xlabel('bid_price') # add x-label

plt.show()
df5[['time','bid_price']].plot(kind='bar', figsize=(8, 5))
plt.title('bid_price for date and time') # add a title to the histogram
plt.ylabel('date and time') # add y-label
plt.xlabel('bid_price') # add x-label

plt.show()
df5[['time','ask_price']].plot(kind='line', figsize=(8, 5))
plt.title('ask_price for date and time') # add a title to the histogram
plt.ylabel('date and time') # add y-label
plt.xlabel('ask_price') # add x-label

plt.show()
df5.plot(kind='scatter', x='time', y='ask_price', figsize=(20,10), color='darkblue')
plt.title('ask_price for date and time')
df5[['time','ask_price']].plot(kind='hist', figsize=(8, 5))
plt.title('ask_price for date and time') # add a title to the histogram
plt.ylabel('date and time') # add y-label
plt.xlabel('ask_price') # add x-label

plt.show()
df5[['time','ask_price']].plot(kind='box', figsize=(8, 5))
plt.title('ask_price for date and time') # add a title to the histogram
plt.ylabel('date and time') # add y-label
plt.xlabel('ask_price') # add x-label

plt.show()
df5[['time','ask_price']].plot(kind='bar', figsize=(8, 5))
plt.title('ask_price for date and time') # add a title to the histogram
plt.ylabel('date and time') # add y-label
plt.xlabel('ask_price') # add x-label

plt.show()
df5.corr()
from scipy import stats
pearson_coef, p_value = stats.pearsonr(df5['bid_price'], df5['ask_price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  
pearson_coef, p_value = stats.pearsonr(df5['ask_size'], df5['ask_price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  
pearson_coef, p_value = stats.pearsonr(df5['bid_size'], df5['bid_price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  
grouped_test2=df5[['ticker', 'bid_price','ask_price']].groupby(['ticker'])
grouped_test2.head(2)
import pandas as pd
df6 = pd.read_csv("/kaggle/input/stock-market-small-wide-dataset/rating.csv")
df6
df7=df6.head(163)
df7
import pandas as pd
df8 = pd.read_csv("/kaggle/input/stock-market-small-wide-dataset/bar.csv")
df8
df10=pd.merge(df7, df8, how='inner', on='symbol', left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True,
         suffixes=('_x', '_y'), copy=True, indicator=False,
         validate=None)
df10
df10[['symbol','ratingBuy','ratingScaleMark','consensusStartDate','consensusEndDate','average_price']]
df11=df10[['consensusStartDate','consensusEndDate','average_price']]
df11
df10[['consensusStartDate','average_price']].plot(kind='line', figsize=(8, 5))
plt.title('average_price for consensusStartDate') # add a title to the histogram
plt.ylabel('consensusStartDate') # add y-label
plt.xlabel('average price') # add x-label

plt.show()
df11.plot(kind='scatter', x='consensusStartDate', y='average_price', figsize=(20,10), color='darkblue')
plt.title('average_price for consensusStartDate')
df10[['consensusStartDate','average_price']].plot(kind='hist', figsize=(8, 5))
plt.title('average_price for consensusStartDate') # add a title to the histogram
plt.ylabel('consensusStartDate') # add y-label
plt.xlabel('average_price') # add x-label

plt.show()
df10[['consensusStartDate','average_price']].plot(kind='box', figsize=(8, 5))
plt.title('average_price for consensusStartDate') # add a title to the histogram
plt.ylabel('consensusStartDate') # add y-label
plt.xlabel('average_price') # add x-label

plt.show()
df10[['consensusEndDate','average_price']].plot(kind='line', figsize=(8, 5))
plt.title('average_price for consensusEndDate') # add a title to the histogram
plt.ylabel('consensusEndDate') # add y-label
plt.xlabel('average price') # add x-label

plt.show()
df11.plot(kind='scatter', x='consensusEndDate', y='average_price', figsize=(20,10), color='darkblue')
plt.title('average_price for consensusEndDate')
df10[['consensusEndDate','average_price']].plot(kind='hist', figsize=(8, 5))
plt.title('average_price for consensusEndDate') # add a title to the histogram
plt.ylabel('consensusEndDate') # add y-label
plt.xlabel('average_price') # add x-label

plt.show()
df10[['consensusEndDate','average_price']].plot(kind='box', figsize=(8, 5))
plt.title('average_price for consensusEndDate') # add a title to the histogram
plt.ylabel('consensusEndDate') # add y-label
plt.xlabel('average_price') # add x-label

plt.show()
x=df[df['time']=='2020-08-31 19:46:00+00:00']
x.index[0]
df22=df[0:x.index[0]]
df22
df23=df[x.index[-1]:1074858]
df23
df22[['time','average_price']].plot(kind='box', figsize=(8, 5))
plt.title('average_price for time') # add a title to the histogram
plt.ylabel('time') # add y-label
plt.xlabel('average_price') # add x-label

plt.show()
df22[['time','average_price']].plot(kind='hist', figsize=(8, 5))
plt.title('average_price for time') # add a title to the histogram
plt.ylabel('time') # add y-label
plt.xlabel('average_price') # add x-label

plt.show()
df23[['time','average_price']].plot(kind='box', figsize=(8, 5))
plt.title('average_price for time') # add a title to the histogram
plt.ylabel('time') # add y-label
plt.xlabel('average_price') # add x-label

plt.show()
df23[['time','average_price']].plot(kind='hist', figsize=(8, 5))
plt.title('average_price for time') # add a title to the histogram
plt.ylabel('time') # add y-label
plt.xlabel('average_price') # add x-label

plt.show()
df14=pd.read_csv('/kaggle/input/stock-market-small-wide-dataset/target.csv')
df14
df25=df14[df14['updatedDate']=='2020-08-31']
df25
df25[['updatedDate','priceTargetAverage']]
df25
df24=df[x.index[0]:x.index[-1]]
df24
df26=pd.merge(df24, df25, how='inner', on='symbol', left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True,
         suffixes=('_x', '_y'), copy=True, indicator=False,
         validate=None)
df26
df26[['symbol','updatedDate','priceTargetAverage','priceTargetHigh','priceTargetLow','numberOfAnalysts','average_price']]
df28=df26[['time','average_price']]
df28
df27=df[0:x.index[0]]
df29=df27[['time','average_price']]
df29
frames = [df28, df29]

result = pd.concat(frames)
result.sort_values(by=['average_price'],ascending=False)
df.head(510098)
f=df[df['time']=='2020-08-12 13:45:00+00:00']
df30=df[f.index[0]:df.index[-1]]
df30

df31=pd.read_csv('/kaggle/input/stock-market-small-wide-dataset/news.csv')
df31
df32=df31.rename(columns={"datetime": "time"})
df32
df33=pd.merge(df, df32, how='inner', on='time', left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True,
         suffixes=('_x', '_y'), copy=True, indicator=False,
         validate=None)
df33
df33[['time','stock','summary','average_price']]
