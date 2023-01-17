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
from datetime import datetime

from datetime import timedelta

import matplotlib.pyplot as plt

data_df=pd.read_csv("/kaggle/input/gafa-stock-prices/YahooFinance - GAFA stock prices.csv")

data_df.head()
data_df['security_symbol'].unique()
data_df['security_symbol']=data_df['security_symbol'].replace(['%5EGSPC'],'YHOO')
data_df.head()
yhoo=data_df[data_df['security_symbol']=='YHOO']

yhoo=yhoo.set_index('date')

#yhoo_close=yhoo.close.values.astype('float32')

#yhoo_close.shape

ax=yhoo['close'].plot()

plt.xticks(rotation=90)
from dateutil import parser

past = datetime.now() - timedelta(days=365)

#new_date = parser.parse("2018-11-10 10:55:31+00:00")

new_date = datetime.now()



    

data_df['date'] = pd.to_datetime(data_df['date'])  

    

#greater than the start date and smaller than the end date

mask = (data_df['date'] < new_date) & (data_df['date'] > past)



# On Covid-19 Pandemic

end = datetime.now()

start = datetime(end.year - 1, end.month, end.day)

yhoo=data_df[data_df['security_symbol']=='YHOO']

yhoo=yhoo.loc[mask]

yhoo=yhoo.set_index('date')



yhoo_close=yhoo.close.values.astype('float32')

#yhoo_close.shape

#ax=yhoo['close'].plot('r+')

ax=yhoo['close'].plot()

plt.xticks(rotation=90)

yhoo.head()



#plt.scatter(yhoo.index, yhoo['close'])
goog=data_df[data_df['security_symbol']=='GOOG']

goog=goog.set_index('date')

#yhoo_close=yhoo.close.values.astype('float32')

#yhoo_close.shape

ax=goog['close'].plot()

plt.xticks(rotation=90)
data_df['date'] = pd.to_datetime(data_df['date'])  

    

#greater than the start date and smaller than the end date

mask = (data_df['date'] < new_date) & (data_df['date'] > past)



# On Covid-19 Pandemic

end = datetime.now()

start = datetime(end.year - 1, end.month, end.day)

goog=data_df[data_df['security_symbol']=='GOOG']

goog=goog.loc[mask]

goog=goog.set_index('date')



#yhoo_close=yhoo.close.values.astype('float32')

#yhoo_close.shape

#ax=yhoo['close'].plot('r+')

ax=goog['close'].plot()

plt.xticks(rotation=90)

goog.head()



#plt.scatter(yhoo.index, yhoo['close'])
fb=data_df[data_df['security_symbol']=='FB']

fb=fb.set_index('date')

ax=fb['close'].plot()

plt.xticks(rotation=90)
data_df['date'] = pd.to_datetime(data_df['date'])  

    

#greater than the start date and smaller than the end date

mask = (data_df['date'] < new_date) & (data_df['date'] > past)



# On Covid-19 Pandemic

end = datetime.now()

start = datetime(end.year - 1, end.month, end.day)

fb=data_df[data_df['security_symbol']=='FB']

fb=fb.loc[mask]

fb=fb.set_index('date')



#yhoo_close=yhoo.close.values.astype('float32')

#yhoo_close.shape

#ax=yhoo['close'].plot('r+')

ax=fb['close'].plot()

plt.xticks(rotation=90)

fb.head()



#plt.scatter(yhoo.index, yhoo['close'])
amzn=data_df[data_df['security_symbol']=='AMZN']

amzn=amzn.set_index('date')

ax=amzn['close'].plot()

plt.xticks(rotation=90)
data_df['date'] = pd.to_datetime(data_df['date'])  

    

#greater than the start date and smaller than the end date

mask = (data_df['date'] < new_date) & (data_df['date'] > past)



# On Covid-19 Pandemic

end = datetime.now()

start = datetime(end.year - 1, end.month, end.day)

amzn=data_df[data_df['security_symbol']=='AMZN']

amzn=amzn.loc[mask]

amzn=amzn.set_index('date')



#yhoo_close=yhoo.close.values.astype('float32')

#yhoo_close.shape

#ax=yhoo['close'].plot('r+')

ax=amzn['close'].plot()

plt.xticks(rotation=90)

amzn.head()



#plt.scatter(yhoo.index, yhoo['close'])
aapl=data_df[data_df['security_symbol']=='AAPL']

aapl=aapl.set_index('date')

ax=aapl['close'].plot()

plt.xticks(rotation=90)
data_df['date'] = pd.to_datetime(data_df['date'])  

    

#greater than the start date and smaller than the end date

mask = (data_df['date'] < new_date) & (data_df['date'] > past)



# On Covid-19 Pandemic

end = datetime.now()

start = datetime(end.year - 1, end.month, end.day)

aapl=data_df[data_df['security_symbol']=='AAPL']

aapl=aapl.loc[mask]

aapl=aapl.set_index('date')



#yhoo_close=yhoo.close.values.astype('float32')

#yhoo_close.shape

#ax=yhoo['close'].plot('r+')

ax=aapl['close'].plot()

plt.xticks(rotation=90)

amzn.head()



#plt.scatter(yhoo.index, yhoo['close'])
ma_yhoo=[]

window_size = 90

i=0

yhoo=data_df[data_df['security_symbol']=='YHOO']



while i<len(yhoo['close'])-window_size+1:

    this_window=yhoo['close'][i:i+window_size]

    window_average=sum(this_window)/window_size

    ma_yhoo.append(window_average)

    i+=1

print(len(yhoo.close))    

print(len(ma_yhoo))



a=np.empty(90-1).tolist()

#print(len(a))

#a=pd.concat(a,ma_yhoo)



a.extend(ma_yhoo)





print(len(a))



plt.plot(a)

plt.title("Yahoo MA 90")
ma_goog=[]

window_size = 90

i=0

goog=data_df[data_df['security_symbol']=='GOOG']



while i<len(goog['close'])-window_size+1:

    this_window=goog['close'][i:i+window_size]

    window_average=sum(this_window)/window_size

    ma_goog.append(window_average)

    i+=1

print(len(goog.close))    

print(len(ma_goog))



a=np.empty(90-1).tolist()

#print(len(a))

#a=pd.concat(a,ma_yhoo)



a.extend(ma_goog)





print(len(a))



plt.plot(a)

plt.title("Google MA 90")
ma_amzn=[]

window_size = 90

i=0

amzn=data_df[data_df['security_symbol']=='AMZN']



while i<len(amzn['close'])-window_size+1:

    this_window=amzn['close'][i:i+window_size]

    window_average=sum(this_window)/window_size

    ma_amzn.append(window_average)

    i+=1

print(len(amzn.close))    

print(len(ma_amzn))



a=np.empty(90-1).tolist()

#print(len(a))

#a=pd.concat(a,ma_yhoo)



a.extend(ma_amzn)





print(len(a))



plt.plot(a)

plt.title("Amazon MA 90")
ma_fb=[]

window_size = 90

i=0

fb=data_df[data_df['security_symbol']=='FB']



while i<len(fb['close'])-window_size+1:

    this_window=fb['close'][i:i+window_size]

    window_average=sum(this_window)/window_size

    ma_fb.append(window_average)

    i+=1





a=np.empty(90-1).tolist()

#print(len(a))

#a=pd.concat(a,ma_yhoo)



a.extend(ma_fb)





print(len(a))



plt.plot(a)

plt.title("FB MA 90")
ma_aapl=[]

window_size = 90

i=0

aapl=data_df[data_df['security_symbol']=='AAPL']



while i<len(aapl['close'])-window_size+1:

    this_window=aapl['close'][i:i+window_size]

    window_average=sum(this_window)/window_size

    ma_aapl.append(window_average)

    i+=1





a=np.empty(90-1).tolist()

#print(len(a))

#a=pd.concat(a,ma_yhoo)



a.extend(ma_aapl)





print(len(a))



plt.plot(a)

plt.title("AAPL MA 90")