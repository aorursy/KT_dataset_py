# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame

import numpy as np



import datetime

import time



import matplotlib.finance as mpf

from matplotlib.pylab import date2num

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('whitegrid')

%matplotlib inline



#import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

spydata = pd.read_csv('../input/SPY Index.csv')

spydata.info()

spydata.head()
data_list = []

for index, row  in spydata.iterrows():

    # 将时间转换为数字

    date_time = datetime.datetime.strptime(row['Date'],'%m/%d/%Y')

    t = date2num(date_time)

    open,high,low,close = row['Open'], row['High'],row['Low'], row['Close']

    datas = (t,open,high,low,close)

    data_list.append(datas)

    

# 创建子图

fig, ax = plt.subplots()

fig.set_size_inches(30, 15)

fig.subplots_adjust(bottom=0.4)

# 设置X轴刻度为日期时间

ax.xaxis_date()

plt.xticks(rotation=45)

plt.yticks()

plt.title("SPY 1993-2017")

plt.xlabel("time")

plt.ylabel("price")

mpf.candlestick_ohlc(ax,data_list,width=1.5,colorup='r',colordown='green')

plt.grid() 
#data_list = []

#for index, row  in spydata.iterrows():

    # 将时间转换为数字

#    date_time = datetime.datetime.strptime(row['Date'],'%m/%d/%Y')

#    Adjclose,volume = row['Adj Close'], row['Volume']

#    datas = (date_time,Adjclose,volume)

#    data_list.append(datas)



#df = pd.DataFrame(data_list,columns=['date','Adj Close','Volume'])



#fig = plt.gcf()

#fig.set_size_inches(30, 15)

#plt.plot(df['date'], df['Adj Close'])
data_list = []

for item in spydata['Date']:    

    date_time = datetime.datetime.strptime(item,'%m/%d/%Y')

    data_list.append(date_time)

spydata['Date'] = data_list

fig = plt.gcf()

fig.set_size_inches(30, 15)

plt.plot(spydata['Date'], spydata['Adj Close'])
spydata['up or down'] = 0

spydata['upper bound'] = 0.0

spydata['lower bound'] = 0.0

a = spydata.shape

for i in range(1,a[0]-1):

    if ((spydata.iat[i,2]-spydata.iat[i-1,2] > 0) & (spydata.iat[i,2]-spydata.iat[i+1,2] > 0) & (spydata.iat[i,3]-spydata.iat[i-1,3] > 0) & (spydata.iat[i,3]-spydata.iat[i+1,3] > 0)):

        spydata.iat[i,7] = 1

    if ((spydata.iat[i,2]-spydata.iat[i-1,2] < 0) & (spydata.iat[i,2]-spydata.iat[i+1,2] < 0) & (spydata.iat[i,3]-spydata.iat[i-1,3] < 0) & (spydata.iat[i,3]-spydata.iat[i+1,3] < 0)):

        spydata.iat[i,7] = -1

    

for i in range(3,a[0]-1):

    if(spydata.iat[i-2,7] == 1):

        spydata.iat[i,8] = spydata.iat[i-2,2]

    else:

        spydata.iat[i,8] = spydata.iat[i-1,8]

        

for i in range(3,a[0]-1):

    if(spydata.iat[i-2,7] == -1):

        spydata.iat[i,9] = spydata.iat[i-2,3]

    else:

        spydata.iat[i,9] = spydata.iat[i-1,9]



#spydata['lower bound']

#spydata['upper bound']    

#spydata['up or down']

spydata.head()

#spydata['up or down'].value_counts()
spydata['upper signal'] = 0.0

spydata['lower siganl'] = 0.0



for i in range(14,a[0]-14):

    max_value = 0.0

    min_value = 1000.0

    for j in range(15):

        max_value = max(max_value,spydata.iat[i+j,8])

        min_value = min(min_value,spydata.iat[i+j,9])

    spydata.iat[i,10] = max_value

    spydata.iat[i,11] = min_value

    

fig = plt.gcf()

fig.set_size_inches(30, 15)

plt.plot(spydata['Date'], spydata['lower siganl'])

plt.plot(spydata['Date'], spydata['Adj Close'])

plt.plot(spydata['Date'], spydata['upper signal'])

plt.show()

spydata.head()

#spydata.index = spydata['Date']

spydata[['lower siganl','Adj Close','upper signal']].ix[100:500].plot(figsize=(12,7))

       
spydata['adjclose']=spydata['Adj Close']

spydata['returns'] = np.log(spydata.adjclose/spydata.adjclose.shift(1))

plt.plot(spydata['Date'], spydata['returns'])

spydata.head()
spydata['buyorsell'] = 0

for i in range(14,a[0]-1):

    if (spydata['Adj Close'].iat[i]-spydata.iat[i,10] > 0):

        spydata['buyorsell'].iat[i] = 1

    if (spydata['Adj Close'].iat[i]-spydata.iat[i,11] < 0): 

        spydata['buyorsell'].iat[i] = -1

 

fig = plt.gcf()

fig.set_size_inches(30, 15)

plt.plot(spydata['Date'], spydata['Adj Close'])

plt.plot(spydata['Date'], spydata['buyorsell'])

plt.show()

#spydata['stragety'] = spydata.buyorsell.shift(1) * spydata.returns

#plt.plot(spydata['Date'], spydata['stragety'])

#spydata[['stragety','returns']].cumsum().apply(np.exp).plot(figsize=(12,7))

  