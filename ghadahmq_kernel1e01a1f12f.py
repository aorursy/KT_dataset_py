# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib as style

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_stocks=pd.read_csv('/kaggle/input/stocksdata/StocksData')
df_stocks.head(50)
df_stocks.shape
df_stocks.info()
df_stocks['Volume Traded'] = df_stocks['Volume Traded'].apply(lambda x: (x.replace(',' ,'')))

df_stocks
df_stocks['Date']=pd.to_datetime(df_stocks['Date'])

df_stocks['Volume Traded'] = df_stocks['Volume Traded'].astype('int')
df_stocks.drop(labels='Unnamed: 0',

    axis=1,inplace=True)
bank_Industry=df_stocks[:300]

cement_Industry=df_stocks[300:]
cement_Industry.head()
cement_Industry.reset_index(drop=True,inplace=True)
cement_Industry[0:30]
cement_Industry['Trading name'].unique()[2]
list_=cement_Industry['Trading name'].unique()
n=0

start=0

end=0

plt.style.use('seaborn-darkgrid')

for i in range(0,10):

 

  

    plt.plot(cement_Industry['Open'][start:30+end], cement_Industry['Date'][start:30+end], marker='', label=list_[n])

    plt.title(list_[n], loc='left', fontsize=12, fontweight=0 )

    plt.suptitle("How the  Company did in the past perior from 2020-02-06 to 2020-03-19 in Open price?", fontsize=13, fontweight=2, color='black', style='italic', y=1.02)

   



    plt.show()

    n=n+1

    start=start+30

    end=end+30

    
import numpy as np

n=0

start=0

end=0

for i in range(0,10):

    fig = plt.figure()

    fig, ax = plt.subplots(figsize=(10,8))

    # evenly sampled time at 200ms intervals

    t = np.arange(0., 5., 0.2)

    #fig.figure(figsize=(10,8))

    # red dashes, blue squares and green triangles

    ax.plot(cement_Industry.Date[start:30+end], cement_Industry['Open'][start:30+end], 'bs-',label='Open $Price')

    ax.plot( cement_Industry.Date[start:30+end], cement_Industry['Close'][start:30+end], 'g^-',label='Close $Price')

    ax.legend()

    plt.suptitle('Show the price close and open every day for one month ',fontsize=20)

    plt.title(list_[n],fontsize=15)

    plt.show()

    n=n+1

    start=start+30

    end=end+30

    
df_max=cement_Industry.groupby(['Trading name'])[['Change%']].max()
df_max=df_max.reset_index()
df_max['Date']=[0,0,0,0,0,0,0,0,0,0]
start=0

end=0



for i in range(0,10):

    

    for j in range(0,300):

        

        if df_max['Trading name'][i] == cement_Industry['Trading name'][j]:

            

            if df_max['Change%'][i] == cement_Industry['Change%'][j]:

                df_max['Date'][i]=cement_Industry['Date'][j]

                

                if end < 301:

                    start = start+30

                    end = end+30





df_max.info()
df_max
bank_Industry
list_n=bank_Industry['Trading name'].unique()
n=0

start=0

end=0

plt.style.use('seaborn-darkgrid')

for i in range(0,10):

 

  

    plt.plot(bank_Industry['Open'][start:30+end], bank_Industry['Date'][start:30+end], marker='', label=list_n[n])

    plt.title(list_n[n], loc='left', fontsize=12, fontweight=0 )

    plt.suptitle("How the  Company did in the past perior from 2020-02-06 to 2020-03-18 in Open price?", fontsize=13, fontweight=2, color='black', style='italic', y=1.02)

   



    plt.show()

    n=n+1

    start=start+30

    end=end+30

  
import numpy as np

n=0

start=0

end=0

for i in range(0,10):

    fig = plt.figure()

    fig, ax = plt.subplots(figsize=(10,8))

    # evenly sampled time at 200ms intervals

    t = np.arange(0., 5., 0.2)

    #fig.figure(figsize=(10,8))

    # red dashes, blue squares and green triangles

    ax.plot(bank_Industry.Date[start:30+end], bank_Industry['Open'][start:30+end], 'bs-',label='Open $Price')

    ax.plot( bank_Industry.Date[start:30+end], bank_Industry['Close'][start:30+end], 'g^-',label='Close $Price')

    ax.legend()

    plt.suptitle('Show the price close and open every day for one month ',fontsize=20)

    plt.title(list_n[n],fontsize=15)

    plt.show()

    n=n+1

    start=start+30

    end=end+30
df_max_=bank_Industry.groupby(['Trading name'])[['Change%']].max()
df_max_
df_max_['Date']=[0,0,0,0,0,0,0,0,0,0]
df_max_=df_max_.reset_index()
start=0

end=0



for i in range(0,10):



    for j in range(0,300):

        

        if df_max_['Trading name'][i] == bank_Industry['Trading name'][j]:

            

            if df_max_['Change%'][i] == bank_Industry['Change%'][j]:

                

                df_max_['Date'][i]=bank_Industry['Date'][j]



                if end < 301:

                    

                    start = start+30

                    end = end+30





df_max_.info()
df_max_