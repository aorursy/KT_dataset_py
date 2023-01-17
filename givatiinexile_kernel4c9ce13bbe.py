import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

import os
filename="../input/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv"
df=pd.read_csv(filename,usecols=['Timestamp','High','Low'],sep=',')
df['Timestamp'] = pd.to_datetime(df['Timestamp'],unit='s')

df['date'] = df['Timestamp'].dt.date

df['year'] = df['Timestamp'].dt.year

df = df[df.date.notnull() & df.High.notnull() & df.Low.notnull()]
df.head()
#Find max drawdown for each day

df1=df.groupby(

   ['date']

).agg(

    {

         'High':max,    

         'Low':min  

    }

)
df1['max_drawdown']=df1['High']-df1['Low']

df1['Percent drawdown']=(((df1['High']-df1['Low'])/df1['Low'].abs())*100).round(2)
#MAX DAILY DRAWDOWN FOR ALL AVAILABLE DATA

idx = df1.groupby(['date'])['max_drawdown'].transform(max) == df1['max_drawdown']

df1[idx].max()
df2=df.groupby(

   ['date','year']

).agg(

    {

         'High':max,   

         'Low':min  

    }

)
df2['max_drawdown']=df2['High']-df2['Low']

df2['Percent drawdown']=(((df2['High']-df2['Low'])/df2['Low'].abs())*100).round(2)
#MAX DAILY MAX DRAWDOWNS FOR EACH YEAR

idx2 = df2.groupby(['year'])['max_drawdown'].transform(max) == df2['max_drawdown']

df2[idx2]
#Find max drawdown for each day

df3=df.groupby(

   ['date','year']

).agg(

    {

         'High':max,    

         'Low':min  

    }

)
df3['max_drawdown']=df3['High']-df3['Low']

df3['Percent drawdown']=(((df3['High']-df3['Low'])/df3['Low'].abs())*100).round(2)
#ANNUAL SUMMATION OF DAILY MAX DRAWDOWNS

df3.groupby(['year'])['max_drawdown'].agg('sum')