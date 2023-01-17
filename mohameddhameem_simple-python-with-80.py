# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.mode.chained_assignment = None
from datetime import datetime, timedelta

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/order-brushing-shopee-code-league/order_brush_order.csv')
df.describe(include='all')
df.shopid.nunique()
#Ok. Lets try to split the date/time column into individual feature
df.dtypes
df['event_time'] = pd.to_datetime(df['event_time'], format="%Y/%m/%d %H:%M:%S") #2019-12-27 00:23:03
df.dtypes
df.head()
#Lets try to group order ID based on hour
df['hour'] = df['event_time'].dt.hour
df['year'] = df['event_time'].dt.year
df['month'] = df['event_time'].dt.month
df['day'] = df['event_time'].dt.day
df.head()
dfObj = pd.DataFrame(columns=['shopid','event_time', 'concentrate_rate', 'userid','no_of_users','no_of_orders'])
for shopid in df.shopid.unique():
    #now we need to find out orders that are taken place in one hr. START = First Order, END = 
    #Lets create a new Data frame for each of the Shop ID
    df_shopid = df[df['shopid']==(shopid)]
    df_shopid['event_time'] = pd.DatetimeIndex(df_shopid['event_time'])
    #Lets Sort it
    df_shopid = df_shopid.sort_values('event_time')
    #print(df_shopid.head(20))
    #print(df_shopid.count())
    df_shopid.index=pd.to_datetime(df_shopid.event_time)    
    for index, row in df_shopid.iterrows():
        user_list = []
        start = row['event_time']
        end = start + timedelta(hours=1)
        mask = (df_shopid['event_time'] > start) & (df_shopid['event_time'] <= end)
        df_inbetween = df_shopid.loc[mask]
        #Now Lets try to calculate "Concentrate rate" which is 
        # (Orders ......) / ( User .......)
        concentrate_rate = (df_inbetween['orderid'].nunique() / df_inbetween['userid'].nunique()) if df_inbetween['userid'].nunique() != 0 else 0
        #concentrate_rate = df_inbetween['orderid'].nunique() / df_inbetween['userid'].nunique()
        t_userid = sorted(df_inbetween['userid'].unique())
        #print('concentrate_rate - >',concentrate_rate)
        #print('user id', t_userid)
        #Lets add the data to dataframe
        if concentrate_rate > 0 :
            dfObj = dfObj.append({'shopid' : shopid , 'event_time' : start,
                                  'concentrate_rate' : concentrate_rate,
                                  'userid': '&'.join(map(str, t_userid)),
                                 'no_of_users' : df_inbetween['userid'].nunique(),
                                 'no_of_orders' : df_inbetween['orderid'].nunique()} , ignore_index=True)
        else:
            dfObj = dfObj.append({'shopid' : shopid , 'event_time' : start,
                                  'concentrate_rate' : 0, 'userid': 0,
                                 'no_of_users' : df_inbetween['userid'].nunique(),
                                 'no_of_orders' : df_inbetween['orderid'].nunique()} , ignore_index=True)
    #print(dfObj.head(20))
    #break
#Lets Sort the dataframe by shopid and concentrate_rate
df_sorted = dfObj.sort_values(by=['shopid','concentrate_rate'])
#Check the datatypes one more time
df_sorted.dtypes
#Simple Group by to get the max of concentrate_rate
u = df_sorted.groupby('shopid')['concentrate_rate'].idxmax()
df_final = df_sorted.loc[u, ['shopid', 'userid']].reset_index(drop=1)
#Prepeare for submission
df_submission = pd.DataFrame(df_final)
df_submission.head()
#Lets write the file
df_submission.to_csv('submission.csv', index=False)
#Validate if the total records is 18770 (No. of unique shops)
df_submission.shape