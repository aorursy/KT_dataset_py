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
#get df
df = pd.read_csv('/kaggle/input/orderbrushing/order_brush_order.csv')
df[:2]
#get general idea of data
df.shopid.nunique(), df.orderid.nunique()
#get overall concentrate rate
df_sum = df.groupby('shopid').agg({'userid':'nunique','orderid':'nunique'}).reset_index()
df_sum['concentrate_rate_est'] = df_sum['orderid']/ df_sum['userid']
#filter out those with one transaction
df_sum_f1 = df_sum[df_sum.userid > 1]

#filter out those with concentrate_rate_est == 1 (means each transaction have unique id)
df_sum_f2 = df_sum_f1[df_sum_f1.concentrate_rate_est > 1]

len(df_sum_f1), len(df_sum_f2)
#we have 3,659 shops that we can investigate further. 
#only get transaction for these shops
df_filter_1 = df[df.shopid.isin(df_sum_f2.shopid.unique())]
len(df_filter_1), df_filter_1.shopid.nunique()
#function to get 
import numpy as np
import tqdm

#select shop with multiple transaction per hour
def multiple_tranc_perH(shop_transaction):
    
    #return df
    return_df = pd.DataFrame()
    
    #add index - easier to remove duplicates
    shop_transaction['index'] = shop_transaction.index
    
    #convert column to datetime
    shop_transaction['event_time'] = pd.to_datetime(shop_transaction['event_time'])
    
    #order transaction by time
    shop_transaction = shop_transaction.sort_values('event_time')
    
    
    #populate value in time dif column
    for i, item in enumerate(shop_transaction['event_time']):
        
        #processing, get ratio
        if i == 0:
            pass
        else:
            start_time = pd.to_datetime(shop_transaction['event_time'].iloc[i-1])
            end_time = start_time +  np.timedelta64(1, 'h')
            transc_perH = shop_transaction[(shop_transaction.event_time >= start_time) &
                                          (shop_transaction.event_time <= end_time)]
            
            try:
                ratio = transc_perH.orderid.nunique()/transc_perH.userid.nunique()
            except ZeroDivisionError:
                ratio = 0

            
            if ratio >= 3:
                print(start_time,end_time,ratio,transc_perH.orderid.nunique())
                #add_start_time_col
                transc_perH['start_time'] = start_time
                transc_perH['ratio'] = ratio
                return_df = pd.concat([return_df,transc_perH])
            
    return return_df


import tqdm

temp_df = pd.DataFrame()
for shops in tqdm.tqdm(df_filter_1.shopid.unique()):
    print(shops)
    shop_transaction = df_filter_1[df_filter_1.shopid == shops]

    try:
        sub_df = multiple_tranc_perH(shop_transaction)
        temp_df = pd.concat([temp_df,sub_df])
    except Exception as e:
        print(e)
        pass
#all events with order brushing
#may contain duplicates - duplicates means all are same, except for start_time
temp_df[temp_df.shopid == 181009364]
