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
import time
import pickle
from collections import OrderedDict
from datetime import datetime, timedelta
from dateutil import rrule
datetime.fromtimestamp(1583138397) - timedelta(hours=4)
df = pd.read_csv('../input/logistics-shopee-code-league/delivery_orders_march.csv')
sla = pd.read_excel('../input/logistics-shopee-code-league/SLA_matrix.xlsx')
df.head()
first = list(map(datetime.fromtimestamp, df['1st_deliver_attempt'].tolist()))
pick = list(map(datetime.fromtimestamp, df['pick'].tolist()))
df
df['1st_deliver_attempt'] = first
df['pick'] = pick
df.head()
second_list = []
for idx,row in df.iterrows():
    second = row['2nd_deliver_attempt']
    if np.isnan(second):
        second_list.append(second)
    else:
        second_list.append(datetime.fromtimestamp(second))
second_list[0:5]
df['2nd_deliver_attempt'] = second_list
df['pick'] = pd.to_datetime(df['pick']).dt.date
df['1st_deliver_attempt'] = pd.to_datetime(df['1st_deliver_attempt']).dt.date
df['2nd_deliver_attempt'] = pd.to_datetime(df['2nd_deliver_attempt']).dt.date
df.head(5)
import datetime
def date_diff(a,b):

    diff_business_days = len(list(rrule.rrule(rrule.DAILY,
                                              dtstart=a,
                                              until=b - datetime.timedelta(days=1),
                                              byweekday=(rrule.MO, rrule.TU, rrule.WE, rrule.TH, rrule.FR,rrule.SA))))
    
    
    wed_holi = datetime.datetime.strptime('2020-03-25', '%Y-%m-%d').date()
    mon_holi = datetime.datetime.strptime('2020-03-30', '%Y-%m-%d').date()
    tue_holi = datetime.datetime.strptime('2020-03-31', '%Y-%m-%d').date()
    
    if a <= wed_holi <= b:
        diff_business_days = diff_business_days - 1
    
    if a <= mon_holi <= b:
        diff_business_days = diff_business_days - 1
        
    if a <= tue_holi <= b:
        diff_business_days = diff_business_days - 1

    return diff_business_days
date_diff(df[1284078:1284079]['pick'][1284078],df[1284078:1284079]['1st_deliver_attempt'][1284078])
date_diff_1 = df.apply(lambda x: date_diff(x['pick'], x['1st_deliver_attempt']), axis=1)
df['date_diff_1'] = date_diff_1
df.head()
Address = {'metro manila','luzon','visayas','mindanao'}
Address_dict = {('metro manila','metro manila'):3,('metro manila','luzon'):5,('metro manila','visayas'):7,('metro manila','mindanao'):7,
                ('luzon','metro manila'):5,('luzon','luzon'):5,('luzon','visayas'):7,('luzon','mindanao'):7,
                ('visayas','metro manila'):7,('visayas','luzon'):7,('visayas','visayas'):7,('visayas','mindanao'):7,
                ('mindanao','metro manila'):7,('mindanao','luzon'):7,('mindanao','visayas'):7,('mindanao','mindanao'):7
               }
# output only address we want
def last_12(x):
    s = x[-12:].strip(' ').lower()
    for a in Address:
        if a in s:
            return a
    return 0
df.selleraddress = df.selleraddress.apply(lambda x:last_12(x))
df.buyeraddress = df.buyeraddress.apply(lambda x:last_12(x))
def agg(x):
    t = (x['buyeraddress'],x['selleraddress'])
    #print(t)
    return Address_dict[t]
df['working_days'] = df.apply(agg,axis=1)
df.head()
df['2nd_deliver_attempt'] = pd.to_datetime(df['2nd_deliver_attempt'])
df['1st_deliver_attempt'] = pd.to_datetime(df['1st_deliver_attempt'])
df['pick'] = pd.to_datetime(df['pick'])
date_diff_2 = df.apply(lambda x: date_diff(x['1st_deliver_attempt'], x['2nd_deliver_attempt']) 
                             if not pd.isnull(x['2nd_deliver_attempt']) else 999
                             , axis=1)
date_diff_2
df['date_diff_2'] = date_diff_2
def get_answer(date_diff_1,date_diff_2,working_days):
    if date_diff_1 > working_days:
        return 1
    elif date_diff_2 == 999:
        return 0
    else:
        if date_diff_2 > 3:
            return 1
    return 0
answer = df.apply(lambda x: get_answer(x['date_diff_1'], x['date_diff_2'], x['working_days']), axis=1)
answer
df['is_late'] = answer
df_submit = df[['orderid','is_late']]
df_submit.to_csv('submission.csv',index = False)
df_submit['is_late'].value_counts()
pd.isnull(df['2nd_deliver_attempt'][0])
