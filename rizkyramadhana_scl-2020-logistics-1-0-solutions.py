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
import datetime as dt

from datetime import timedelta

data = pd.read_csv('../input/logistics-shopee-code-league/delivery_orders_march.csv')

data
for x in ['pick', '1st_deliver_attempt','2nd_deliver_attempt']:

    data[x] = data[x].apply(lambda y: y + 28800)

    data[x] = pd.to_datetime(data[x], unit = 's')

    data[x] = data[x].dt.date

data
data['dest'] = data['buyeraddress'].apply(lambda x: x.split()[-1].lower())

data['origin'] = data['selleraddress'].apply(lambda x: x.split()[-1].lower())

data
matrix = pd.read_excel('../input/logistics-shopee-code-league/SLA_matrix.xlsx')

matrix
data['is_late']=[0]*len(data)

data['duration']=[7]*len(data)
data.loc[((data['origin']=='luzon') & (data['dest']=='luzon')) | ((data['origin']=='luzon') & (data['dest']=='manila')) | ((data['origin']=='manila') & (data['dest']=='luzon')), 'duration'] = 5

data.loc[(data['origin']=='manila') & (data['dest']=='manila'), 'duration'] = 3
def add_bus_days(date, days):

    def today_is_holiday(somedate):

        return ((somedate.weekday() == 6) or (somedate == dt.date(2020,3,25)) or (somedate == dt.date(2020,3,30)) or (somedate == dt.date(2020,3,31)) )

    while days!=0:

        while today_is_holiday(date):

            date = date + timedelta(days=1)

        date = date + timedelta(days=1)

        days = days-1

    while today_is_holiday(date):

        date = date + timedelta(days=1)

    return date
data['1st_deliver_deadline']=data.apply(lambda x: add_bus_days(x['pick'], x['duration']), axis=1)
data.loc[(data['1st_deliver_attempt'] > data['1st_deliver_deadline']), 'is_late'] = 1

data[data['is_late']==1]
data.loc[(data['2nd_deliver_attempt'].notnull() & (data['2nd_deliver_attempt']>data.apply(lambda x: add_bus_days(x['1st_deliver_attempt'], 3),axis=1))), 'is_late']=1

data[data['is_late']==1]
submission = pd.DataFrame({'orderid':data['orderid'],'is_late':data['is_late']})

submission
submission.to_csv('submission.csv', index = False)