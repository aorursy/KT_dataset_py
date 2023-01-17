# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/hotel-booking-demand'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')
data.describe()
data.shape
data.info()
data.isnull().sum()
data.dtypes
data.head()
columns = data.columns

print(columns)
data['hotel'].value_counts()
data['hotel']= data['hotel'].map({'City Hotel':0,'Resort Hotel':1})
data.head()
data['hotel'].value_counts()
data['arrival_date_month'].value_counts()
data['arrival_date_month'] = data['arrival_date_month'].map({'January':1,'February':2,'March':3,'April':4,

                                                            'May':5,'June':6,'July':7,'August':8,'September':9,'October':10,

                                                            'November':11,'December':10})
data['meal'].value_counts()
data['meal']= data['meal'].map({'BB':0,'HB':1,'SC':2,'Undefined':3,'FB':4})
data['meal'].value_counts()
data['country'].value_counts()
data['market_segment'].value_counts()
data['market_segment']= data['market_segment'].map({'Online TA':0,'Offline TA/TO':1,'Groups':2,'Direct':3,

                                                   'Corporate':4,'Complementary':5,'Aviation':6,'Undefined':7})
data['market_segment'].value_counts()
data['distribution_channel'].value_counts()
data['distribution_channel']=data['distribution_channel'].map({'TA/TO':0,'Direct':1,'Corporate':2,'GDS':3,'Undefined':4})
data['distribution_channel'].value_counts()
data['reserved_room_type'].value_counts()
data['reserved_room_type']=data['reserved_room_type'].map({'A':1,'D':2,'E':3,'F':4,'G':5,'B':6,'C':7,'H':8,'P':9,'L':0})
data['reserved_room_type'].value_counts()
data['assigned_room_type'].value_counts()
data['assigned_room_type']=data['assigned_room_type'].map({'A':1,'D':2,'E':3,'F':4,'G':5,'B':6,'C':7,'H':8,'P':9,'L':0})
data['assigned_room_type'].value_counts()
data['deposit_type'].value_counts()
data['deposit_type']=data['deposit_type'].value_counts({'No Deposit':0,'Non Refund':1,'Refundable':2})
data['deposit_type'].value_counts()