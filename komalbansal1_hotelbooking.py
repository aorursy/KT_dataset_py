# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")
df.head()
df.describe()
df.shape
print("Nan in each columns" , df.isna().sum(),sep='\n')
df = df.drop(['company'],axis=1)

df.head()
df.info()
df['agent'] = df['agent'].fillna(df['agent'].mean())
df['hotel'] = df['hotel'].map({'Resort Hotel':0, 'City Hotel':1})

df['hotel'].unique()
df = df.drop(['country'], axis=1)
df['arrival_date_month'] = df['arrival_date_month'].map({'January':1, 'February': 2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,

                                                            'August':8, 'September':9, 'October':10, 'November':11, 'December':12})

df['arrival_date_month'].unique()
df['customer_type'].unique()
df['deposit_type'].unique()
df['reservation_status'].unique()
df['assigned_room_type'].unique()
df.columns
number_features = ['hotel', 'is_canceled', 'lead_time', 'arrival_date_year',

       'arrival_date_month', 'arrival_date_week_number',

       'arrival_date_day_of_month', 'stays_in_weekend_nights',

       'stays_in_week_nights', 'adults', 'children', 'babies', 'meal',

       'country', 'market_segment', 'distribution_channel',

       'is_repeated_guest', 'previous_cancellations',

       'previous_bookings_not_canceled', 'reserved_room_type',

       'assigned_room_type', 'booking_changes', 'deposit_type', 'agent',

       'company', 'days_in_waiting_list', 'customer_type', 'adr',

       'required_car_parking_spaces', 'total_of_special_requests',

       'reservation_status', 'reservation_status_date']
df = df.replace([np.inf, -np.inf], np.nan)
from sklearn import preprocessing

le = preprocessing.LabelEncoder() 
df['customer_type']= le.fit_transform(df['customer_type'].astype(str)) 

df['assigned_room_type'] = le.fit_transform(df['assigned_room_type'].astype(str))

df['deposit_type'] = le.fit_transform(df['deposit_type'].astype(str))

df['reservation_status'] = le.fit_transform(df['reservation_status'].astype(str))

df['meal'] = le.fit_transform(df['meal'].astype(str))

#df['country'] = le.fit_transform(df['country'].astype(str))

df['distribution_channel'] = le.fit_transform(df['distribution_channel'].astype(str))

df['market_segment'] = le.fit_transform(df['market_segment'].astype(str))

df['reserved_room_type'] = le.fit_transform(df['reserved_room_type'].astype(str))

print('customer_type:', df['customer_type'].unique())

print('reservation_status', df['reservation_status'].unique())

print('deposit_type', df['deposit_type'].unique())

print('assigned_room_type', df['assigned_room_type'].unique())

print('meal', df['meal'].unique())

#print('Country:',df['country'].unique())

print('Dist_Channel:',df['distribution_channel'].unique())

print('Market_seg:', df['market_segment'].unique())

print('reserved_room_type:', df['reserved_room_type'].unique())
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree, metrics

from sklearn.ensemble import RandomForestClassifier
df = df.drop(['reservation_status_date'],axis=1)
df['children'] = df['children'].fillna(df['children'].mean())
X = df.drop(['previous_cancellations'], axis = 1)

Y = df['previous_cancellations']
from sklearn.linear_model import LogisticRegression
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
df.head()
print("Nan in each columns" , df.isna().sum(),sep='\n')
regressor = LogisticRegression(max_iter=2000)

regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)
Y_test
y_pred
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))