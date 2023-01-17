# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataframe = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
use_columns = list(dataframe.columns)
use_columns.remove('previous_cancellations')
pd.set_option('display.max_columns', None)

display(dataframe.head())

pd.set_option('display.max_columns', 10)
dataframe.info()
dataframe['company'] = dataframe['company'].fillna('no company')

dataframe['agent'] = dataframe['agent'].fillna('no agent')
display(f"Number of unique companies '{dataframe.company.unique().shape[0]}' ")

display(f"Number of unique agent '{dataframe.agent.unique().shape[0]}' ")
dataframe.loc[dataframe['company'] != 'no company', 'company'] = 1

dataframe.loc[dataframe['company'] == 'no company', 'company'] = 0
dataframe.loc[dataframe['agent'] != 'no agent', 'agent'] = 1

dataframe.loc[dataframe['agent'] == 'no agent', 'agent'] = 0
sns.countplot(dataframe['is_canceled'])

plt.title("Number of cancel booking and not cancel booking")

plt.show()
use_columns.remove('arrival_date_year')

use_columns.remove('arrival_date_month')

use_columns.remove('arrival_date_week_number')

use_columns.remove('arrival_date_day_of_month')
pd.DataFrame(dataframe.groupby(['is_canceled', 'hotel']).size())
use_columns.remove('hotel')
dataframe.groupby(['assigned_room_type','is_canceled']).size()
dataframe.groupby(['reserved_room_type','is_canceled']).size()
use_columns.remove('reserved_room_type')

use_columns.remove('assigned_room_type')
dataframe.groupby(['deposit_type','is_canceled']).size()
pd.DataFrame(dataframe.groupby(['is_canceled'])['days_in_waiting_list'].mean())
pd.DataFrame(dataframe.groupby(['is_canceled'])['days_in_waiting_list'].std())
pd.DataFrame(dataframe.groupby('is_canceled')['days_in_waiting_list'].median())
use_columns.remove('days_in_waiting_list')
dataframe.groupby(['total_of_special_requests', 'is_canceled']).size()
dataframe.groupby(['is_canceled', 'required_car_parking_spaces']).size()
use_columns.remove('reservation_status_date')

use_columns.remove('reservation_status')
labelEncoder = LabelEncoder()

encoded_dataframe = dataframe[use_columns].apply(lambda x: labelEncoder.fit_transform(x.astype(str)))
x_train, x_test, y_train, y_test = train_test_split(encoded_dataframe.drop('is_canceled', axis=1), encoded_dataframe['is_canceled'])
model = RandomForestClassifier()

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
confusion_matrix(y_pred, y_test)
pd.DataFrame.from_dict([dict(zip(encoded_dataframe.drop('is_canceled', axis=1), model.feature_importances_))]).T.sort_values(0)
pd.DataFrame.from_dict([dict(zip(encoded_dataframe.drop('is_canceled', axis=1), model.feature_importances_))]).T.sort_values(0)[:6].index
x_train, x_test, y_train, y_test = train_test_split(encoded_dataframe.drop(['babies', 'is_repeated_guest', 'company',

       'previous_bookings_not_canceled', 'agent', 'children', 'is_canceled'], axis=1), encoded_dataframe['is_canceled'])
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

confusion_matrix(y_pred, y_test)
pd.DataFrame.from_dict([dict(zip(x_train, model.feature_importances_))]).T.sort_values(0)