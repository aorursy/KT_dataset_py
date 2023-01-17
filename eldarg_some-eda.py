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
train = pd.read_csv('/kaggle/input/hse-pml-2/train_resort.csv')
test = pd.read_csv('/kaggle/input/hse-pml-2/test_resort.csv')
train.head()
test.head()
train.shape
test.shape
train.booking_date = pd.to_datetime(train.booking_date, format="%Y/%m/%d")
train.checkin_date = pd.to_datetime(train.checkin_date, format="%Y/%m/%d")
train.checkout_date = pd.to_datetime(train.checkout_date, format="%Y/%m/%d")

test.checkin_date = pd.to_datetime(test.checkin_date, format="%Y/%m/%d")
test.checkout_date = pd.to_datetime(test.checkout_date, format="%Y/%m/%d")
test.booking_date = pd.to_datetime(test.booking_date, format="%Y/%m/%d")
train.booking_date.dt.year.value_counts()
test.booking_date.dt.year.value_counts()
train.checkin_date.dt.year.value_counts()
test.checkin_date.dt.year.value_counts()
train[train['checkin_date'] > train['checkout_date']]
train[train['booking_date'] > train['checkin_date']]
test[test['booking_date'] > test['checkin_date']]
test[test['checkin_date'] > test['checkout_date']]
train.booking_date.sort_values()
test.booking_date.sort_values()
train['book_year'] = train.booking_date.dt.year
import seaborn as sns
sns.boxplot(x='book_year', y='amount_spent_per_room_night_scaled', data=train)
import matplotlib.pyplot as plt
plt.hist(train.amount_spent_per_room_night_scaled)
train.checkin_date.dt.weekday.value_counts()
test.checkin_date.dt.weekday.value_counts()
train.columns
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].hist(train['numberofadults'])
ax[1].hist(test['numberofadults'])
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(train['numberofchildren'])
ax[1].hist(test['numberofchildren'])
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(train['total_pax'])
ax[1].hist(test['total_pax'])
train[train['total_pax'] < train['numberofadults']] 
test[test['total_pax'] < test['numberofadults']] 
train[train['numberofadults']==0]
test[test['numberofadults']==0]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(train['roomnights'])
ax[1].hist(test['roomnights'])
train[train['roomnights']<=0][['roomnights', 'booking_date', 'checkin_date', 'checkout_date']]
test[test['roomnights']<=0][['roomnights', 'booking_date', 'checkin_date', 'checkout_date']]
cat_columns = ['channel_code', 'main_product_code',
        'persontravellingid', 'resort_region_code',
       'resort_type_code', 'room_type_booked_code', 
       'season_holidayed_code', 'state_code_residence', 'state_code_resort',
        'member_age_buckets', 'booking_type_code', 'memberid',
       'cluster_code', 'reservationstatusid_code', 'resort_id']
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(train['channel_code'])
ax[1].hist(test['channel_code'])
train['state_code_residence'].isna().value_counts()
test['state_code_residence'].isna().value_counts()
train['season_holidayed_code'].isna().value_counts()
test['season_holidayed_code'].isna().value_counts()