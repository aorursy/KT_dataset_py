import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
plt.rcParams['font.size'] = 12

plt.rcParams['figure.figsize'] = (14,7)

sns.set_style('whitegrid')
train = pd.read_csv("/kaggle/input/hse-pml-2/train_resort.csv")

test = pd.read_csv("/kaggle/input/hse-pml-2/test_resort.csv")
sns.distplot(train.amount_spent_per_room_night_scaled, rug=True);
train.columns[np.where(train.isna().any())]
test.columns[np.where(test.isna().any())]
train.season_holidayed_code.isna().value_counts()
test.season_holidayed_code.isna().value_counts()
train.state_code_residence.isna().value_counts()
test.state_code_residence.isna().value_counts()
from datetime import datetime



def get_date(date):

    return datetime.strptime(date, "%Y-%m-%d")



train['booking_date'] = train['booking_date'].apply(get_date)

train['checkin_date'] = train['checkin_date'].apply(get_date)

train['checkout_date'] = train['checkout_date'].apply(get_date)

test['booking_date'] = test['booking_date'].apply(get_date)

test['checkin_date'] = test['checkin_date'].apply(get_date)

test['checkout_date'] = test['checkout_date'].apply(get_date)
train['booking_date'].sort_values()
test['booking_date'].sort_values()
train['checkin_date'].sort_values()
test['checkin_date'].sort_values()
train['checkout_date'].sort_values()
test['checkout_date'].sort_values()
test[test['checkin_date'] < '2015-01-01']
train.reservation_id[0]
train.columns
cat_cols = ['channel_code',

 'main_product_code',

 'persontravellingid',

 'resort_region_code',

 'resort_type_code',

 'room_type_booked_code',

 'season_holidayed_code',

 'state_code_residence',

 'state_code_resort',

 'booking_type_code',

 'cluster_code',

 'reservationstatusid_code']
len(cat_cols)
fig, axes = plt.subplots(12, 2, figsize=(12, 12*5))

for ax, col in zip(axes, cat_cols):

    ax[0].hist(train[col])

    ax[0].set_title('train '+ col)

    ax[0].set_xlabel(col)

    ax[1].hist(test[col])

    ax[1].set_title('test '+ col)

    ax[1].set_xlabel(col)

plt.tight_layout();
num_cols = ['numberofadults', 'numberofchildren', 'roomnights', 'total_pax']
fig, axes = plt.subplots(4, 2, figsize=(12, 4*5))

for ax, col in zip(axes, num_cols):

    ax[0].hist(train[col])

    ax[0].set_title('train '+ col)

    ax[0].set_xlabel(col)

    ax[1].hist(test[col])

    ax[1].set_title('test '+ col)

    ax[1].set_xlabel(col)

plt.tight_layout();
train.loc[train['roomnights'] <= 0, 'roomnights']
def intersects(train, test, cols):

    res = pd.DataFrame(index=cols,

                       columns=['# common', 'intersection fraction (train)', 'intersection fraction (test)'])

    for col in cols:

        unq_test = set(test[col].unique())

        unq_train = set(train[col].unique())

        n_common = len(unq_test & unq_train) 

        res.loc[col, '# common'] = n_common

        res.loc[col, 'intersection fraction (train)'] = n_common / len(unq_train)

        res.loc[col, 'intersection fraction (test)'] = n_common / len(unq_test)

    return res
intersects(train, test, cat_cols)
intersects(train, test, ['memberid'])