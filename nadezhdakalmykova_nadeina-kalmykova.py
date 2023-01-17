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
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFE, RFECV,VarianceThreshold,chi2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler,Normalizer 
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
train = pd.read_csv("/kaggle/input/hse-pml-2/train_resort.csv")
test = pd.read_csv("/kaggle/input/hse-pml-2/test_resort.csv")
names = pd.read_excel("/kaggle/input/hse-pml-2/column_names.xlsx")

names
train['target'] = train['amount_spent_per_room_night_scaled']
train.drop(columns = ['amount_spent_per_room_night_scaled'], axis = 1, inplace = True)
train['booking_date'] = pd.to_datetime(train['booking_date'])
train['checkin_date'] = pd.to_datetime(train['checkin_date'])
train['checkout_date'] = pd.to_datetime(train['checkout_date'])

test['booking_date'] = pd.to_datetime(test['booking_date'])
test['checkin_date'] = pd.to_datetime(test['checkin_date'])
test['checkout_date'] = pd.to_datetime(test['checkout_date'])
#number of days between booking and checkin
train['checkin_booking'] = train['checkin_date'] - train['booking_date']
train['checkin_booking'] = train['checkin_booking'].dt.days

test['checkin_booking'] = test['checkin_date'] - test['booking_date']
test['checkin_booking'] = test['checkin_booking'].dt.days

#number of days spent on vacation

train['days_spent'] = train['checkout_date'] - train['checkin_date']
train['days_spent'] = train['days_spent'].dt.days

test['days_spent'] = test['checkout_date'] - test['checkin_date']
test['days_spent'] = test['days_spent'].dt.days
#total number of travellers
train['travellers'] = train['numberofadults'] + train['numberofchildren']
test['travellers'] = test['numberofadults'] + test['numberofchildren']
def preproc_text(column):
    le = LabelEncoder()
    train[column] = le.fit_transform(train[column])
    test[column] = le.fit_transform(test[column])
preproc_text('reservation_id')
preproc_text('main_product_code')
preproc_text('persontravellingid')
preproc_text('resort_type_code')
preproc_text('resort_region_code')
preproc_text('member_age_buckets')
preproc_text('cluster_code')
preproc_text('reservationstatusid_code')
preproc_text('resort_id')
train.columns
columns_check = [ 'channel_code', 'main_product_code', 'numberofadults',
      'numberofchildren', 'resort_region_code',
       'resort_type_code', 'room_type_booked_code', 'roomnights',
      'season_holidayed_code', 'state_code_residence', 'state_code_resort',
      'total_pax', 'member_age_buckets', 'booking_type_code',
      'cluster_code', 'reservationstatusid_code', 'resort_id', 
      'checkin_booking', 'days_spent', 'travellers']
columns = [x for x in columns_check]
for col in columns:
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title(f'train {col}')
    ax[0].hist(train[col])
    ax[0].grid()
    ax[1].hist(test[col])
    ax[1].set_title(f'test {col}')
    ax[1].grid()
plt.show()
train.drop(columns = ['roomnights'], axis = 1, inplace = True)
train.drop(columns = ['total_pax'], axis = 1, inplace = True)
# посчитали сами, в этой переменной содержится ошибка
train['target'].hist()
train['checkin_booking'].hist()
print(pd.unique(train['checkin_booking'].sort_values()))
sns.distplot(train['days_spent'])
print(pd.unique(train['days_spent'].sort_values()))

print(train['days_spent'].sort_values().tail(5))
train['channel_code'].hist()
print(pd.unique(train['channel_code'].values))
train['main_product_code'].hist()
sns.distplot(train['main_product_code'])
print(pd.unique(train['main_product_code'].values))
train['resort_region_code'].hist()
sns.distplot(train['resort_region_code'])
print(pd.unique(train['resort_region_code'].values))
sns.distplot(train['room_type_booked_code'])
train['season_holidayed_code'].hist()
print(pd.unique(train['season_holidayed_code'].values))
train['state_code_residence'].hist()
print(pd.unique(train['state_code_residence'].values))
train['numberofadults'].hist()
print(pd.unique(train['numberofadults'].sort_values()))

print(train['numberofadults'].sort_values().head(7))
print(train['numberofadults'].sort_values().tail(7))
train['numberofchildren'].hist()
print(pd.unique(train['numberofchildren'].sort_values()))
print(train['numberofchildren'].sort_values().tail(7))
train['member_age_buckets'].hist()
print(pd.unique(train['member_age_buckets'].sort_values()))
r=train.sort_values(['booking_date'])
print(r[r.columns[12]].head(5))
print(r[r.columns[1:4]].head(5))
q=train.sort_values(['numberofchildren'])
q[['numberofchildren', 'numberofadults']].tail(10)
print(train['target'].min())
print(train['target'].max())
print(train['target'].mean())
print(train['target'].std())
sns.distplot(train['target'])
sns.boxplot(train['target'])
print(train['target'].sort_values().head(200))
print(train['target'].sort_values().tail(200))