# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#invite people for the Kaggle party

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder
sessions = pd.read_csv('../input/sample_submission.csv')

sessions.head(10)
# Load the data into DataFrames

train_users = pd.read_csv('../input/train_users_2.csv')

test_users = pd.read_csv('../input/test_users.csv')

age_gender_bkts = pd.read_csv('../input/age_gender_bkts.csv')
train_users.head(10)
age_gender_bkts.head(10)
df_train = train_users.copy()

df_test = test_users.copy()
#missing data

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
#descriptive statistics summary

df_train['age'].describe()
df_train = df_train[df_train['age'] <= 100]
#histogram

sns.distplot(df_train['age'].dropna());
df_test.head(10)
df_train = df_train.dropna(subset=['age'])

df_test['age'] = df_test['age'].fillna(36.0)
df_test.head(10)
df_train['first_affiliate_tracked'].unique()

df_test['first_affiliate_tracked'].unique()
df_train['first_affiliate_tracked'].value_counts()

df_test['first_affiliate_tracked'].value_counts()
df_train['first_affiliate_tracked'] = df_train['first_affiliate_tracked'].fillna('untracked')

df_test['first_affiliate_tracked'] = df_test['first_affiliate_tracked'].fillna('untracked')
#missing data

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
df_train = df_train.drop(['date_first_booking','id'], 1)

df_test = df_test.drop(['date_first_booking','id'], 1)




# label_encoder.fit(df_train['country_destination'])

# print(label_encoder.classes_)



# df_train['country_destination'] = label_encoder.transform(df_train['country_destination'])

# print(df_train.head(10))
df_train['gender'].value_counts()
# df_train = df_train.drop(df_train[(df_train['gender'] == '-unknown-')|(df_train['gender'] == 'OTHER')].index)

# df_train['gender'].value_counts()



# df_test = df_test.drop(df_test[(df_test['gender'] == '-unknown-')|(df_test['gender'] == 'OTHER')].index)

# df_test['gender'].value_counts()
# #Label Encoding

label_encoder = preprocessing.LabelEncoder() 



label_encoder.fit(df_train['gender'])

print(label_encoder.classes_)



df_train['gender'] = label_encoder.transform(df_train['gender']) 

df_train.head(10)



df_test['gender'] = label_encoder.transform(df_test['gender']) 
label_encoder.fit(df_train['signup_method'])

print(label_encoder.classes_)



df_train['signup_method'] = label_encoder.transform(df_train['signup_method']) 

df_train.head(10)



df_test.loc[df_test.signup_method == 'weibo', 'signup_method'] = 'google'

df_test['signup_method'] = label_encoder.transform(df_test['signup_method']) 
label_encoder.fit(df_train['language'])

print(label_encoder.classes_)



df_train['language'] = label_encoder.transform(df_train['language']) 

df_train.head(10)



df_test.loc[df_test.language == '-unknown-', 'language'] = 'en'

df_test['language'] = label_encoder.transform(df_test['language']) 



label_encoder.fit(df_train['affiliate_channel'])

print(label_encoder.classes_)



df_train['affiliate_channel'] = label_encoder.transform(df_train['affiliate_channel']) 

df_train.head(10)



df_test['affiliate_channel'] = label_encoder.transform(df_test['affiliate_channel']) 
label_encoder.fit(df_train['affiliate_provider'])

print(label_encoder.classes_)



df_train['affiliate_provider'] = label_encoder.transform(df_train['affiliate_provider']) 

df_train.head(10)



df_test.loc[df_test.affiliate_provider == 'daum', 'affiliate_provider'] = 'other'

df_test['affiliate_provider'] = label_encoder.transform(df_test['affiliate_provider'])
label_encoder.fit(df_train['first_affiliate_tracked'])

print(label_encoder.classes_)



df_train['first_affiliate_tracked'] = label_encoder.transform(df_train['first_affiliate_tracked']) 

df_train.head(10)



df_test['first_affiliate_tracked'] = label_encoder.transform(df_test['first_affiliate_tracked']) 
label_encoder.fit(df_train['signup_app'])

print(label_encoder.classes_)



df_train['signup_app'] = label_encoder.transform(df_train['signup_app']) 

df_train.head(10)



df_test['signup_app'] = label_encoder.transform(df_test['signup_app']) 
label_encoder.fit(df_train['first_device_type'])

print(label_encoder.classes_)



df_train['first_device_type'] = label_encoder.transform(df_train['first_device_type']) 

df_train.head(10)



df_test['first_device_type'] = label_encoder.transform(df_test['first_device_type']) 
label_encoder.fit(df_train['first_browser'])

print(label_encoder.classes_)



df_train['first_browser'] = label_encoder.transform(df_train['first_browser']) 

df_train.head(10)



df_test.loc[df_test.first_browser == 'wOSBrowser', 'first_browser'] = '-unknown-'

df_test.loc[df_test.first_browser == 'Mobile Safari', 'first_browser'] = '-unknown-'

df_test.loc[df_test.first_browser == 'UC Browser', 'first_browser'] = '-unknown-'

df_test.loc[df_test.first_browser == 'IBrowse', 'first_browser'] = '-unknown-'

df_test.loc[df_test.first_browser == 'Nintendo Browser', 'first_browser'] = '-unknown-'





df_test['first_browser'] = label_encoder.transform(df_test['first_browser']) 
df_train['date_account_created_year'] = pd.DatetimeIndex(df_train['date_account_created']).year

df_train['date_account_created_month'] = pd.DatetimeIndex(df_train['date_account_created']).month



df_test['date_account_created_year'] = pd.DatetimeIndex(df_test['date_account_created']).year

df_test['date_account_created_month'] = pd.DatetimeIndex(df_test['date_account_created']).month
df_train = df_train.drop(['date_account_created','timestamp_first_active'], 1)

df_train.head(10)



df_test = df_test.drop(['date_account_created','timestamp_first_active'], 1)
print(df_train.dtypes)
for col in ['gender',

'signup_method',

'signup_flow',

'language',

'affiliate_channel',

'affiliate_provider',

'first_affiliate_tracked',

'signup_app',

'first_device_type',

'first_browser',

'country_destination',

'date_account_created_year',

'date_account_created_month']:

    df_train[col] = df_train[col].astype('str')

print(df_train.dtypes)
feature = ['gender',

 'age',

'signup_method',

'signup_flow',

'language',

'affiliate_channel',

'affiliate_provider',

'first_affiliate_tracked',

'signup_app',

'first_device_type',

'first_browser',

'date_account_created_year',

'date_account_created_month']

target = 'country_destination'
df = df_train.copy()



df_t = df_test.copy()
df_test.head(10)
df_t.head(10)
from sklearn.model_selection import train_test_split



df_train, df_test = train_test_split(df, test_size=0.3, random_state=0)
df_train_target = df_train['country_destination']

df_train = df_train.drop(['country_destination'],1)

df_test_target = df_test['country_destination']

df_test = df_test.drop(['country_destination'],1)
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)

rf.fit(df_train, df_train_target)
from sklearn.metrics import accuracy_score



predicted = rf.predict(df_test)

accuracy = accuracy_score(df_test_target, predicted)



print(f'Out-of-bag score estimate: {rf.oob_score_:.3}')

print(f'Mean accuracy score: {accuracy:.3}')
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)

rf.fit(df_train, df_train_target)
df_t.head(100)
print(df_t.shape)
from sklearn.metrics import accuracy_score



predicted = rf.predict(df_t)

test = pd.DataFrame(columns=['id', 'country'])

test['id'] = test_users['id']

test['country'] = predicted



test.to_csv('submission.csv',index=False)