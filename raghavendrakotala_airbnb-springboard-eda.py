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
#loading all the files and checking their shapes respectively

train_df = pd.read_csv('../input/train_users_2.csv',parse_dates=['date_account_created'])

test_df = pd.read_csv('../input/test_users.csv',parse_dates=['date_account_created'])

age_gender_df = pd.read_csv('../input/age_gender_bkts.csv')

countries_df = pd.read_csv('../input/countries.csv')

session_df = pd.read_csv('../input/sessions.csv')

train_df.shape,test_df.shape,age_gender_df.shape,countries_df.shape,session_df.shape
# lets see first 5 rows

train_df.head()
#lets inspect the columns

train_df.info()
#first find out any missing values

(train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
#lets see value counts for each of the categorical columns to  see if any of the values hardcoded in some common formate

category_columns = list(train_df.columns[train_df.dtypes == np.object].values)

category_columns.remove('date_first_booking') #doesnt make any sense to  see how many unique dates are present

category_columns.remove('country_destination') #target column 

category_columns.remove('id') #because it's unique one

print(category_columns)
#lets print values counts for above columns

for a in category_columns:

    print(train_df[a].value_counts(),'value coutns',a)
#lets analysis for remainng columns

remaining_columns = list(train_df.columns[train_df.dtypes != np.object].values)

remaining_columns



#we can convert timestamp_first_active to date one

train_df.timestamp_first_active = train_df.timestamp_first_active.apply(lambda x:pd.to_datetime(str(x)))
import seaborn as sns
train_df.age.apply(lambda x : x>120).sum()
sns.barplot(data=train_df,x='age',y='country_destination')
train_df.first_browser.value_counts().sort_values(ascending=False)/train_df.first_browser.count()
sns.countplot(data=train_df,x='signup_method')
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 6))

sns.countplot(data=train_df,x='affiliate_provider')
sns.countplot(data=train_df,x='language')
plt.figure(figsize=(16, 6))

sns.countplot(data=train_df,x='affiliate_channel')
plt.figure(figsize=(16, 6))

sns.countplot(data=train_df,x='first_device_type')
sns.countplot(data=train_df,x='country_destination')
# category_columns = list(train_df.columns[train_df.dtypes == np.object].values)
##Han
train_df.columns
# train_df['gender']= train_df.gender.replace('-unknown-',np.nan)
train_df.gender.value_counts()
train_df.isnull().sum().sort_values(ascending=False)
# train_df['first_browser']= train_df.first_browser.replace('-unknown-',np.nan)
#handling Age column

train_df.loc[train_df.age>120,'age'] = np.nan

train_df.age.fillna(train_df.age.mean(),inplace=True)
train_df.isnull().sum().sort_values(ascending=False)
train_df.age.isnull().sum()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
session_df.isnull().sum()
session_df.head()
session_df.info()
total_seconds = session_df.groupby('user_id')['secs_elapsed'].sum()
average_seconds = session_df.groupby('user_id')['secs_elapsed'].mean()
total_sessions = session_df.groupby('user_id')['secs_elapsed'].count()
unique_sessions = session_df.groupby('user_id')['secs_elapsed'].nunique()
total_seconds.head(),total_sessions.head(),average_seconds.head(),unique_sessions.head()
num_short_sessions = session_df[session_df['secs_elapsed'] <= 300].groupby('user_id')['secs_elapsed'].count()

num_long_sessions = session_df[session_df['secs_elapsed'] >= 2000].groupby('user_id')['secs_elapsed'].count()
num_long_sessions.head(),num_short_sessions.head()
num_devices = session_df.groupby('user_id')['device_type'].nunique()
def session_id_features(df):

    df['total_seconds'] = df['id'].apply(lambda x:total_seconds[x] if x in total_seconds else 0)

    df['average_seconds'] = df['id'].apply(lambda x:average_seconds[x] if x in average_seconds else 0)

    df['total_sessions'] = df['id'].apply(lambda x:total_sessions[x] if x in  total_sessions else 0)

    df['unique_sessions'] = df['id'].apply(lambda x:unique_sessions[x] if x in  unique_sessions else 0)

    df['num_short_sessions'] = df['id'].apply(lambda x:num_short_sessions[x] if x in num_short_sessions else 0)

    df['num_long_sessions'] = df['id'].apply(lambda x:num_long_sessions[x] if x in num_long_sessions else 0)

    df['num_devices'] = df['id'].apply(lambda x:num_devices[x] if x in num_devices else 0)

    return df
train_df.shape
train_df = session_id_features(train_df)
train_df[train_df.average_seconds.isnull()]

train_df.average_seconds.fillna(0,inplace=True)
train_df.isnull().sum().sort_values(ascending=False)
def language(df):

    df['language'] = df['language'].apply(lambda x:'foreign' if x!='en' else x)

    return df
def affiliate_provider(df):

    df['affiliate_provider'] = df['affiliate_provider'].apply(lambda x:'rest' if x not in 

                                                              ['direct','google','other'] else x)

    return df
def browser(df):

    df['first_browser'] = df['first_browser'].apply(lambda x: "Mobile_Safari" if x=='Mobile Safari' else x)

    major_browser = ['Chrome','Safari','Firefox','IE','Mobile_Safari']

    df['first_browser'] = df['first_browser'].apply(lambda x : 'Other' if x not in major_browser else x)

    return df
def classify_device(x):

    if x.find('Desktop') != -1:

        return  "Desktop"

    elif x.find("Tablet") != -1 or x.find('iPad') != -1:

        return "Tablet"

    elif x.find('Phone') != -1:

        return  "Phone"

    else:

        return "Unknown"
train_df = language(train_df)

train_df = affiliate_provider(train_df)

train_df = browser(train_df)
train_df.shape
train_df.drop(['date_first_booking','date_account_created','timestamp_first_active'],axis=1,inplace=True)

train_df = pd.get_dummies(train_df,columns=category_columns,drop_first=True)
X_train,X_test,y_train,y_test = train_test_split(train_df.drop(['country_destination','id'],axis=1),train_df['country_destination'],test_size=0.3,random_state=42)

X_train.shape,X_test.shape,y_train.shape,y_test.shape


lg = LogisticRegression()

lg.fit(X_train,y_train)

lg.score(X_test,y_test)


lg.score(X_train,y_train)
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

gbc.fit(X_train,y_train)
gbc.score(X_train,y_train)