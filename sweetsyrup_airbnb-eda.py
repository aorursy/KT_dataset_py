%matplotlib inline
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

%matplotlib inline


train_users = pd.read_csv("../input/train_users_2.csv")
test_users = pd.read_csv("../input/test_users.csv")
countries = pd.read_csv("../input/countries.csv")
sessions = pd.read_csv("../input/sessions.csv")
age_gender = pd.read_csv("../input/age_gender_bkts.csv")

train_users.info(); test_users.info(); countries.info(); sessions.info(); age_gender.info();
train_users['gender'].unique()

train_users["date_first_booking"] = train_users["date_first_booking"].fillna(-100)
train_users["age"] = train_users['age'].fillna(-100)
train_users["first_affiliate_tracked"] = train_users['first_affiliate_tracked'].fillna(-100)
train_users["gender"] = train_users["gender"].apply((lambda x: "-unknown-" if x == -100 else x))
## Parsing Booking date - year/month/date

def parse_booking_date(df):
    temp = df["date_first_booking"].str.split('-').apply(Series, 1).stack()
    df['Booking_year'] = temp[:,0]
    df['Booking_month'] = temp[:,1]
    df['Booking_date'] = temp[:,2]
    
def parse_create_date(df):
    temp = df['date_account_created'].str.split('-').apply(Series,1).stack()
    df['Create_year']= temp[:,0]
    df['Create_month']= temp[:,1]
    df['Create_date']= temp[:,2]
parse_booking_date(train_users)
parse_create_date(train_users)
# train_users.head()
plt.subplots(nrows=2, ncols=2, figsize=(20,10))
plt.subplot(221)
sns.countplot(y='Booking_year',hue='Booking_month',\
              data=train_users.sort(["Booking_year",'Booking_month'], ascending=[0,1]))

plt.subplot(222)
sns.countplot(x='age_binned', hue='gender', data=train_users)

plt.subplot(223)
sns.countplot(x='language', hue='affiliate_channel', data=train_users)


plt.subplot(224)
sns.countplot(x='signup_method', hue='first_device_type', data=train_users)
plt.subplots(nrows=2, ncols=1, figsize=(20,10), sharex=True)

plt.subplot(211)
sns.countplot(x='Create_year', hue='Create_month',\
              data=train_users.sort(["Create_year",'Create_month'], ascending=[0,1]))

plt.subplot(212)
sns.countplot(x='Booking_year', hue='Booking_month',\
              data=train_users.sort(['Booking_year','Booking_month'], ascending=[0,1]))
train_users['Booking_flag'] = train_users['date_first_booking'].apply(lambda x: 1 if x != -100 else 0)

plt.subplots(nrows=2, ncols=2, figsize=(20,10))

plt.subplot(221)
sns.countplot(x='gender', hue='Booking_flag', data=train_users.sort('gender', ascending=1))

plt.subplot(222)
sns.countplot(x='signup_method', hue='Booking_flag', data=train_users.sort('signup_method', ascending=1))

plt.subplot(223)
sns.countplot(x='affiliate_channel', hue='Booking_flag', data=train_users)

plt.subplot(224)
sns.countplot(x='language', hue='Booking_flag', data=train_users)
