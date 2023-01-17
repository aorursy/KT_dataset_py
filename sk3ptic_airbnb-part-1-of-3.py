# Import relevant libraraies

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import missingno as msno

%matplotlib inline
# Load the data

train = pd.read_csv("../input/airbnb-recruiting-new-user-bookings/train_users_2.csv")

train.head()
train.columns
train.info()
print("Null value % in age is: " + "{0:.2%}".format(sum(train.age.isnull())/train.shape[0]))
train.age.describe()
plt.figure(figsize=(20,6))

sns.countplot(train.age)

plt.xticks(rotation=90) 
train.loc[train.age < 18, 'age'] = np.nan

train.loc[train.age > 95, 'age'] = np.nan
train.age = train.age.replace("NaN", np.nan)
plt.figure(figsize=(20,6))

sns.countplot(train.age)

plt.xticks(rotation=90) 
plt.figure(figsize=(6,6))

temp = train.age

sns.distplot(temp.dropna())
temp.dropna().describe()
print("Now the % of null values in age is: " + "{0:.2%}".format(sum(train.age.isnull())/train.shape[0]))
print("% of people with age <= 40: " + "{0:0.2%}".format(sum(train.age <= 40)/sum(train.age.notnull()))

     + "\n% of people with age > 40: "+ "{0:0.2%}".format(sum(train.age > 40)/sum(train.age.notnull())))
print("Null value % in gender is: " + "{0:.2%}".format(sum(train.gender.isnull())/train.shape[0]))
print("Unique values in Gender:",set(train.gender))
train.gender.replace('-unknown-', np.nan, inplace=True)
print("Unique values in Gender:",set(train.gender))
print("New null value % in gender is: " + "{0:.2%}".format(sum(train.gender.isnull())/train.shape[0]))
sns.countplot(train.gender)
print(train.gender.value_counts()/sum(train.gender.notnull())*100)
print("Null value % in language is: " + "{0:.2%}".format(sum(train.language.isnull())/train.shape[0]))
print(set(train.language))
plt.figure(figsize=(20,6))

sns.countplot(train.language)
for i in set(train.language):

    print(i,": " + "{0:.2%}".format(sum(train.language == i)/sum(train.language.notnull())))
print(set(train.country_destination))
print("Null % is: " + "{0:0.2%}".format(sum(train.country_destination.isnull())/train.shape[0]))
plt.figure(figsize=(14,4))

sns.countplot(train.country_destination)
print(train.country_destination.value_counts()/sum(train.country_destination.notnull())*100)
print("Signup Method null % is: " + "{0:0.2%}".format(sum(train.signup_method.isnull())/train.shape[0]))

print("Signup Flow null % is: " + "{0:0.2%}".format(sum(train.signup_flow.isnull())/train.shape[0]))

print("Signup App null % is: " + "{0:0.2%}".format(sum(train.signup_app.isnull())/train.shape[0]))
sns.countplot(train.signup_method)
print(train.signup_method.value_counts()/sum(train.signup_method.notnull())*100)
sns.countplot(train.signup_flow)
sns.countplot(train.signup_app)
print(train.signup_app.value_counts()/sum(train.signup_app.notnull())*100)
print("Affiliate Channel null % is: " + "{0:0.2%}".format(sum(train.affiliate_channel.isnull())/train.shape[0]))

print("Affiliate Provider null % is: " + "{0:0.2%}".format(sum(train.affiliate_provider.isnull())/train.shape[0]))

print("First Affiliate Tracked null % is: " + "{0:0.2%}".format(sum(train.first_affiliate_tracked.isnull())/train.shape[0]))
print("Channel: ",set(train.affiliate_channel), "\nProvider: ", 

      set(train.affiliate_provider), "\nFirst Tracked: ",set(train.first_affiliate_tracked))
plt.figure(figsize=(14,4))

sns.countplot(train.affiliate_channel)
plt.figure(figsize=(20,6))

sns.countplot(train.affiliate_provider)

plt.xticks(rotation=45) 
plt.figure(figsize=(14,4))

sns.countplot(train.first_affiliate_tracked)
print("First Device Type null % is: " + "{0:0.2%}".format(sum(train.first_device_type.isnull())/train.shape[0]))

print("First Browser null % is: " + "{0:0.2%}".format(sum(train.first_browser.isnull())/train.shape[0]))
print("First Device Type: ",set(train.first_device_type), "\nFirst Browser: ", 

      set(train.first_browser))
train.first_browser.replace('-unknown-',np.nan,inplace=True)
plt.figure(figsize=(20,4))

sns.countplot(train.first_device_type)
print(train.first_device_type.value_counts()/sum(train.first_device_type.notnull())*100)
plt.figure(figsize=(20,4))

sns.countplot(train.first_browser)

plt.xticks(rotation=45) 
print(train.first_browser.value_counts()/sum(train.first_browser.notnull())*100)
print("Date Account Created % is: " + "{0:0.2%}".format(sum(train.date_account_created.isnull())/train.shape[0]))

print("Timestamp First Active null % is: " + "{0:0.2%}".format(sum(train.timestamp_first_active.isnull())/train.shape[0]))

print("Date First Booking % is: " + "{0:0.2%}".format(sum(train.date_first_booking.isnull())/train.shape[0]))
train.date_account_created = pd.to_datetime(train.date_account_created)

# print(train.date_account_created)
plt.figure(figsize=(80,8))

sns.countplot(train.date_account_created)

plt.xticks(rotation=90)
plt.figure(figsize=(16,4))

train.date_account_created.value_counts().plot(kind='line')
train.timestamp_first_active = pd.to_datetime(train.timestamp_first_active//1000000, format='%Y%m%d')

# print(train.timestamp_first_active)
plt.figure(figsize=(16,4))

train.timestamp_first_active.value_counts().plot(kind='line')
train.date_first_booking = pd.to_datetime(train.date_first_booking)
plt.figure(figsize=(16,4))

train.date_first_booking.value_counts().plot(kind='line')
# An overarching look at the missing data

msno.matrix(train)
# train.to_csv('train_users_3.csv',index=False)