import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
from sklearn import linear_model
# Load the data into DataFrames
train_data = pd.read_csv('../input/train_users_2.csv', index_col='id')
test_data = pd.read_csv('../input/test_users.csv', index_col='id')
print('number of training samples ', train_data.shape[0])
train_data.head()
print('number of testing samples ',test_data.shape[0])
test_data.head()
#Convert to categorical variables
train_data.gender = train_data.gender.astype('category')
train_data.signup_method = train_data.signup_method.astype('category')
train_data.signup_flow = train_data.signup_flow.astype('category')
train_data.language = train_data.language.astype('category')
train_data.affiliate_channel = train_data.affiliate_channel.astype('category')
train_data.affiliate_provider = train_data.affiliate_provider.astype('category')
train_data.first_affiliate_tracked = train_data.first_affiliate_tracked.astype('category')
train_data.signup_app = train_data.signup_app.astype('category')
train_data.first_device_type = train_data.first_device_type.astype('category')
train_data.first_browser = train_data.first_browser.astype('category')
train_data.country_destination = train_data.country_destination.astype('category')
test_data.gender = test_data.gender.astype('category')
test_data.signup_method = test_data.signup_method.astype('category')
test_data.signup_flow = test_data.signup_flow.astype('category')
test_data.language = test_data.language.astype('category')
test_data.affiliate_channel = test_data.affiliate_channel.astype('category')
test_data.affiliate_provider = test_data.affiliate_provider.astype('category')
test_data.first_affiliate_tracked = test_data.first_affiliate_tracked.astype('category')
test_data.signup_app = test_data.signup_app.astype('category')
test_data.first_device_type = test_data.first_device_type.astype('category')
test_data.first_browser = test_data.first_browser.astype('category')
train_data.describe()

#Change time format for training
train_data.date_account_created = pd.to_datetime(train_data.date_account_created)
train_data.timestamp_first_active = pd.to_datetime(train_data.timestamp_first_active.astype('str'), format='%Y%m%d%H%M%S')
train_data.date_first_booking = pd.to_datetime(train_data.date_first_booking)
train_data.head(10)
#Change time format for testing
test_data.date_account_created = pd.to_datetime(test_data.date_account_created)
test_data.timestamp_first_active = pd.to_datetime(test_data.timestamp_first_active.astype('str'), format='%Y%m%d%H%M%S')
test_data.date_first_booking = pd.to_datetime(test_data.date_first_booking)
test_data.head(10)
train_data.describe()
#Explore and filter ages based on threshold (official minimum age is 18)
train_data.loc[train_data.age < 18, 'age'] = np.nan
train_data.loc[train_data.age > 100, 'age'] = np.nan
test_data.loc[test_data.age < 18, 'age'] = np.nan
test_data.loc[test_data.age > 100, 'age'] = np.nan
test_data.head(20)
#Explore times 
print('Training data account created dates')
print(min(train_data.date_account_created))
print(max(train_data.date_account_created))
print('Training data time first active')
print(min(train_data.timestamp_first_active))
print(max(train_data.timestamp_first_active))
print('Training data first booking dates')
print(min(train_data.date_first_booking.loc[pd.notnull(train_data.date_first_booking)]))
print(max(train_data.date_first_booking.loc[pd.notnull(train_data.date_first_booking)]))

print('Test data account created dates')
print(min(test_data.date_account_created))
print(max(test_data.date_account_created))
print('Test data time first active')
print(min(test_data.timestamp_first_active))
print(max(test_data.timestamp_first_active))

#Visualise balance
train_data.gender.value_counts().plot(kind='bar')
plt.xlabel('gender')
train_data.language.value_counts().plot(kind='bar')
plt.xlabel('language')
train_data.signup_method.value_counts().plot(kind='bar')
plt.xlabel('signup method')
train_data.signup_flow.value_counts().plot(kind='bar')
plt.xlabel('signup flow')
train_data.country_destination.value_counts().plot(kind='bar', color='r')
plt.xlabel('country destination')
train_data.age.hist(bins=20)
train_data['is_booking_made'] = np.ones((len(train_data.country_destination),1))
train_data.loc[train_data.country_destination=='NDF', 'is_booking_made'] = 0
train_data.head()
train_data_features = train_data.copy(deep=True)
train_data_features.head(10)
train_data_features = train_data_features.drop(['date_first_booking', 'country_destination', 'is_booking_made'], axis=1)

train_data_features['time_diff_to_reg']=train_data_features.date_account_created-\
train_data_features.timestamp_first_active
train_data_features = train_data_features.drop(['timestamp_first_active', 'date_account_created'], axis=1)
train_data_features.time_diff_to_reg = [x.total_seconds() for x in train_data_features.time_diff_to_reg]
train_data_features.time_diff_to_reg.head()
log_reg = linear_model.LogisticRegression()
log_reg.fit(train_data_features, train_data.is_booking_made)
