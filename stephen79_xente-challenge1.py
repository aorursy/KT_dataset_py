# load modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import pickle



from pandas.plotting import scatter_matrix



# stop warnings

import warnings 

warnings.filterwarnings('ignore')
# display column limita

pd.set_option('display.max_columns',500)
# load data

train = pd.read_csv('../input/xente-challenge/training.csv')

validation = pd.read_csv('../input/xente-challenge/test.csv')

train.head()
# checking the balance of the data

print('The number of Non-Frauds are: ' + str(train['FraudResult'].value_counts()[0]) + ' which is', round(train['FraudResult'].value_counts()[0]/len(train) * 100,2), '% of the dataset')

print('The number of Frauds are: ' + str(train['FraudResult'].value_counts()[1]) + ' which is', round(train['FraudResult'].value_counts()[1]/len(train) * 100,2), '% of the dataset')
# visualize category class

sns.countplot(x='FraudResult', data=train)
# SMOTE

# oversampling

from imblearn.over_sampling import SMOTE



count_class_0, count_class_1 = train.FraudResult.value_counts()



# divide by class

train_class_0 = train[train['FraudResult'] == 0]

train_class_1 = train[train['FraudResult'] == 1]
train_class_1_over = train_class_1.sample(count_class_0, replace=True)

train_test_over = pd.concat([train_class_0, train_class_1_over], axis=0)



print('Random over-sampling:')

print(train_test_over.FraudResult.value_counts())



train_test_over.FraudResult.value_counts().plot(kind='bar', title='Count (FraudResult)');
train1 = train_test_over 
numeric_features = train.select_dtypes(include=[np.number])

numeric_features.columns
categorical_features = train.select_dtypes(include=[np.object])

categorical_features.columns
# pricing and fraudresults

sns.countplot(y='ProviderId', data=train1, hue='FraudResult')

plt.show
# pricingstrategy and fraudresult

sns.countplot(x='PricingStrategy', data=train1, hue='FraudResult')

plt.show()
# product category and fraudresult

sns.countplot(y='ProductCategory',data = train1, hue = 'FraudResult')
# ProductId and fraudresult

sns.countplot(y='ProductId', data = train1, hue = 'FraudResult')
# channelid and fraudresult

sns.countplot(x='ChannelId', data = train1, hue = 'FraudResult')
# TIME WRANGLING

# train1

train1['hour'] = pd.to_datetime(train1.TransactionStartTime).dt.hour

train1['minute'] = pd.to_datetime(train1.TransactionStartTime).dt.minute

train1['day'] = pd.to_datetime(train1.TransactionStartTime).dt.dayofweek



# validation

validation['hour'] = pd.to_datetime(validation.TransactionStartTime).dt.hour

validation['minute'] = pd.to_datetime(validation.TransactionStartTime).dt.minute

validation['day'] = pd.to_datetime(validation.TransactionStartTime).dt.dayofweek

train1['period'] = np.nan

validation['period'] = np.nan
# train1

train1.loc[train1.hour < 7, 'period']= 'em'

train1.loc[(train1.hour >= 7) & (train1.hour < 11), 'period'] = 'am'

train1.loc[(train1.hour >= 11) & (train1.hour < 15), 'period'] = 'mid'

train1.loc[(train1.hour >= 15) & (train1.hour < 19), 'period'] = 'eve'

train1.loc[(train1.hour >= 19) & (train1.hour <=24), 'period'] = 'pm'



# validation

validation.loc[validation.hour < 7, 'period']= 'em'

validation.loc[(validation.hour >= 7) & (validation.hour < 11), 'period'] = 'am'

validation.loc[(validation.hour >= 11) & (validation.hour < 15), 'period'] = 'mid'

validation.loc[(validation.hour >= 15) & (validation.hour < 19), 'period'] = 'eve'

validation.loc[(validation.hour >= 19) & (validation.hour <=24), 'period'] = 'pm'
train1['minutes'] = train1['hour']*60 + train1['minute'] + train1['day']*24*60

validation['minutes'] = validation['hour']*60 + validation['minute'] + validation['day']*24*60

train1.head()
# drop features

train1 = train1.drop(['BatchId','AccountId','SubscriptionId','CustomerId','CurrencyCode','CountryCode','Amount','TransactionStartTime','hour','minute','day'], axis=1)

validation = validation.drop(['BatchId','AccountId','SubscriptionId','CustomerId','CurrencyCode','CountryCode','Amount','TransactionStartTime','hour','minute','day'], axis=1)
# normalize

from sklearn.preprocessing import MinMaxScaler

# minutes

scaler_minutes = MinMaxScaler()

train1['minutes'] = train1['minutes'].astype('float64')

train1['minutes'] = scaler_minutes.fit_transform(train1.minutes.values.reshape(-1,1))



validation['minutes'] = scaler_minutes.fit_transform(validation.minutes.values.reshape(-1,1))

# value

scaler_Value = MinMaxScaler()

train1['Value'] = scaler_Value.fit_transform(train1.Value.values.reshape(-1,1))



validation['Value'] = scaler_Value.fit_transform(validation.Value.values.reshape(-1,1))
validation1 = validation.copy()
# drop Transactionid

train1 = train1.drop(['TransactionId'], axis=1)

validation = validation.drop(['TransactionId'], axis=1)

train1.head()
# dummies

train1 = pd.get_dummies(train1, prefix_sep='_', drop_first=True)

validation = pd.get_dummies(validation, prefix_sep='_', drop_first=True)
# drop irrelevant features

train1 = train1.drop(['ProviderId_ProviderId_2','ProductId_ProductId_10','ProductId_ProductId_11','ProductId_ProductId_12','ProductId_ProductId_15','ProductId_ProductId_16','ProductId_ProductId_19','ProductId_ProductId_2','ProductId_ProductId_22','ProductId_ProductId_23','ProductId_ProductId_4','ProductId_ProductId_5','ProductId_ProductId_7','ProductId_ProductId_9','ProductCategory_data_bundles','ProductCategory_movies','ProductCategory_other','ProductCategory_ticket','ProductCategory_tv','ChannelId_ChannelId_5'], axis=1)

# drop irrelevant features

validation = validation.drop(['ProviderId_ProviderId_2','ProductId_ProductId_10','ProductId_ProductId_11','ProductId_ProductId_15','ProductId_ProductId_16','ProductId_ProductId_18','ProductId_ProductId_17','ProductId_ProductId_19','ProductId_ProductId_2','ProductId_ProductId_22','ProductId_ProductId_23','ProductId_ProductId_25','ProductId_ProductId_26','ProductId_ProductId_4','ProductId_ProductId_5','ProductId_ProductId_7','ProductId_ProductId_9','ProductCategory_data_bundles','ProductCategory_movies','ProductCategory_retail','ProductCategory_ticket','ProductCategory_tv','ChannelId_ChannelId_4','ChannelId_ChannelId_5'], axis=1)
# bring the fraudresult column to be 1st

FraudResult = train1['FraudResult']

train1.drop(['FraudResult'], axis=1, inplace=True)

train1.insert(0,'FraudResult', FraudResult)
# rename columns

train1.rename(columns={'ProviderId_ProviderId_3':'ProviderId3',

                       'ProviderId_ProviderId_4':'ProviderId4',

                       'ProviderId_ProviderId_5':'ProviderId5',

                       'ProviderId_ProviderId_6':'ProviderId6',

                       'ProductId_ProductId_13' :'ProductId13',

                       'ProductId_ProductId_14' :'ProductId14',

                       'ProductId_ProductId_20' :'ProductId20',

                       'ProductId_ProductId_21' :'ProductId21',

                       'ProductId_ProductId_24' :'ProductId24',

                       'ProductId_ProductId_27' :'ProductId27',

                       'ProductId_ProductId_3' :'ProductId3',

                       'ProductId_ProductId_6' :'ProductId6',

                       'ProductId_ProductId_8' :'ProductId8',

                       'ProductCategory_financial_services':'financial_services',

                       'ProductCategory_transport':'transport',

                       'ProductCategory_utility_bill':'utility_bill',

                       'ChannelId_ChannelId_2':'ChannelId2',

                       'ChannelId_ChannelId_3':'ChannelId3',

                       'period_em':'em',

                       'period_eve':'eve',

                       'period_mid':'mid',

                       'period_pm':'pm'}, inplace=True)



validation.rename(columns={'ProviderId_ProviderId_3':'ProviderId3',

                       'ProviderId_ProviderId_4':'ProviderId4',

                       'ProviderId_ProviderId_5':'ProviderId5',

                       'ProviderId_ProviderId_6':'ProviderId6',

                       'ProductId_ProductId_13' :'ProductId13',

                       'ProductId_ProductId_14' :'ProductId14',

                       'ProductId_ProductId_20' :'ProductId20',

                       'ProductId_ProductId_21' :'ProductId21',

                       'ProductId_ProductId_24' :'ProductId24',

                       'ProductId_ProductId_27' :'ProductId27',

                       'ProductId_ProductId_3' :'ProductId3',

                       'ProductId_ProductId_6' :'ProductId6',

                       'ProductId_ProductId_8' :'ProductId8',

                       'ProductCategory_financial_services':'financial_services',

                       'ProductCategory_transport':'transport',

                       'ProductCategory_utility_bill':'utility_bill',

                       'ChannelId_ChannelId_2':'ChannelId2',

                       'ChannelId_ChannelId_3':'ChannelId3',

                       'period_em':'em',

                       'period_eve':'eve',

                       'period_mid':'mid',

                       'period_pm':'pm'}, inplace=True)
train1.head()
train1['exponential'] = np.log(train1['Value']**2 + train1['PricingStrategy']**2 + train1['ProviderId3']**2 + train1['ProviderId4']**2 + train1['ProviderId6']**2 + train1['ProductId13']**2 +train1['ProductId14']**2 + train1['ProductId20']**2 + train1['ProductId21']**2 + train1['ProductId24']**2 + train1['ProductId27']**2 + train1['ProductId3']**2 + train1['ProductId6']**2 + train1['ProductId8']**2 + train1['financial_services']**2 + train1['transport']**2 + train1['utility_bill']**2 + train1['ChannelId2']**2 + train1['ChannelId3']**2)

train1['matrix'] = np.log(train1['ProviderId3']**2 + train1['ProviderId4']**2 + train1['ProviderId6']**2 + train1['Value']**2 + train1['ChannelId2']**2 + train1['ChannelId3']**2)

 

validation['exponential'] = np.log(validation['Value']**2 + validation['PricingStrategy']**2 + validation['ProviderId3']**2 + validation['ProviderId4']**2 + validation['ProviderId6']**2 + validation['ProductId13']**2 +validation['ProductId14']**2 + validation['ProductId20']**2 + validation['ProductId21']**2 + validation['ProductId24']**2 + validation['ProductId27']**2 + validation['ProductId3']**2 + validation['ProductId6']**2 + validation['ProductId8']**2 + validation['financial_services']**2 + validation['transport']**2 + validation['utility_bill']**2 + validation['ChannelId2']**2 + validation['ChannelId3']**2)

validation['matrix'] = np.log(validation['ProviderId3']**2 + validation['ProviderId4']**2 + validation['ProviderId6']**2 + validation['Value']**2 + validation['ChannelId2']**2 + validation['ChannelId3']**2)

     
train1.head()
scaler_exponential = MinMaxScaler()

train1['exponential'] = train1['exponential'].astype('float64')

train1['exponential'] = scaler_exponential.fit_transform(train1.exponential.values.reshape(-1,1))
# selection of features

y = train1.FraudResult

X = train1.drop(['FraudResult'], axis=1)
# split data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# random forest

from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier()

RFC = RFC.fit(X_train,y_train)

y_pred = RFC.predict(X_test)



# bring the test dataset

# random forest

from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier()

RFC = RFC.fit(X_train,y_train)



submit = RFC.predict(validation)
submission = pd.DataFrame({'TransactionId':validation1['TransactionId'],'FraudResult':submit})
submission.to_csv('submit70.csv', index=False)
# save the model to disk

filename = 'XenteFraud_detection_model_7.sav'

pickle.dump(RFC, open(filename, 'wb'))