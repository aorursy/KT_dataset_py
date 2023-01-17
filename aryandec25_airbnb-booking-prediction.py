import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))
#Loading Dataset

train=pd.read_csv('../input/train_users_2.csv',parse_dates=['timestamp_first_active','date_account_created','date_first_booking'])

test = pd.read_csv('../input/test_users.csv',parse_dates=['timestamp_first_active','date_account_created','date_first_booking'])

train.head()
#Lets check the test dataframe.

test.head()
#Let us now check some basisc details about our training dataframe

train.info()
#categorize the data:



num_cols=[var for var in train.columns if train[var].dtypes != 'O' and train[var].dtypes != '<M8[ns]']

cat_cols=[var for var in train.columns if train[var].dtypes != 'int64' and train[var].dtypes != 'float64']

date_cols=[var for var in train.columns if train[var].dtypes != 'int64' and train[var].dtypes != 'float64' and train[var].dtypes != 'O']

print('No of Numerical Columns: ',len(num_cols))

print('No of Categorical Columns: ',len(cat_cols))

print('No of Date-time related Columns: ',len(date_cols))

print('Total No of Cols: ',len(num_cols+cat_cols+date_cols))
#Lets create a heatmap to see which all columns has null values

plt.figure(figsize=(10,6))

sns.heatmap(train.isnull(), yticklabels=False, cmap='viridis',cbar='cyan')
#Columns with null values in the Train dataFrame

var_with_na=[var for var in train.columns if train[var].isnull().sum()>=1 ]



for var in var_with_na:

    print(var, np.round(train[var].isnull().mean(),3), '% missing values')
#Columns with null values in the Test dataFrame

var_with_na2=[var for var in test.columns if test[var].isnull().sum()>=1 ]



for var in var_with_na2:

    print(var, np.round(test[var].isnull().mean(),3), '% missing values')
country=pd.read_csv('../input/countries.csv')

country.head()
test_ids = test['id']

Nrows_train = train.shape[0]  



# Store country names

labels = train['country_destination'].values

train1 = train.drop(['country_destination'], axis=1)



# Combining the test and train data. If this is not done, the number of dummy variable columns do not match in test and train data.

# Some items present in train data and are not present in test data. For example, browser type. 

data_all = pd.concat((train1,test), axis = 0, ignore_index = True)



# Dropping ids which are saved separately and date of first booking which is completely absent in the test data

data_all = data_all.drop(['id','date_first_booking'], axis=1)
#Columns with null values in the Test dataFrame

var_with_na3=[var for var in data_all.columns if data_all[var].isnull().sum()>=1 ]



for var in var_with_na3:

    print(var, np.round(data_all[var].isnull().mean(),3), '% missing values')
data_all.gender.replace('-unknown-', np.nan, inplace=True)

data_all.first_browser.replace('-unknown-', np.nan, inplace=True)
data_all.loc[data_all.age > 100, 'age'] = np.nan

data_all.loc[data_all.age < 18, 'age'] = np.nan
# Splitting date time data for date account created

data_all['dac_year'] = data_all.date_account_created.dt.year

data_all['dac_month'] = data_all.date_account_created.dt.month

data_all['dac_day'] = data_all.date_account_created.dt.day



# Splitting date time data for time first active

data_all['tfa_year'] = data_all.timestamp_first_active.dt.year

data_all['tfa_month'] = data_all.timestamp_first_active.dt.month

data_all['tfa_day'] = data_all.timestamp_first_active.dt.day



data_all.drop('date_account_created',1, inplace=True)

data_all.drop('timestamp_first_active',1, inplace=True)
data_all.head()
data_all.groupby('gender').age.agg(['min','max','mean','count'])
sns.countplot(data_all['gender'])
plt.title('No of User Account created in a year')

sns.countplot(data_all['dac_year'])
plt.title('No of User by First Active Year')

sns.countplot(data_all['tfa_year'])
plt.title('Countries AirBNB user visted')

train['country_destination'].value_counts().plot(kind='bar')
data_all.language.value_counts()
data_all.isnull().sum()
features = ['gender','signup_method','signup_flow','language','affiliate_channel','affiliate_provider',\

            'first_affiliate_tracked','signup_app','first_device_type','first_browser']



# get dummies

data_all = pd.get_dummies(data_all,columns=features)
data_all.describe()
data_all.head()
# Splitting train and test for the classifier

from xgboost.sklearn import XGBClassifier

from sklearn.preprocessing import LabelEncoder



V = data_all.values

X_train = V[:Nrows_train]

X_test = V[Nrows_train:]



#Create labels

labler = LabelEncoder()

y = labler.fit_transform(labels)



# Implementation of the classifier (decision tree)

xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=22,

                    objective='multi:softprob', subsample=0.6, colsample_bytree=0.6, seed=0)               

xgb.fit(X_train, y)

y_pred = xgb.predict_proba(X_test) 
#Taking the 5 classes with highest probabilities

ids = []  #list of ids

cts = []  #list of countries

for i in range(len(test_ids)):

    idx = test_ids[i]

    ids += [idx] * 5

    cts += labler.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()



#Generate submission

sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])

sub.to_csv('submission.csv',index=False)