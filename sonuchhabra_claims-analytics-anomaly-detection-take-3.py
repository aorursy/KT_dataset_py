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
# importing libraries
# we already have numpy as np and pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# importing more libraries, particular to modeling
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# reading competition data files
train_org = pd.read_csv('../input/uconn_comp_2018_train.csv')
test_org = pd.read_csv('../input/uconn_comp_2018_test.csv')
# checking length of data files
len(train_org)
len(test_org)
# checking number of variables in data files
len(train_org.columns)
len(test_org.columns)
# getting to know the column names
train_org.columns
# taking a peek at some data
##~ train_org.head()
# checking missing values in train data
##~ train_org.isnull().sum()
# checking missing values in development set data
##~ test_org.isnull().sum()
train = train_org.copy(deep=True)
test = test_org.copy(deep=True)
# overview of train data
##~ train.describe()
# overview of test data
##~ test.describe()
# checking if claim_number has all unique values
len(np.unique(train.claim_number)) == len(train)
# reading the data type of claim_number
train.claim_number.dtype

# reading a sample of claim_number
##~ train.claim_number.sample(5)

# number of missing values in claim_number
train.claim_number.isnull().sum()
# description of variable
##~ train.claim_number.describe()
# reading the data type of the variable
train.age_of_driver.dtype

# reading a sample of age_of_driver
##~ train.age_of_driver.sample(5)

# number of missing values in age_of_driver
train.age_of_driver.isnull().sum()
# mean age of driver
train.age_of_driver.mean()

# median age of driver
train.age_of_driver.median()
# lets get to know the distribution of the age_of_driver
sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(24,6)
sns.distplot(train.age_of_driver)
# outliers
##~ train.age_of_driver.describe()
train.age_of_driver.max()
# modifying ages above 90 years as 90
train.age_of_driver[train.age_of_driver > 90] = 90
test.age_of_driver[test.age_of_driver > 90] = 90
# adding safty rating buckets, classes based on Standard Deviation and Quartile Ranges
train['age_of_driver_buckets'] = train.age_of_driver
train['age_of_driver_buckets'][train.age_of_driver >= 62] = 'Very High'
train['age_of_driver_buckets'][(train.age_of_driver >= 51) & (train.age_of_driver < 62)] = 'High'
train['age_of_driver_buckets'][(train.age_of_driver >= 43) & (train.age_of_driver < 51)] = 'High Average'
train['age_of_driver_buckets'][(train.age_of_driver >= 35) & (train.age_of_driver < 43)] = 'Low Average'
train['age_of_driver_buckets'][(train.age_of_driver >= 24) & (train.age_of_driver < 35)] = 'Low'
train['age_of_driver_buckets'][train.age_of_driver < 24] = 'Very Low'

test['age_of_driver_buckets'] = test.age_of_driver
test['age_of_driver_buckets'][test.age_of_driver >= 62] = 'Very High'
test['age_of_driver_buckets'][(test.age_of_driver >= 51) & (test.age_of_driver < 62)] = 'High'
test['age_of_driver_buckets'][(test.age_of_driver >= 43) & (test.age_of_driver < 51)] = 'High Average'
test['age_of_driver_buckets'][(test.age_of_driver >= 35) & (test.age_of_driver < 43)] = 'Low Average'
test['age_of_driver_buckets'][(test.age_of_driver >= 24) & (test.age_of_driver < 35)] = 'Low'
test['age_of_driver_buckets'][test.age_of_driver < 24] = 'Very Low'
# reading the data type of the variable
train.gender.dtype

# reading a sample of gender
##~ train.gender.sample(5)

# number of missing values in gender
train.gender.isnull().sum()
# distribution of gender
train.gender.value_counts()
# reading the data type of the variable
train.marital_status.dtype

# reading a sample of variable
##~ train.marital_status.sample(5)

# number of missing values for the variable
train.marital_status.isnull().sum()
# distribution of variable
train.marital_status.value_counts()
# here we observe the gender type 1 is dominant
# hence lets impute the 5 missing values with the mode, that is 1
# this is fine as the number of missing value is very low
# we will do the same treatment in the development set as well
train.marital_status.fillna(train.marital_status.value_counts().index[0], inplace=True)
test.marital_status.fillna(test.marital_status.value_counts().index[0], inplace=True)
train.safty_rating.describe()
fig, ax = plt.subplots()
fig.set_size_inches(12,6)
sns.distplot(train.safty_rating)
# adding safty rating buckets, classes based on Standard Deviation and Quartile Ranges
train['safty_rating_buckets'] = train.safty_rating
train['safty_rating_buckets'][train.safty_rating >= 105] = 'Very High'
train['safty_rating_buckets'][(train.safty_rating >= 90) & (train.safty_rating < 105)] = 'High'
train['safty_rating_buckets'][(train.safty_rating >= 76) & (train.safty_rating < 90)] = 'High Average'
train['safty_rating_buckets'][(train.safty_rating >= 65) & (train.safty_rating < 76)] = 'Low Average'
train['safty_rating_buckets'][(train.safty_rating >= 50) & (train.safty_rating < 65)] = 'Low'
train['safty_rating_buckets'][train.safty_rating < 50] = 'Very Low'

test['safty_rating_buckets'] = test.safty_rating
test['safty_rating_buckets'][test.safty_rating >= 105] = 'Very High'
test['safty_rating_buckets'][(test.safty_rating >= 90) & (test.safty_rating < 105)] = 'High'
test['safty_rating_buckets'][(test.safty_rating >= 76) & (test.safty_rating < 90)] = 'High Average'
test['safty_rating_buckets'][(test.safty_rating >= 65) & (test.safty_rating < 76)] = 'Low Average'
test['safty_rating_buckets'][(test.safty_rating >= 50) & (test.safty_rating < 65)] = 'Low'
test['safty_rating_buckets'][test.safty_rating < 50] = 'Very Low'
test['safty_rating_buckets']
# adding 0-9 interger variable ratings
train['safty_rating_int'] = np.round(train.safty_rating / 10)
test['safty_rating_int'] = np.round(test.safty_rating / 10)
# adding safty_rating transformed variable, square of log of safty_rating
train['safty_rating_logsq'] = np.square(np.log(train.safty_rating))
test['safty_rating_logsq'] = np.square(np.log(test.safty_rating))
fig, ax = plt.subplots()
fig.set_size_inches(12,6)
sns.distplot(train.safty_rating_logsq)
train.annual_income.describe()
# imputing Median value for Incomes where Income is less than 10000
train.annual_income[train.annual_income < 10000] = 37610
test.annual_income[test.annual_income < 10000] = 37610
sns.set_style('whitegrid')
fig, ax = plt.subplots()
fig.set_size_inches(12,6)
sns.distplot(train.annual_income)
# treating outliers
train.annual_income[train.annual_income > 48000] = 48000
test.annual_income[test.annual_income > 48000] = 48000
# adding income buckets, classes based on Standard Deviation and Quartile Ranges
train['annual_income_buckets'] = train.annual_income
train['annual_income_buckets'][train.annual_income >= 42500] = 'Very High'
train['annual_income_buckets'][(train.annual_income >= 39300) & (train.annual_income < 42500)] = 'High'
train['annual_income_buckets'][(train.annual_income >= 37600) & (train.annual_income < 39300)] = 'High Average'
train['annual_income_buckets'][(train.annual_income >= 35500) & (train.annual_income < 37600)] = 'Low Average'
train['annual_income_buckets'][(train.annual_income >= 32500) & (train.annual_income < 35500)] = 'Low'
train['annual_income_buckets'][train.annual_income < 32500] = 'Very Low'
train['annual_income_buckets']

test['annual_income_buckets'] = test.annual_income
test['annual_income_buckets'][test.annual_income >= 42500] = 'Very High'
test['annual_income_buckets'][(test.annual_income >= 39300) & (test.annual_income < 42500)] = 'High'
test['annual_income_buckets'][(test.annual_income >= 37600) & (test.annual_income < 39300)] = 'High Average'
test['annual_income_buckets'][(test.annual_income >= 35500) & (test.annual_income < 37600)] = 'Low Average'
test['annual_income_buckets'][(test.annual_income >= 32500) & (test.annual_income < 35500)] = 'Low'
test['annual_income_buckets'][test.annual_income < 32500] = 'Very Low'
test['annual_income_buckets']
train.high_education_ind.value_counts()
train.high_education_ind.isnull().sum()
test.high_education_ind.isnull().sum()
train.address_change_ind.value_counts()

train.address_change_ind.isnull().sum()
test.address_change_ind.isnull().sum()
train.living_status.value_counts()

train.living_status.isnull().sum()
test.living_status.isnull().sum()
train.zip_code.describe()
len(train.zip_code.value_counts())
len(test.zip_code.value_counts())
# truncating last 2 digits of zip_code
train.zip_code = np.round(train.zip_code / 100)
test.zip_code = np.round(test.zip_code / 100)
train.zip_code.value_counts()
test.zip_code.value_counts()
train.claim_date.describe()
# extracting claim_date month
train['claim_date_formatted'] = pd.to_datetime(train['claim_date'], format = "%m/%d/%Y")
train['claim_month'] = train['claim_date_formatted'].dt.month

# extracting claim_date_delta
train['claim_date_delta'] = (train['claim_date_formatted'] - train['claim_date_formatted'].min())
train['claim_date_delta'] = train['claim_date_delta'].apply(lambda x: str(x).split()[0]).astype(int)
train.drop(['claim_date', 'claim_date_formatted'], axis = 1, inplace = True)
# extracting claim_date_delta
test['claim_date_formatted'] = pd.to_datetime(test['claim_date'], format = "%m/%d/%Y")
test['claim_month'] = test['claim_date_formatted'].dt.month

# extracting claim_date_delta
test['claim_date_delta'] = (test['claim_date_formatted'] - test['claim_date_formatted'].min())
test['claim_date_delta'] = test['claim_date_delta'].apply(lambda x: str(x).split()[0]).astype(int)
test.drop(['claim_date', 'claim_date_formatted'], axis = 1, inplace = True)
train.claim_day_of_week.value_counts()

train.claim_day_of_week.isnull().sum()
test.claim_day_of_week.isnull().sum()
train.accident_site.value_counts()

train.accident_site.isnull().sum()
test.accident_site.isnull().sum()
train.past_num_of_claims.value_counts()

train.past_num_of_claims.isnull().sum()
test.past_num_of_claims.isnull().sum()
train.witness_present_ind.value_counts()

train.witness_present_ind.isnull().sum()
test.witness_present_ind.isnull().sum()
train.liab_prct.describe()
fig, ax = plt.subplots()
fig.set_size_inches(12,6)
sns.distplot(train.liab_prct)
# adding income buckets, classes based on Standard Deviation and Quartile Ranges
train['liab_prct_buckets'] = np.round(train.liab_prct / 20)
test['liab_prct_buckets'] = np.round(test.liab_prct / 20)
train.channel.value_counts()

train.channel.isnull().sum()
test.channel.isnull().sum()
train.policy_report_filed_ind.value_counts()

train.policy_report_filed_ind.isnull().sum()
test.policy_report_filed_ind.isnull().sum()
train.claim_est_payout.describe()

train.claim_est_payout.isnull().sum()
# adding income buckets, classes based on Standard Deviation and Quartile Ranges
train['claim_est_payout_buckets'] = train.claim_est_payout
train['claim_est_payout_buckets'][train.claim_est_payout >= 8500] = 'Very High'
train['claim_est_payout_buckets'][(train.claim_est_payout >= 6250) & (train.claim_est_payout < 8500)] = 'High'
train['claim_est_payout_buckets'][(train.claim_est_payout >= 4650) & (train.claim_est_payout < 6250)] = 'High Average'
train['claim_est_payout_buckets'][(train.claim_est_payout >= 3350) & (train.claim_est_payout < 4650)] = 'Low Average'
train['claim_est_payout_buckets'][(train.claim_est_payout >= 1100) & (train.claim_est_payout < 3350)] = 'Low'
train['claim_est_payout_buckets'][train.claim_est_payout < 1100] = 'Very Low'
train['claim_est_payout_buckets']

test['claim_est_payout_buckets'] = test.claim_est_payout
test['claim_est_payout_buckets'][test.claim_est_payout >= 8500] = 'Very High'
test['claim_est_payout_buckets'][(test.claim_est_payout >= 6250) & (test.claim_est_payout < 8500)] = 'High'
test['claim_est_payout_buckets'][(test.claim_est_payout >= 4650) & (test.claim_est_payout < 6250)] = 'High Average'
test['claim_est_payout_buckets'][(test.claim_est_payout >= 3350) & (test.claim_est_payout < 4650)] = 'Low Average'
test['claim_est_payout_buckets'][(test.claim_est_payout >= 1100) & (test.claim_est_payout < 3350)] = 'Low'
test['claim_est_payout_buckets'][test.claim_est_payout < 1100] = 'Very Low'
test['claim_est_payout_buckets']
train.age_of_vehicle.describe()

sum(train.age_of_vehicle.isnull())
train.age_of_vehicle[train.age_of_vehicle.isnull()] = 5
test.age_of_vehicle[test.age_of_vehicle.isnull()] = 5
fig, ax = plt.subplots()
fig.set_size_inches(12,6)
sns.distplot(train.age_of_vehicle)
# adding income buckets, classes based on Standard Deviation and Quartile Ranges
train['age_of_vehicle_buckets'] = train.age_of_vehicle
train['age_of_vehicle_buckets'][train.age_of_vehicle >= 6] = 'Very High'
train['age_of_vehicle_buckets'][(train.age_of_vehicle >= 5) & (train.age_of_vehicle < 6)] = 'High'
train['age_of_vehicle_buckets'][(train.age_of_vehicle >= 3) & (train.age_of_vehicle < 5)] = 'Low'
train['age_of_vehicle_buckets'][train.age_of_vehicle < 3] = 'Very Low'
train['age_of_vehicle_buckets']

test['age_of_vehicle_buckets'] = test.age_of_vehicle
test['age_of_vehicle_buckets'][test.age_of_vehicle >= 6] = 'Very High'
test['age_of_vehicle_buckets'][(test.age_of_vehicle >= 5) & (test.age_of_vehicle < 6)] = 'High'
test['age_of_vehicle_buckets'][(test.age_of_vehicle >= 3) & (test.age_of_vehicle < 5)] = 'Low'
test['age_of_vehicle_buckets'][test.age_of_vehicle < 3] = 'Very Low'
test['age_of_vehicle_buckets']
train.vehicle_category.value_counts()

sum(train.vehicle_category.isnull())
train.vehicle_price.describe()

sum(train.vehicle_price.isnull())
# adding income buckets, classes based on Standard Deviation and Quartile Ranges
train['vehicle_price_buckets'] = train.vehicle_price
train['vehicle_price_buckets'][train.vehicle_price >= 41500] = 'Very High'
train['vehicle_price_buckets'][(train.vehicle_price >= 29500) & (train.vehicle_price < 41500)] = 'High'
train['vehicle_price_buckets'][(train.vehicle_price >= 21000) & (train.vehicle_price < 29500)] = 'Average High'
train['vehicle_price_buckets'][(train.vehicle_price >= 14250) & (train.vehicle_price < 21000)] = 'Average Low'
train['vehicle_price_buckets'][(train.vehicle_price >= 5000) & (train.vehicle_price < 14250)] = 'Low'
train['vehicle_price_buckets'][train.vehicle_price < 5000] = 'Very Low'
train['vehicle_price_buckets']

test['vehicle_price_buckets'] = test.vehicle_price
test['vehicle_price_buckets'][test.vehicle_price >= 41500] = 'Very High'
test['vehicle_price_buckets'][(test.vehicle_price >= 29500) & (test.vehicle_price < 41500)] = 'High'
test['vehicle_price_buckets'][(test.vehicle_price >= 21000) & (test.vehicle_price < 29500)] = 'Average High'
test['vehicle_price_buckets'][(test.vehicle_price >= 14250) & (test.vehicle_price < 21000)] = 'Average Low'
test['vehicle_price_buckets'][(test.vehicle_price >= 5000) & (test.vehicle_price < 14250)] = 'Low'
test['vehicle_price_buckets'][test.vehicle_price < 5000] = 'Very Low'
test['vehicle_price_buckets']
train.vehicle_color.value_counts()

sum(train.vehicle_color.isnull())
train.vehicle_weight.describe()

sum(train.vehicle_weight.isnull())
# adding income buckets, classes based on Standard Deviation and Quartile Ranges
train['vehicle_weight_buckets'] = train.vehicle_weight
train['vehicle_weight_buckets'][train.vehicle_weight >= 41500] = 'Very High'
train['vehicle_weight_buckets'][(train.vehicle_weight >= 29500) & (train.vehicle_weight < 41500)] = 'High'
train['vehicle_weight_buckets'][(train.vehicle_weight >= 21000) & (train.vehicle_weight < 29500)] = 'Average High'
train['vehicle_weight_buckets'][(train.vehicle_weight >= 14250) & (train.vehicle_weight < 21000)] = 'Average Low'
train['vehicle_weight_buckets'][(train.vehicle_weight >= 5000) & (train.vehicle_weight < 14250)] = 'Low'
train['vehicle_weight_buckets'][train.vehicle_weight < 5000] = 'Very Low'
train['vehicle_weight_buckets']

test['vehicle_weight_buckets'] = test.vehicle_weight
test['vehicle_weight_buckets'][test.vehicle_weight >= 41500] = 'Very High'
test['vehicle_weight_buckets'][(test.vehicle_weight >= 29500) & (test.vehicle_weight < 41500)] = 'High'
test['vehicle_weight_buckets'][(test.vehicle_weight >= 21000) & (test.vehicle_weight < 29500)] = 'Average High'
test['vehicle_weight_buckets'][(test.vehicle_weight >= 14250) & (test.vehicle_weight < 21000)] = 'Average Low'
test['vehicle_weight_buckets'][(test.vehicle_weight >= 5000) & (test.vehicle_weight < 14250)] = 'Low'
test['vehicle_weight_buckets'][test.vehicle_weight < 5000] = 'Very Low'
test['vehicle_weight_buckets']
train.fraud.value_counts()

sum(train.fraud.isnull())
train = train[~(train.fraud == -1)]
train = train.apply(lambda x: x.fillna(x.value_counts().index[0]))
test = test.apply(lambda x: x.fillna(x.value_counts().index[0]))
# verifing no missing values are remaining in train and test data
train.isnull().sum().sum()
np.sum(test.isnull()).sum()
# Amount Claim per unit Weight of Vehicle
train['f1_claim_weight'] = np.log(train.claim_est_payout**2 / train.vehicle_weight)
test['f1_claim_weight'] = np.log(test.claim_est_payout**2 / test.vehicle_weight)

train['f1_claim_weight'] = np.round(train['f1_claim_weight'], 1)
test['f1_claim_weight'] = np.round(test['f1_claim_weight'], 1)

train['f1_claim_weight'].describe()
# checking f1 distribution
fig, ax = plt.subplots()
fig.set_size_inches(12,6)
sns.distplot(train['f1_claim_weight'])
# Amount Claim per unit Weight of Vehicle
train['f2_income_claims'] = np.round(train.annual_income / 1000)*10 + train.past_num_of_claims
test['f2_income_claims'] = np.round(test.annual_income / 1000)*10 + test.past_num_of_claims

train['f2_income_claims'].describe()
len(set(train['f2_income_claims']))
# verifing no missing values are remaining in train and test data
train.isnull().sum().sum()
np.sum(test.isnull()).sum()
# creating subsets of data to be used for modeling
train_total_x = train.drop('fraud', axis = 1)
train_total_y = train['fraud']
test_x = test
# verifing no missing values are remaining in train and test data
train_total_x.isnull().sum().sum()
np.sum(test_x.isnull()).sum()
# label encoder
le = LabelEncoder()
obj_columns = [col for col in train_total_x.select_dtypes(include = ['object'])]
# applying label encoder
for col in obj_columns:
    train_total_x[col] = le.fit_transform(train_total_x[col])
    test_x[col] = le.transform(test[col])
len(set(np.round(train_total_x.claim_est_payout, -2)))
train.marital_status.value_counts()
train_total_x.high_education_ind.value_counts()
# Amount Claim per unit Weight of Vehicle
train_total_x['f3_accidents_claims'] = (train_total_x.accident_site * 10) + (train_total_x.past_num_of_claims)
test_x['f3_accidents_claims'] = (test_x.accident_site * 10) + (test_x.past_num_of_claims)

train_total_x['f3_accidents_claims'].describe()
len(set(train_total_x['f3_accidents_claims']))
train_total_x['f4_liab_income'] = np.round((train_total_x.liab_prct * np.round(train_total_x.annual_income,-2)), -4)
test_x['f4_liab_income'] = np.round((test_x.liab_prct * np.round(test_x.annual_income,-2)), -4)

train_total_x['f4_liab_income'].describe()
len(set(train_total_x['f4_liab_income']))
train_total_x['f5_liab_price'] = np.round((train_total_x.liab_prct * np.round(train_total_x.vehicle_price,-2)), -4)
test_x['f5_liab_price'] = np.round((test_x.liab_prct * np.round(test_x.vehicle_price,-2)), -4)

train_total_x['f5_liab_price'].describe()
len(set(train_total_x['f5_liab_price']))
train_total_x['f6_liab_price_safty'] = np.round((train_total_x.liab_prct * np.round(train_total_x.vehicle_price,-2)), -4) / train_total_x.safty_rating
test_x['f6_liab_price_safty'] = np.round((test_x.liab_prct * np.round(test_x.vehicle_price,-2)), -4) / test_x.safty_rating

train_total_x['f6_liab_price_safty'].describe()
len(set(train_total_x['f6_liab_price_safty']))
train_total_x['f7_claim_num_payout'] = (train_total_x.claim_number**3) * train_total_x.claim_est_payout
test_x['f7_claim_num_payout'] = (test_x.claim_number**3) * test_x.claim_est_payout

train_total_x['f7_claim_num_payout'].describe()
len(set(train_total_x['f7_claim_num_payout']))
train_total_x['f8_living_claims'] = (train_total_x.living_status + 1) * (train_total_x.past_num_of_claims + 1)
test_x['f8_living_claims'] = (test_x.living_status + 1) * (test_x.past_num_of_claims + 1)

train_total_x['f8_living_claims'].describe()
len(set(train_total_x['f8_living_claims']))
train_total_x['f9_witness_payout'] = (train_total_x.witness_present_ind + 1) * (train_total_x.past_num_of_claims + 1) * np.round(train_total_x.claim_est_payout, -2)
test_x['f9_witness_payout'] = (test_x.witness_present_ind + 1) * (test_x.past_num_of_claims + 1) * np.round(test_x.claim_est_payout, -2)

train_total_x['f9_witness_payout'].describe()
len(set(train_total_x['f9_witness_payout']))
train_total_x['f10_interaction_f2_f3'] = np.sqrt((train_total_x.f2_income_claims + 1) * (train_total_x.f3_accidents_claims + 1))
test_x['f10_interaction_f2_f3'] = np.sqrt((test_x.f2_income_claims + 1) * (test_x.f3_accidents_claims + 1))

train_total_x['f10_interaction_f2_f3'].describe()
len(set(train_total_x['f10_interaction_f2_f3']))
train_total_x['f11_education_claim'] = (train_total_x.claim_est_payout) / (train_total_x.high_education_ind + 1)
test_x['f11_education_claim'] = (test_x.claim_est_payout) / (test_x.high_education_ind + 1)

train_total_x['f11_education_claim'].describe()
len(set(train_total_x['f11_education_claim']))
train_total_x['f12_witness_f5'] = (train_total_x.witness_present_ind + 1) * np.log(train_total_x.f5_liab_price + 1)
test_x['f12_witness_f5'] = (test_x.witness_present_ind + 1) * np.log(test_x.f5_liab_price + 1)

train_total_x['f12_witness_f5'].describe()
len(set(train_total_x['f12_witness_f5']))
train_total_x['f13_matital_f2'] = np.round((train_total_x.marital_status + 1) * np.sqrt(train_total_x.f2_income_claims),1)
test_x['f13_matital_f2'] = np.round((test_x.marital_status + 1) * np.sqrt(test_x.f2_income_claims),1)

train_total_x['f12_witness_f5'].describe()
len(set(train_total_x['f12_witness_f5']))

# removing features not important - Sawyer Model
remove_cols = ['f1_claim_weight', 'f7_claim_num_payout', 'claim_day_of_week', 'claim_est_payout_buckets',
               'f9_witness_payout', 'vehicle_price_buckets', 'f4_liab_income', 'witness_present_ind', 'age_of_driver', 'policy_report_filed_ind',
              'marital_status', 'living_status', 'accident_site', 'channel', 'f5_liab_price', 'age_of_driver_buckets', 'vehicle_category',
              'vehicle_weight_buckets', 'age_of_vehicle_buckets', 'liab_prct_buckets', 'vehicle_color', 'claim_month', 'annual_income_buckets',
              'safty_rating_logsq', 'safty_rating_int', 'safty_rating_buckets']

train_total_x = train_total_x.drop(remove_cols, axis = 1)
test_x = test_x.drop(remove_cols, axis = 1)
# columns list
final_cols = train_total_x.columns
# Normalizing Data
##~ norm = Normalizer()
##~ train_total_x = norm.fit_transform(train_total_x)
##~ test_x = norm.transform(test_x)
# MinMax Scaling
scaler = MinMaxScaler()
train_total_x = scaler.fit_transform(train_total_x)
test_x = scaler.transform(test_x)
train_total_x = pd.DataFrame(train_total_x)
test_x = pd.DataFrame(test_x)
train_total_x.head()
np.sum(np.sum(train_total_x.isnull()))
np.sum(np.sum(test_x.isnull()))
# train-development split
train_x, devl_x, train_y, devl_y = train_test_split(train_total_x, train_total_y, test_size = 0.35, random_state = 3)
len(train_x)
len(train_y)
##! len(devl_x)
##! len(devl_y)
# number of columns
len(train_x.columns)
##! len(devl_x.columns)
len(test_x.columns)
train_y.sum()
train_y.sum() / len(train_y)

##! devl_y.sum()
##! devl_y.sum() / len(devl_y)
## Sawyer

from xgboost import XGBClassifier
param_test2 = {'learning_rate':[0.1], 'reg_alpha':[3,30,70,150,200]}
gsearch2 = GridSearchCV(estimator = XGBClassifier(n_estimators=1000,gamma=4,max_depth=2,min_child_weight=5),
                        param_grid = param_test2, scoring='roc_auc',cv=10)
gsearch2.fit(train_x, train_y)
print('Accurary of Xgboost Classifier on train_x: {:.3f}' .format(gsearch2.score(train_x, train_y)))
print('Accurary of Xgboost Classifier on devl_x: {:.3f}' .format(gsearch2.score(devl_x, devl_y)))

print('Grid best parameter (max. accuary): ', gsearch2.best_params_)
print('Grid best score (accuary):', gsearch2.best_score_)
xgb2_train = gsearch2.predict_proba(train_x)[:,1]
xgb2_devl = gsearch2.predict_proba(devl_x)[:,1]
xgb2_predictions = gsearch2.predict_proba(test_x)[:,1]
sum(np.round(xgb2_train))
sum(np.round(xgb2_devl))
sum(np.round(xgb2_predictions))
imp_feat = pd.DataFrame(gsearch2.best_estimator_.feature_importances_, final_cols)
imp_feat.sort_values(by=[0], ascending=False)
train_predictions = xgb2_train
devl_predictions = xgb2_devl
test_predictions = xgb2_predictions
print('Accurary of Final Classifier on train_x: {:.3f}' .format(roc_auc_score(train_y, train_predictions)))
print('Accurary of Final Classifier on devl_x: {:.3f}' .format(roc_auc_score(devl_y, devl_predictions)))
my_submission = pd.DataFrame({'claim_number': test_org.claim_number, 'fraud': test_predictions})
my_submission.to_csv('submission.csv', index=False)
my_submission.head()