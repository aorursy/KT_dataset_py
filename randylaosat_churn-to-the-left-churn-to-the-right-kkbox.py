# Import the neccessary modules for data manipulation and visual representation

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as matplot

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/train.csv')

members = pd.read_csv('../input/members.csv')

transactions = pd.read_csv('../input/transactions.csv')

#sample_submission_zero= pd.read_csv('../input/sample_submission_zero.csv')

#user_logs = pd.read_csv('../input/user_logs.csv',nrows = 2e7)
train.head()
# The dataset contains 2 columns and 992931 observations

train.shape
# Check to see if the train set has any missing values. No missing values!

train.isnull().any()
# Looks like about 93.6% of customers stayed and 6.4% of customers left. 

# NOTE: When performing cross validation, its important to maintain this turnover ratio

churn_rate = train.is_churn.value_counts() / len(train)

churn_rate
members.tail()
# The dataset contains 2 columns and 992931 observations

members.shape
# Check to see if the train set has any missing values.

members.isnull().any()
# Quick Overview of the members dataframe

members.describe()
members.city.describe()
# Display the unique values in the city variable

# It has 21 unique city values and the #2 is missing

members.city.unique()
# Display the unique values in the bd variable

# It contains many outliers and random numbers. Maybe this variable shouldn't be used

members.bd.unique()
# Display the distrubtion of gender variable

members.gender.value_counts() / len(members)
members.registered_via.unique()
transactions.head()
# This data frame 

transactions.shape
# Check to see if the transaction set has any missing values.

transactions.isnull().any()
# Convert is_churn into a categorical variable

train["is_churn"] = train["is_churn"].astype('category')



# Convert these features from members dataset into categorical variables

members["city"] = members["city"].astype('category')

members["gender"] = members["gender"].astype('category')

members["registered_via"] = members["registered_via"].astype('category')

members["registration_init_time"] = members["registration_init_time"].astype('category')

members["expiration_date"] = members["expiration_date"].astype('category')
training = pd.merge(left = train,right = members,how = 'left',on=['msno'])

training.head()
training.dtypes
training['city'].fillna(method='ffill', inplace=True)

training['bd'].fillna(method='ffill', inplace=True)



training['gender'].fillna(method='ffill', inplace=True)



training['registered_via'].fillna(method='ffill', inplace=True)

training.isnull().any()
# Set up the matplotlib figure

f, axes = plt.subplots(ncols=3, figsize=(20, 6))



# Graph User City Distribution

# sns.distplot(training.city, kde=False, color="g",  ax=axes[0]).set_title('User City Distribution')

data = training.groupby('city').aggregate({'msno':'count'}).reset_index()

sns.barplot(x='city', y='msno', data=data, ax=axes[0]).set_title('User City Distribution')



# Graph User Gender Distrubtion

##sns.barplot(x="gender", data=training, ax=axes[1]).set_title('User Register_Via Distribution')

sns.countplot(y="gender", data=training, color="c",  ax=axes[1]).set_title('User Gender Distribution')



# Graph User Churn Distribution

sns.distplot(training.is_churn, kde=False, color="b", bins = 3,  ax=axes[2]).set_title('User Churn Distribution')
sns.countplot(y="registered_via", data=training, color="c").set_title('Registration Type Distribution')