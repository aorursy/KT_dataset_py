import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

#For date Datatype

import datetime

#Decision tree

from sklearn.tree import DecisionTreeRegressor

#Evaluation Metric

from sklearn.metrics import mean_squared_log_error

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer
# Reading in the data

TRAIN_DATA="../input/cais-x-t1-2021/train.csv"

TEST_DATA="../input/cais-x-t1-2021/test.csv"



#Creating DataFrame

df=pd.read_csv(TRAIN_DATA)

test_df=pd.read_csv(TEST_DATA)
#This shows how many rows and columns are in the dataset

#In other words, the data has 2460 records split across 12 columns

df.shape
#This shows the columns we are working with and the first few rows of the 

#dataset

df.head()
df.tail()
#Here is another way of doing the above

#This shows the data types of each column as well

df.dtypes
#change date_report column from object data types to date datatypes 

df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

test_df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
#Notice the change in type of date_report

df.dtypes
#Let's take a look at cases by visualizing the data

#create the graph and make it size 15 x 7

fig, ax = plt.subplots(figsize=(15,7))

# First grouping based on the Date 

# For each Date we further group based on the Province

# With this grouped information, we'll find the Number of Confirmed Cases for that particular grouping 

df.groupby(['Date','Province']).mean()['# Confirmed_Cases'].unstack().plot(ax=ax)

#set x-axis label

ax.set_xlabel('Date')

#set y-axis label

ax.set_ylabel('Number of Cases')
fig, ax = plt.subplots(figsize=(15,7))

# First grouping based on the Date 

# For each Date we further group based on the Province

# With this grouped information, we'll find the Number of Deaths for that particular grouping 

df.groupby(['Date','Province']).mean()['# Deaths'].unstack().plot(ax=ax)

#set x-axis label

ax.set_xlabel('Date')

#set y-axis label

ax.set_ylabel('Number of Deaths')
#To make the data a little smoother, lets visualize the data by weekly periods 

fig, ax = plt.subplots(figsize=(15,7))

df.groupby(['Date','Province']).mean()['# Confirmed_Cases'].unstack().rolling(7).mean().plot(ax=ax)

ax.set_xlabel('Date')

ax.set_ylabel('Number of Cases(weekly)')
fig, ax = plt.subplots(figsize=(15,7))

df.groupby(['Date','Province']).mean()['# Deaths'].unstack().rolling(7).mean().plot(ax=ax)

ax.set_xlabel('Date')

ax.set_ylabel('Number of Deaths(Weekly)')
df.head()
#Replace all null(NaN) values with 0

df=df.fillna(0)

test_df = test_df.fillna(0)

df.head()
#Make new column called 'Day' out of the translation between the date and day of year

df['Day'] = df['Date'].apply(lambda x: x.dayofyear)

test_df['Day'] = df['Date'].apply(lambda x: x.dayofyear)

df['Day']
#Change validation size later

split_date = datetime.datetime(year = 2020, month = 3, day = 19)

print(split_date)
# Hold every date that is before March 19th (20% of the data set)

split = df['Date'] < split_date

#Have the validation set be every date before March 19th

validation_set = df[split]

#Have the trainign set be every date from March 19th onwards(80%)

train_set = df[~split]
validation_set.tail()
train_set.tail()
df.dtypes
#Choose the columns of data that we want to train our model with

trainable_col = ['Day', 'Population', '# Tested', 'Long','Lat']

#Choose the columns for which we want to predict 

target_col = ['# Deaths','# Confirmed_Cases', '# Recovered']



#Create the model

model = DecisionTreeRegressor()

#Fit the training model with the training set

model = model.fit(train_set[trainable_col], train_set[target_col])
#Make predictions using with the validation set trained model

validation_prediction = model.predict(validation_set[trainable_col])

train_prediction = model.predict(train_set[trainable_col])

#Make predictions on the test data

test_prediction = model.predict(test_df[trainable_col])



#Should be 0

print(mean_squared_log_error(train_prediction, train_set[target_col]))

#Print the Accuracy of predictions 

mean_squared_log_error(validation_prediction, validation_set[target_col])
#This will be used to force the order of the columns

column_names = ['ForcastId','# Deaths','# Confirmed_Cases', '# Recovered']



sub_df = test_df

sub_df[['# Deaths','# Confirmed_Cases', '# Recovered']] = test_prediction

sub_df = sub_df[['# Deaths','# Confirmed_Cases', '# Recovered']]

#Make the index column called 'ForcastId'

sub_df['ForcastId'] = sub_df.index

sub_df = sub_df[column_names]
sub_df
#Turn the dataframe into a csv file that contains the predictions 

sub_df.to_csv("predictions.csv",index=False)