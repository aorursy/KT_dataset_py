!pip install git+https://github.com/goolig/dsClass.git
# imports:

import pandas as pd

from dsClass.path_helper import *
theft_data_path = get_file_path('theft_data_bgu.csv')

theft_data = pd.read_csv(theft_data_path)

holiday_data_path = get_file_path('holidays_data_bgu.csv')

holidays_data_bgu = pd.read_csv(holiday_data_path)
# check the columns:

theft_data.columns
#Q1:



# consider only important columns:

# keep only the following columns in your dataframe- Date, District, Count(our label), Year



#### insert your code here:



####



theft_data.columns
#Q2:



# change the Date from string to timestamp, in theft_data & holidays_data_bgu:

# hint - use dt.datetime.strptime

import datetime as dt



#### insert your code here:



####

# add time information:



#Q3:



# create day of week column:

# name the column Week_day, it will contain a number for each day of week: 

# Monday=0, ...., Sunday=6

#### insert your code here:



####



#Q4:

# create season column:

# name the column Season, it will contain a string which tells us the season based on the Date: 

# hint - use the following package to indicate specific dates:

from datetime import date

# and write function that gets the date and returns the corresponding season

#### insert your code here:



####
#Q5:



# add information from other sources

# holidays data

# create column is_holiday that will take 1 if its an holiday and 0 otherwise.



#### insert your code here:



####
# now you suppose to have the following columns: Date, is_holiday, district, count(our label), year, week_day, season

theft_data.columns
#Q6:

# create dummy variables:

# hint - use the pandas function 'get_dummies'

#### insert your code here:



####
# now you suppose to have the following columns: Date, is_holiday, count, year,

# dummies for season, week_day and district

theft_data.columns
#Q7:



# remove the original date columns: (we won't use it for modelling)

#### insert your code here:



####
# choose years for train and test

train_start_year = 2014

train_end_year = 2015

test_year = 2016
# split the data into train/test

dataTrain = theft_data[(theft_data["Year"] >= train_start_year) & (theft_data["Year"] <= train_end_year)]

labelsTrain = dataTrain.Count

dataTrain = dataTrain.drop('Count', axis=1)



dataTest = theft_data[(theft_data["Year"] == test_year)]

labelsTest = dataTest.Count

dataTest = dataTest.drop('Count', axis=1)



# Remove unnecessary columns:

dataTrain = dataTrain.drop('Year', axis=1)

dataTest = dataTest.drop('Year', axis=1)



print("Train data shape: " , dataTrain.shape)

print("Test data shape: " , dataTest.shape)
# check for null values (should print 0)

print(theft_data.isnull().sum().sum())
from sklearn.linear_model import LinearRegression



mlModel = LinearRegression()



mlModel.fit(dataTrain, labelsTrain)
predTest = mlModel.predict(dataTest)



# print the Rsquare:

print("Test set R^2: ", mlModel.score(dataTest, labelsTest))
#Q8:

# train the model on years 2013-2015 and test it on 2016.

# how does the model results change?

# try other combinations of train/test