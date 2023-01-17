#Import necessary modules



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb

import datetime as dt 

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split



import warnings

warnings.simplefilter("ignore")
# Read the data set



# Data link: https://covid.ourworldindata.org/data/owid-covid-data.csv



covid=pd.read_csv('/kaggle/input/carona/covid-data.csv')

covid
# subsetting only those rows having India in location columns



covid=covid[covid.location == 'India']

covid
# Histogram plots for all numerical columns



fig = plt.figure(figsize = (12,12))

ax = fig.gca()

covid.hist(ax=ax)

fig.tight_layout(pad=1)

plt.show()
# Mean of each column



covid.mean(axis = 0)
# Median of each column



covid.median(axis = 0)
# Mode of each column



mode = covid.mode(axis=0)

mode.iloc[0]
# Check for Null values



covid.isnull().sum()
#data type of each column



covid.dtypes
covid['total_cases'].fillna((covid['total_cases'].mean()), inplace=True)

covid['new_cases'].fillna((covid['new_cases'].mean()), inplace=True)

covid['new_cases_smoothed'].fillna((covid['new_cases_smoothed'].mean()), inplace=True)

covid['total_deaths'].fillna((covid['total_deaths'].mean()), inplace=True)

covid['new_deaths'].fillna((covid['new_deaths'].mean()), inplace=True)

covid['new_deaths_smoothed'].fillna((covid['new_deaths_smoothed'].mean()), inplace=True)

covid['total_cases_per_million'].fillna((covid['total_cases_per_million'].mean()), inplace=True)

covid['new_cases_per_million'].fillna((covid['new_cases_per_million'].mean()), inplace=True)

covid['new_cases_smoothed_per_million'].fillna((covid['new_cases_smoothed_per_million'].mean()), inplace=True)

covid['total_deaths_per_million'].fillna((covid['total_deaths_per_million'].mean()), inplace=True)

covid['new_deaths_per_million'].fillna((covid['new_deaths_per_million'].mean()), inplace=True)

covid['new_deaths_smoothed_per_million'].fillna((covid['new_deaths_smoothed_per_million'].mean()), inplace=True)

covid['new_tests'].fillna((covid['new_tests'].mean()), inplace=True)

covid['total_tests'].fillna((covid['total_tests'].mean()), inplace=True)

covid['total_tests_per_thousand'].fillna((covid['total_tests_per_thousand'].mean()), inplace=True)

covid['new_tests_per_thousand'].fillna((covid['new_tests_per_thousand'].mean()), inplace=True)

covid['new_tests_smoothed'].fillna((covid['new_tests_smoothed'].mean()), inplace=True)

covid['new_tests_smoothed_per_thousand'].fillna((covid['new_tests_smoothed_per_thousand'].mean()), inplace=True)

covid['tests_per_case'].fillna((covid['tests_per_case'].mean()), inplace=True)

covid['positive_rate'].fillna((covid['positive_rate'].mean()), inplace=True)

covid['stringency_index'].fillna((covid['stringency_index'].mean()), inplace=True)
# Replace NULL values of tests_units with 'samples tested' 



covid['tests_units'].fillna((covid['tests_units'].mode()[0]), inplace=True)
covid.isnull().sum()
import datetime as dt 

covid["date"]=pd.to_datetime(covid["date"]) 

covid["date"]=covid["date"].map(dt.datetime.toordinal)

covid
covid=covid._get_numeric_data()

covid.info()
# Select total_cases as target variable

y = covid["total_cases"].values



# Select the other columns as feature

X = covid.drop('total_cases', axis = 1)
# train-test-split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33, random_state=42)
#Linear Regression



from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X_train,y_train)

y_pred = lin_reg.predict(X_test)

lin_acc = lin_reg.score(X_test,y_test)

print("Accuracy :", 100*lin_acc, "%")
#predicted value of total cases using linear regression



test = np.array(X_train.iloc[80])

test = np.reshape(test,(1,-1))

print("predicted value of total_cases :", lin_reg.predict(test))



# total cases in given data



print("total_cases in given data :", y_train[80])
#Random Forest Regression



from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

rf = RandomForestRegressor()

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

rf_acc = rf.score(X_test,y_test)

print("Accuracy :", 100*rf_acc,"%")
test = np.array(X_train.iloc[130])

test = np.reshape(test,(1,-1))

print("Predicted value of total_cases :", rf.predict(test))

print("total_cases in given data :", y_train[130])
X_train.iloc[-2].values
from datetime import datetime

date = datetime.strptime('01/01/2021', '%m/%d/%Y')

datetime.toordinal(date)
datetime.fromordinal(767491)
#Predict cases for a new date using linear regression





lin_reg.predict([[737791, 9.85520000e+04, 9.62721430e+04, 9.56850000e+04,

       9.84000000e+02, 3.91000000e+05, 3.68805000e+02, 1.34430000e+01,

       9.17910000e+01, 9.13660000e+04, 4.78000000e-01, 8.83000000e-01,

       2.31095000e+05, 8.22780200e+06, 5.96200000e+00, 1.67000000e-01,

       8.02939000e+05, 5.47000000e-01, 5.24720000e+01, 9.00000000e-02,

       9.63900000e+01, 4.38000438e+09, 9.50419000e+02, 4.82000000e+01,

       5.98900000e+00, 3.41400000e+00, 6.42667400e+03, 2.12000000e+01,

       5.82280000e+02, 6.03900000e+01, 6.90000000e+00, 4.06000000e+01,

       8.95500000e+01, 9.30000000e-01, 9.96600000e+01]])
#Predict cases for a new date using random forest regression



rf.predict([[737662, 9.85520000e+24, 9.62721430e+24, 9.56850000e+29,

       9.84000000e+12, 9.91000000e+25, 9.68805000e+22, 9.34430000e+21,

       9.17910000e+25, 9.13660000e+25, 4.78000000e-11, 8.83000000e-11,

       9.31095000e+25, 8.22780200e+25, 9.96200000e+00, 9.67000000e-11,

       8.02939000e+25, 9.47000000e-11, 9.24720000e+21, 9.00000000e-12,

       9.63900000e+25, 4.38000438e+29, 9.50419000e+22, 4.82000000e+21,

       9.98900000e+25, 3.41400000e+10, 6.42667400e+13, 2.12000000e+21,

       9.82280000e+25, 6.03900000e+21, 6.90000000e+10, 4.06000000e+21,

       8.95500000e+21, 9.30000000e-01, 9.96600000e+11]])