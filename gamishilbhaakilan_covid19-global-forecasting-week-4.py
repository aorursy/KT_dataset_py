# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import confusion_matrix

import datetime

from datetime import datetime



# Importing the training data set

dataset = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

x = dataset.iloc[:, :-2]

display(x.head(5))

y1 = dataset.iloc[:, -2]

y2 = dataset.iloc[:, -1]

display(dataset.describe())

# Importing the Latitude and Longitude dataset

latlong_dataset = pd.read_csv('../input/latitude-and-longitude-for-every-country-and-state/world_country_and_usa_states_latitude_and_longitude_values.csv')



# Latitude and longitude for all countries except USA

a = latlong_dataset.iloc[:, 1:4]



# Latitude and longitude only for USA

b = latlong_dataset.iloc[:, 5:8]

# EDA 



# Path of the file to read

dataset_filepath = "../input/covid19-global-forecasting-week-4/train.csv"



# Read the file into a variable CV_data

CV_data = pd.read_csv(dataset_filepath, index_col="Date", parse_dates=True)



display(CV_data.head(5))

# Data Visualization

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(14,6))

plt.title("Confirmed Cases and Fatalities")

plt.xlabel("Date")

sns.lineplot(data=CV_data['ConfirmedCases'], label="ConfirmedCases")

sns.lineplot(data=CV_data['Fatalities'], label="Fatalities")

# Merging the Latitude and Longitude along with training dataset



# Filling the missing values on Province_State by 0

x['Province_State'] = x['Province_State'].fillna(0)



# Combining the longitude and latitude for all countries (except USA states) with the training dataset

x = pd.merge(x,a,left_on = 'Country_Region', right_on ='country', how ='left')



# Dropping the additional 'country' column

x.drop(['country'], axis=1, inplace=True)



# Combining the longitude and latitude of USA states with the training dataset

x = pd.merge(x,b,left_on = 'Province_State', right_on ='usa_state', how ='left')



# Dropping the additional 'usa_state' column

x.drop(['usa_state'], axis=1, inplace=True)



# Filling the missing values 

x['latitude'] = x['latitude'].fillna(0)

x['longitude'] = x['longitude'].fillna(0)

x['usa_state_latitude'] = x['usa_state_latitude'].fillna(0)

x['usa_state_longitude'] = x['usa_state_longitude'].fillna(0)



x['Latitude'] = x['latitude'] + x['usa_state_latitude']

x['Longitude'] = x['longitude'] + x['usa_state_longitude']



x.drop(['usa_state_latitude'], axis=1, inplace=True)

x.drop(['usa_state_longitude'], axis=1, inplace=True)

x.drop(['latitude'], axis=1, inplace=True)

x.drop(['longitude'], axis=1, inplace=True)

display(x.head(5))



# Data Cleansing



x.rename(columns={'Country_Region':'Country'}, inplace=True)

x.rename(columns={'Province_State':'State'}, inplace=True)

x['Date'] = pd.to_datetime(x['Date'], infer_datetime_format=True)



x.loc[:, 'Date'] = x.Date.dt.strftime("%y%m%d")

x["Date"]  = x["Date"].astype(int)



x.drop(['Country', 'State'], axis=1, inplace=True)

display(x.head(5))
# Prediction using Random Forest Regressor

# Splitting the training dataset (Confirmed Case Prediction)

train_x1, val_x1, train_y1, val_y1 = train_test_split(x, y1, random_state=1)



# Training the data using Randomforest Regressor (Confirmed Case Prediction)

rf_model1 = RandomForestRegressor(random_state=1)

rf_model1.fit(train_x1, train_y1)



# Predicting Confirmed Cases in the training dataset

rf_val_predictions1 = rf_model1.predict(val_x1)

print("Predicted Confirmedcase (training set): ", rf_val_predictions1[: 5])

print("Actual Confirmedcase (training set): ", val_y1.head(5))





# Mean Absolute error (Confirmed Case Prediction)

val_mae1 = mean_absolute_error(rf_val_predictions1, val_y1)

print("Mean Absolute Error (Confirmed Case Prediction) using Random Forest Regressor: {:,.0f}".format(val_mae1))



#-------------------------------------



# Splitting the training dataset (Fatalities Prediction)

train_x2, val_x2, train_y2, val_y2 = train_test_split(x, y2, random_state=1)



# Training the data using Randomforest Regressor (Fatalities Prediction)

rf_model2 = RandomForestRegressor(random_state=1)

rf_model2.fit(train_x2, train_y2)



# Predicting Fatalities in the training dataset

rf_val_predictions2 = rf_model2.predict(val_x2)

print("Predicted Fatalities (training set): ", rf_val_predictions2[: 5])

print("Actual Fatalities (training set): ", val_y2.head(5))



# Mean Absolute error (Fatalities Prediction)

val_mae2 = mean_absolute_error(rf_val_predictions2, val_y2)

print("Mean Absolute Error (Fatalities) using Random Forest Regressor: {:,.0f}".format(val_mae2))
# Importing and Preparing test data



x_test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')



# Filling the missing values on Province_State by 0

x_test['Province_State'] = x_test['Province_State'].fillna(0)



# Combining the longitude and latitude for all countries (except USA states) with the training dataset

x_test = pd.merge(x_test,a,left_on = 'Country_Region', right_on ='country', how ='left')



# Dropping the additional 'country' column

x_test.drop(['country'], axis=1, inplace=True)



# Combining the longitude and latitude of USA states with the training dataset

x_test = pd.merge(x_test,b,left_on = 'Province_State', right_on ='usa_state', how ='left')



# Dropping the additional 'usa_state' column

x_test.drop(['usa_state'], axis=1, inplace=True)



# Filling the missing values 

x_test['latitude'] = x_test['latitude'].fillna(0)

x_test['longitude'] = x_test['longitude'].fillna(0)

x_test['usa_state_latitude'] = x_test['usa_state_latitude'].fillna(0)

x_test['usa_state_longitude'] = x_test['usa_state_longitude'].fillna(0)



x_test['Latitude'] = x_test['latitude'] + x_test['usa_state_latitude']

x_test['Longitude'] = x_test['longitude'] + x_test['usa_state_longitude']



x_test.drop(['usa_state_latitude'], axis=1, inplace=True)

x_test.drop(['usa_state_longitude'], axis=1, inplace=True)

x_test.drop(['latitude'], axis=1, inplace=True)

x_test.drop(['longitude'], axis=1, inplace=True)





x_test.head(5)
x_test.rename(columns={'Country_Region':'Country'}, inplace=True)

x_test.rename(columns={'Province_State':'State'}, inplace=True)

x_test['Date'] = pd.to_datetime(x_test['Date'], infer_datetime_format=True)



x_test.loc[:, 'Date'] = x_test.Date.dt.strftime("%y%m%d")

x_test["Date"]  = x_test["Date"].astype(int)



x_test.drop(['Country', 'State'], axis=1, inplace=True)



x_test.head(5)

# Predicting Confirmed Cases in the training dataset

rf_test_predictions1 = rf_model1.predict(x_test)



# Predicting Fatalities in the training dataset

rf_test_predictions2 = rf_model2.predict(x_test)



print(rf_test_predictions1)

print(rf_test_predictions2)
output = pd.DataFrame({'ForecastId': x_test.ForecastId,

                       'ConfirmedCases': rf_test_predictions1,

                        'Fatalities': rf_test_predictions2})

print(output)

output.to_csv('submission.csv', index=False)