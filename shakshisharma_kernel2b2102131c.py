#Import Libararies
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import time
import collections
from datetime import timedelta
from datetime import datetime 
import scipy.stats as stats

import pycountry
import plotly
import plotly.io as pio
import plotly.express as px

from ipywidgets import interact
import statsmodels.api as sm
#  Read datasets
test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")
train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")
train.head()
# Manipulating the original dataframe
#train = pd.read_csv("train.csv")
countrydate_evolution = train[train['ConfirmedCases']>0]
countrydate_evolution = countrydate_evolution.groupby(['Date','Country_Region']).sum().reset_index()

# Creating the visualization
fig = px.choropleth(countrydate_evolution, locations="Country_Region", locationmode = "country names", color="ConfirmedCases", 
                    hover_name="Country_Region", animation_frame="Date", 
                   )

fig.update_layout(
    title_text = 'Global Spread of Coronavirus',
    title_x = 0.5,
    autosize=True,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
    
fig.show()
#converting Date to DateTime
train.Date = pd.to_datetime(train['Date'])
print(train['Date'].max())
print(test['Date'].min())
date_filter = train['Date'] <test['Date'].min()
train = train.loc[date_filter]
train.info()
# lets get Cumulative sum of ConfirmedCases and Fatalities for each country on each data (same as original data)
# Doing to create copy without ID and 

train_country_date = train.groupby(['Country_Region', 'Date'],as_index=False)['ConfirmedCases', 'Fatalities'].sum()

print(train_country_date.info())
print(train_country_date.isnull().sum())

train_country_date.info()


# Adding day, month, day of week columns 

train_country_date['Month'] = train_country_date['Date'].dt.month
train_country_date['Day'] = train_country_date['Date'].dt.day
train_country_date['Day_Week'] = train_country_date['Date'].dt.dayofweek
train_country_date['quarter'] = train_country_date['Date'].dt.quarter
train_country_date['dayofyear'] = train_country_date['Date'].dt.dayofyear
train_country_date['weekofyear'] = train_country_date['Date'].dt.weekofyear
train_country_date.info()
# Converting Date Object to Datetime type

test.Date = pd.to_datetime(test['Date'])
test.Date.head(2)
# adding Month, DAy, Day_week columns Using Pandas Series.dt.month

test['Month'] = test['Date'].dt.month
test['Day'] = test['Date'].dt.day
test['Day_Week'] = test['Date'].dt.dayofweek
test['quarter'] = test['Date'].dt.quarter
test['dayofyear'] =test['Date'].dt.dayofyear
test['weekofyear'] = test['Date'].dt.weekofyear
test.info()

# Lets select the Common Labels and concatenate.

labels = ['Country_Region','Date','Month', 'Day', 'Day_Week','quarter', 'dayofyear', 'weekofyear']
train_clean = train_country_date[labels]
test_clean = test[labels]
cleaned_data = pd.concat([train_clean, test_clean], axis = 0)
cleaned_data.head(5)
cleaned_data.info()
#label encoding data for model generation
from sklearn.preprocessing import LabelEncoder
# Label Encoder for Countries 

enc = LabelEncoder()
cleaned_data['Country'] = enc.fit_transform(cleaned_data['Country_Region'])
cleaned_data
# Dropping Country/Region and Date

cleaned_data.drop(['Country_Region', 'Date'], axis = 1, inplace=True)
cleaned_data.info()
index_split = train.shape[0]
cleaned_train_data = cleaned_data[:index_split]
cleaned_test_data = cleaned_data[index_split:]
cleaned_train_data.tail(5)

from sklearn.model_selection import train_test_split
y_confirmed = train['ConfirmedCases']
y_fatalities = train['Fatalities']
x = cleaned_train_data[['Month', 'Day', 'Day_Week','quarter', 'dayofyear', 'weekofyear', 'Country']]
from sklearn.ensemble import RandomForestClassifier
Tree_model = RandomForestClassifier(max_depth=200, random_state=0)

x_train, x_test, y_train, y_test = train_test_split(x, y_confirmed, test_size = 0.3, random_state = 42)
x_train_fatal, x_test_fatal, y_train_fatal, y_test_fatal = train_test_split(x, y_fatalities, test_size = 0.3, random_state = 42)

from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import mean_squared_error
#confirmed cases
random_forest = RandomForestClassifier(n_estimators =100)
random_forest.fit(x_train, y_train.values.ravel())
#random_forest.score(x_train, y_train)
#random_forest.score(x_test, y_test)
# Predicted Values
y_pred_train = random_forest.predict(x_test)
#for fatalities
random_forest.fit(x, y_fatalities.values.ravel())
random_forest_pred_fatal = random_forest.predict(x_test_fatal)
submission = pd.DataFrame(data = np.zeros((y_pred_train.shape[0],3)), columns = ['ForecastId', 'ConfirmedCases', 'Fatalities'])
submission.shape
y_pred1 = pd.DataFrame(y_pred_train)
y_pred2 = pd.DataFrame(random_forest_pred_fatal)
for i in range(0, len(submission)):
    submission.loc[i,'ForecastId'] = i + 1
    submission.loc[i,'ConfirmedCases'] = y_pred1.iloc[i, 0]
    submission.loc[i,'Fatalities'] = y_pred2.iloc[i, 0]
submission['ForecastId'] = submission['ForecastId'].astype(int)
submission['ConfirmedCases'] = submission['ConfirmedCases'].astype(int)
submission['Fatalities'] = submission['Fatalities'].astype(int)
submission.to_csv('submission.csv', index = False)
#XGBoost Classifier
import xgboost as xgb
reg = xgb.XGBClassifier(n_estimators=100)
#for confirmed cases


reg.fit(x_train, y_train)
reg_y_pred = reg.predict(x_test)
reg.fit(x, y_fatalities)
xgb_pred_fatalities = reg.predict(cleaned_test_data)
submission1 = pd.DataFrame(data = np.zeros((y_pred_train.shape[0],3)), columns = ['ForecastId', 'ConfirmedCases', 'Fatalities'])
submission1.shape
y_pred11 = pd.DataFrame(reg_y_pred)
y_pred22 = pd.DataFrame(xgb_pred_fatalities)
for i in range(0, len(submission1)):
    submission1.loc[i,'ForecastId'] = i + 1
    submission1.loc[i,'ConfirmedCases'] = y_pred11.iloc[i, 0]
    submission1.loc[i,'Fatalities'] = y_pred22.iloc[i, 0]
submission1['ForecastId'] = submission1['ForecastId'].astype(int)
submission1['ConfirmedCases'] = submission1['ConfirmedCases'].astype(int)
submission1['Fatalities'] = submission1['Fatalities'].astype(int)
submission1.to_csv('submission.csv', index = False)
