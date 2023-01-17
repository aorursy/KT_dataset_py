# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Fitting Random Forest Regression to the dataset 

# import the regressor 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor 

from sklearn.metrics import mean_squared_error

# Import scikit_learn module for k-fold cross validation

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_ds = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

test_ds =  pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')
train_ds.head()

train_ds.info()

print("*"*150)

test_ds.head()

test_ds.info()

print("*"*150)

submission.head()

submission.info()
train_ds.head()
test_ds.head()
submission.head()
train_ds.isnull().sum()
test_ds.isnull().sum()
train = train_ds.drop(['Province/State'], axis=1)

test = test_ds.drop(['Province/State'], axis=1)
train.head()
test.head()
gdf = train.groupby(['Date', 'Country/Region'])['ConfirmedCases'].max()

gdf = gdf.reset_index()

gdf['Date'] = pd.to_datetime(gdf['Date'])

gdf['Date'] = gdf['Date'].dt.strftime('%m/%d/%Y')

gdf['size'] = gdf['ConfirmedCases'].pow(0.3)



fig = px.scatter_geo(gdf, locations="Country/Region", locationmode='country names', 

                     color="ConfirmedCases", size='size', hover_name="Country/Region", 

                     range_color= [0, 1500], 

                     projection="natural earth", animation_frame="Date", 

                     title='Spread of COVID-19 in world from Jan', color_continuous_scale="Reds")

fig.show()
train_data = train[:17688]

train_data.info()

train_data.columns
train_data.Date =  pd.to_datetime(train_data['Date'])

train_data['dayofyear']=train_data['Date'].dt.dayofyear

train_data.head()
train_data_df = pd.DataFrame(train_data)

print(train_data_df)
train_data_df['CCO'] = pd.factorize(train_data_df['Country/Region'])[0] + 1

print(train_data_df)
train_data_cleaned = train_data_df.drop(['Country/Region'], axis=1)

train_data_cleaned = train_data_df.drop(['Date'], axis=1)
train_data_cleaned.head()
test.head()

test_data = pd.DataFrame(test)

print(test_data)
test_data.Date =  pd.to_datetime(test_data['Date'])

test_data['dayofyear']=test_data['Date'].dt.dayofyear

test_data['CCO'] = pd.factorize(test_data['Country/Region'])[0] + 1

test_data.head()
test_data = test_data.drop(['Date'], axis=1)

test_data = test_data.drop(['Country/Region'], axis=1)

test_data = test_data.drop(['ForecastId'], axis=1)

test_data.head()
columns =['CCO', 'Lat', 'Long', 'dayofyear']

X = train_data_cleaned[columns]

Y_cases = train_data_cleaned['ConfirmedCases']

Y_fatal = train_data_cleaned['Fatalities']
X_train_case, X_val_case, y_train_case, y_val_case = train_test_split(X, Y_cases, test_size=0.3, random_state=42)
 # create regressor object 

regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 



regressor.fit(X_train_case, y_train_case)

y_predicted_cases_val = regressor.predict(X_val_case)
print (y_predicted_cases_val)

print (y_predicted_cases_val.shape)

mse_cases_val = mean_squared_error(y_val_case, y_predicted_cases_val)

print (mse_cases_val)
y_predicted_cases_test = regressor.predict(test_data)

print (y_predicted_cases_test.shape)

print (y_predicted_cases_test[:5])
X_train_fatal, X_val_fatal, y_train_fatal, y_val_fatal = train_test_split(X, Y_fatal, test_size=0.3, random_state=42)
rf_fatal = RandomForestRegressor()

rf_fatal.fit(X_train_fatal, y_train_fatal)

y_predicted_fatal_val = rf_fatal.predict(X_val_fatal)
mse_fatal_val = mean_squared_error(y_val_fatal, y_predicted_fatal_val)

print (mse_fatal_val)
y_predicted_fatal_test = rf_fatal.predict(test_data)

print (y_predicted_fatal_test.shape)

print (y_predicted_fatal_test[:5])
print (submission.shape)

print (submission.head())
submission.drop(['ConfirmedCases','Fatalities'], axis=1, inplace=True)

print (submission.head())

print (submission.shape)
print (y_predicted_cases_test.shape)

print (y_predicted_fatal_test.shape)
submission['ConfirmedCases'] = y_predicted_cases_test

submission['Fatalities'] = y_predicted_fatal_test

submission.tail(5)
submission.to_csv('submission.csv', index=False)