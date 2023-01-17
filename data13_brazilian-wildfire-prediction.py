# import necessery libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import plotly.express as px

import plotly.graph_objects as go



from sklearn import metrics

from datetime import datetime

from sklearn import linear_model

from sklearn import preprocessing

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# load csv file

data = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv', encoding='latin1')
# check the data types

data.info()
# display first few rows

data.head()
# view basic statistical details

data.describe()
# chaeck range of years

data['year'].unique()
# check number of states

data['state'].nunique()
# check fires distribution across all states

plt.figure(figsize = (15, 7))

sns.distplot(data['number'], kde = False)
# check for duplicates

data.duplicated().any()
# drop duplicates and only keep the unique values

data = data.drop_duplicates()
data['month'].unique()
data['month'] = data['month'].map({'Janeiro': 'Jan', 'Fevereiro': 'Feb', 'Mar\xe7o': 'Mar', 'Abril': 'Apr', 'Maio': 'May', 'Junho': 'June', 'Julho': 'July', 'Agosto': 'Aug', 'Setembro': 'Sept', 'Outubro': 'Oct', 'Novembro': 'Nov', 'Dezembro': 'Dec'})
# check for the changes

data['month'].unique()
# check for null values

data.isnull().values.any()
data['state'] = data['state'].apply(lambda x: x.replace('Par\xe1', 'Para'))
data['state'].unique()
# convert date strings into datetime objects

data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
# fire counts by year

per_year = data.groupby('year')['number'].sum().round(0).reset_index()

per_year

# per_year.sort_values('number', ascending = False)
# visualise number of fires per year

per_year = go.Figure(go.Scatter(x = per_year['year'], y = per_year['number'], mode = 'lines+markers'))

per_year.update_layout(title = 'Number of Fires Per Year', xaxis_title = 'Year', yaxis_title = 'Fires')

per_year.show()
# total number of fires by months

per_month = data.groupby('month')['number'].sum().round(0).reset_index()

per_month
# visualise number of fires per month

per_month = go.Figure(go.Bar(x = per_month['month'], y = per_month['number'], marker_color = 'lightslategrey'))

per_month.update_layout(title = 'Number of Fires Per Month', xaxis_title = 'Month', yaxis_title = 'Fires')

per_month.show()
# total number of fires per state

per_state = data.groupby('state')['number'].sum().round(0).reset_index()

per_state
per_state = go.Figure(go.Bar(x = per_state['state'], y = per_state['number'], marker_color = 'lightslategrey'))

per_state.update_layout(title = 'Number of Fires Per State', xaxis_title = 'State', yaxis_title = 'Fires')

per_state.show()
# top 10 states per year

top_states = pd.DataFrame(data.groupby('state')['number'].sum().round(0).reset_index())

top_states.head(10)
# get total number of fires per state during the period of 1998-2017

states_per_year = pd.pivot_table(data.drop(['date', 'month'], axis = 1), values = 'number', columns = 'state', index = 'year', aggfunc = np.sum)

states_per_year.head()
# top 10 states

top_10 = states_per_year.sum().round(0).nlargest(10)

top_10
# visualise top 10 states

to_10_plot = go.Figure()

colours = ['#bcbd22', '#ff7f0e', '#e377c2', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#1f77b4', '#7f7f7f', '#17becf']

for state, i in zip(top_10.index, range(len(top_10.index))):

    to_10_plot.add_trace(go.Scatter(x = states_per_year.index, y = states_per_year[state], mode = 'lines+markers', name = state, line={'color': colours[i]}))



to_10_plot.update_layout(title = 'Total Number of Fires Per State During the Period of 1998-2017', xaxis_title = 'Year', yaxis_title = 'Fires')

to_10_plot.show()
# extract the day of the week, monday = 0 and sunday = 6

data['day'] = data['date'].dt.dayofweek
# determine whether the day is a weekend or not, weekend = 1 and weekday = 0

data['weekend'] = np.where(data['day'].isin([5, 6]), 1, 0)
# encode months

month = pd.get_dummies(data['month'])
# encode states

state = pd.get_dummies(data['state'])
# encode year

year = pd.get_dummies(data['year'])
# encode date

date = pd.get_dummies(data['date'])
# spilt data set into two subsets: training set (X) and test set (y)

X = pd.concat([data['day'], data['weekend'], state, date, month, year], axis = 1)

y = data['number']
# split 61% of the data to the training set and 39% of the data to test set 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.39, random_state = 5)
# build the model on the training set 

lr = linear_model.LinearRegression()

lr.fit(X_train, y_train)
# use the test set as a holdout sample to test the trained model using the test data

predictions = lr.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))