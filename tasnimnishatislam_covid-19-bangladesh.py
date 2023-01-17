import pandas as pd

from sklearn import datasets, linear_model, svm

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

import math

import pandas as pd

from fbprophet import Prophet
# Import Dependencies

%matplotlib inline



# Start Python Imports

import math, time, random, datetime



# Data Manipulation

import numpy as np

import pandas as pd

from scipy.stats import invgauss



# Visualization 

import matplotlib.pyplot as plt

import missingno

import seaborn as sns

plt.style.use('seaborn-whitegrid')



# Preprocessing

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize



# Machine learning

from sklearn.model_selection import train_test_split

from sklearn import model_selection, tree, preprocessing, metrics, linear_model

from sklearn.svm import LinearSVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier

from sklearn.tree import DecisionTreeClassifier



# Let's be rebels and ignore warnings for now

import warnings

warnings.filterwarnings('ignore')
# Load the Diabetes dataset

data = pd.read_csv("../input/novel-corona-virus-2019-datasetbangladesh/covid_19_data_Bangladesh.csv")

data.head()
# create training and testing vars

train, test = train_test_split(data, test_size=0.15, random_state = 0)

print (train.shape)

print (test.shape)
train.head()
test.head()
train.describe()
#Plot graphic of missing value

missingno.matrix(train, figsize = (30,5))
#to perform data analysis we create data frame

df_bin = pd.DataFrame() #for discreated continuous variable

df_con = pd.DataFrame() #for continuous variables
train.dtypes
# How many people confirmed/death/recovered in context of SNo?

sns.jointplot(x='SNo', y="Confirmed", data=train, color = "y");



sns.jointplot(x='SNo', y="Deaths", data=train, color="r");



sns.jointplot(x='SNo', y="Recovered", data=train, color= "g");
# Let's add this to our subset dataframes

df_bin['SNo'] = train['SNo']

df_con['SNo'] = train['SNo']
df_bin.head()
df_con.head()
# How many people confirmed/death/recovered in context of ObservationDate?

train['ObservationDate'] = pd.to_datetime(train['ObservationDate'])



fig = plt.figure(figsize=(15,1))

train[['ObservationDate','Confirmed']].set_index('ObservationDate').plot(color = "y")



fig = plt.figure(figsize=(15,1))

train[['ObservationDate','Deaths']].set_index('ObservationDate').plot(color = "r")



fig = plt.figure(figsize=(15,1))

train[['ObservationDate','Recovered']].set_index('ObservationDate').plot(color = "g")
# Let's add this to our subset dataframes

df_bin['ObservationDate'] = train['ObservationDate']

df_con['ObservationDate'] = train['ObservationDate']
df_bin.head()
df_con.head()
df_bin.head()
df_con.head()
# How many people confirmed/death/recovered in context of LastUpdate?

train['Last Update'] = pd.to_datetime(train['Last Update'])



fig = plt.figure(figsize=(15,1))

train[['Last Update','Confirmed']].set_index('Last Update').plot(color = "y")



fig = plt.figure(figsize=(15,1))

train[['Last Update','Deaths']].set_index('Last Update').plot(color = "r")



fig = plt.figure(figsize=(15,1))

train[['Last Update','Recovered']].set_index('Last Update').plot(color = "g")
df_bin['Last Update'] = train['Last Update']

df_con['Last Update'] = train['Last Update']
df_bin.head()
df_con.head()
df_bin['Confirmed'] = train['Confirmed']

df_con['Confirmed'] = train['Confirmed']



df_bin['Deaths'] = train['Deaths']

df_con['Deaths'] = train['Deaths']



df_bin['Recovered'] = train['Recovered']

df_con['Recovered'] = train['Recovered']
df_bin.head()
df_con.head()
print(len(df_bin))
print(len(df_con))
confirmed = df_con.groupby('ObservationDate').sum()['Confirmed'].reset_index()

deaths = df_con.groupby('ObservationDate').sum()['Deaths'].reset_index()

recovered = df_con.groupby('ObservationDate').sum()['Recovered'].reset_index()
confirmed.head()
deaths.head()
recovered.head()
days_to_forecast = 30 # changable

first_forecasted_date = sorted(list(set(df_con['ObservationDate'].values)))[-days_to_forecast]



print('The first date to perform forecasts for is: ' + str(first_forecasted_date))
confirmed_df = df_con[['SNo', 'ObservationDate', 'Last Update', 'Confirmed']]

confirmed_df.tail()
confirmed.columns = ['ds','y']

confirmed['ds'] = pd.to_datetime(confirmed['ds'])
confirmed.head()
deaths_df = df_con[['SNo', 'ObservationDate', 'Last Update', 'Deaths']]

deaths_df.tail()
deaths.columns = ['ds','y']

deaths['ds'] = pd.to_datetime(deaths['ds'])
recovered_df = df_con[['SNo', 'ObservationDate', 'Last Update', 'Recovered']]

recovered_df.tail()
recovered.columns = ['ds','y']

recovered['ds'] = pd.to_datetime(confirmed['ds'])
m = Prophet(interval_width=0.95)

m.fit(confirmed)

future = m.make_future_dataframe(periods=30)

future_confirmed = future.copy() # for non-baseline predictions later on

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
confirmed_forecast_plot = m.plot(forecast)
forecast_components = m.plot_components(forecast)
m = Prophet(interval_width=0.95)

m.fit(deaths)

future = m.make_future_dataframe(periods=30)

future_deaths = future.copy() # for non-baseline predictions later on

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
deaths_forecast_plot = m.plot(forecast)
forecast_components = m.plot_components(forecast)
m = Prophet(interval_width=0.95)

m.fit(recovered)

future = m.make_future_dataframe(periods=30)

future_confirmed = future.copy() # for non-baseline predictions later on

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
deaths_forecast_plot = m.plot(forecast)
forecast_components = m.plot_components(forecast)