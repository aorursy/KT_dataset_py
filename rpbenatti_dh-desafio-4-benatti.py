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
base_url = '/kaggle/input/web-traffic-time-series-forecasting/'



key_1 = pd.read_csv(base_url+'key_1.csv')

train_1 = pd.read_csv(base_url+'train_1.csv')

sample_submission_1 = pd.read_csv(base_url+'sample_submission_1.csv')
print(train_1.shape, key_1.shape, sample_submission_1.shape)
train_1.head()
key_1.head()
print(key_1.Page[0])

print

print(key_1.Page[59])

print

print(key_1.Page[60])
sample_submission_1.head()
train_1.info()
train_1.head()
# Creating a list of wikipedia main sites 

sites = ["wikipedia.org", "commons.wikimedia.org", "www.mediawiki.org"]



# Function to create a new column having the site part of the article page

def filter_by_site(page):

    for site in sites:

        if site in page:

            return site



# Creating a new column having the site part of the article page

train_1['Site'] = train_1.Page.apply(filter_by_site)
train_1['Site'].value_counts(dropna=False)
import matplotlib.pyplot as plt

import seaborn as sns; sns.set()



plt.figure(figsize=(12, 6))

plt.title("Number of Wikipedia Articles by Sites", fontsize="18")

train_1['Site'].value_counts().plot.bar(rot=0);
# Checking which country codes exist in the article pages

train_1.Page.str.split(pat=".wikipedia.org", expand=True).iloc[:,0].str[-3:].value_counts().index.to_list()
# Creating a list of country codes

train_1.Page.str.split(pat=".wikipedia.org", expand=True).iloc[:,0].str[-2:].value_counts().index.to_list()[0:7]
# Checking which agents + access exist in the article pages and creating a list with them

train_1.Page.str.split(pat=".wikipedia.org", expand=True).iloc[:,1].str[1:].value_counts().index.to_list()
# Creating the list of country codes and agents

countries = train_1.Page.str.split(pat=".wikipedia.org", expand=True).iloc[:,0].str[-2:].value_counts().index.to_list()[0:7]

agents = train_1.Page.str.split(pat=".wikipedia.org", expand=True).iloc[:,1].str[1:].value_counts().index.to_list()



# Function to create a new column having the country code part of the article page

def filter_by_country(page):

    for country in countries:

        if "_"+country+"." in page:

            return country



# Creating a new column having the country code part of the article page

train_1['Country'] = train_1.Page.apply(filter_by_country)



# Function to create a new column having the agent + access part of the article page

def filter_by_agent(page):

    for agent in agents:

        if agent in page:

            return agent



# Creating a new column having the agent part of the article page

train_1['Agent'] = train_1.Page.apply(filter_by_agent)
# Understanding what are the NaN values for the Country column

# It seems that the URL page does not contain the country code for those cases



train_1.Page[train_1['Country'].isna() == True]
plt.figure(figsize=(12, 6))

plt.title("Number of Wikipedia Articles by Country", fontsize="18")

train_1['Country'].value_counts(dropna=False).plot.bar(rot=0);
train_1['Agent'].value_counts(dropna=False)
plt.figure(figsize=(12, 6))

plt.title("Number of Wikipedia Articles by Agents/Access", fontsize="18")

train_1['Agent'].value_counts().plot.bar(rot=0);
# Creating a sample dataset from the Train dataset for analysis

train_1_sample = train_1.drop(['Site','Country','Agent'], axis=1).sample(6, random_state=42)

train_1_sample
# Transposing the sample dataset to have Date Time at the index

train_1_sampleT = train_1_sample.drop('Page', axis=1).T

train_1_sampleT.columns = train_1_sample.Page.values

train_1_sampleT.shape
train_1_sampleT.head()
# Plotting the Series from the sample dataset 

plt.figure(figsize=(16,8))



for k, v in enumerate(train_1_sampleT.columns):

    plt.subplot(2, 3, k + 1)

    plt.title( str(v.split(".org")[0])+".org"+"\n"+str(v.split(".org")[1]) )

    train_1_sampleT[v].plot()



plt.tight_layout();
# Plotting the Series from the sample dataset at the same graph

plt.figure(figsize=(15,8))



for v in train_1_sampleT.columns:

    plt.plot(train_1_sampleT[v])

    plt.legend(loc='upper center');
# Plotting the histograms for the Series from the sample dataset

plt.figure(figsize=(16,8))



for k, v in enumerate(train_1_sampleT.columns):

    plt.subplot(2, 3, k + 1)

    plt.title( str(v.split(".org")[0])+".org"+"\n"+str(v.split(".org")[1]) )

    sns.distplot(train_1_sampleT[v])



plt.tight_layout();
# Checking that the number of visits to the Wikipedia Articles have Gaussian Distribution (p-value=0)

from scipy.stats import kstest, ks_2samp



pages = list(train_1_sampleT.columns)



print("Kolgomorov-Smirnov - Normality Test")

print()



for p in pages:

    print(p,':', kstest(train_1_sampleT[p], 'norm', alternative = 'less'))    
# List of the main Wikipedia Article sites

sites
# Creating sample datasets from the train dataset and filtering them by sites

train_1_sample_site0 = train_1[train_1['Site'] == sites[0]].drop(['Site','Country','Agent'], axis=1).sample(6, random_state=42)

train_1_sample_site1 = train_1[train_1['Site'] == sites[1]].drop(['Site','Country','Agent'], axis=1).sample(6, random_state=42)

train_1_sample_site2 = train_1[train_1['Site'] == sites[2]].drop(['Site','Country','Agent'], axis=1).sample(6, random_state=42)



# Transposing them to have the Date Time as index

train_1_sampleT_site0 = train_1_sample_site0.drop('Page', axis=1).T

train_1_sampleT_site0.columns = train_1_sample_site0.Page.values

train_1_sampleT_site1 = train_1_sample_site1.drop('Page', axis=1).T

train_1_sampleT_site1.columns = train_1_sample_site1.Page.values

train_1_sampleT_site2 = train_1_sample_site2.drop('Page', axis=1).T

train_1_sampleT_site2.columns = train_1_sample_site2.Page.values
# Plotting the Series from the sample datasets

plt.figure(figsize=(16,8))



for k, v in enumerate(train_1_sampleT_site0.columns):

    plt.subplot(2, 3, k + 1)

    plt.title( str(v.split(".org")[0])+".org"+"\n"+str(v.split(".org")[1]) )

    train_1_sampleT_site0[v].plot()



plt.tight_layout();
# Plotting the Series from the sample datasets at the same graph

plt.figure(figsize=(15,8))



for v in train_1_sampleT_site0.columns:

    plt.plot(train_1_sampleT_site0[v])

    plt.legend(loc='upper center');
# Plotting the Series from the sample datasets

plt.figure(figsize=(16,8))



for k, v in enumerate(train_1_sampleT_site1.columns):

    plt.subplot(2, 3, k + 1)

    plt.title( str(v.split(".org")[0])+".org"+"\n"+str(v.split(".org")[1]) )

    train_1_sampleT_site1[v].plot()



plt.tight_layout();
# Plotting the Series from the sample datasets at the same graph

plt.figure(figsize=(15,8))



for v in train_1_sampleT_site1.columns:

    plt.plot(train_1_sampleT_site1[v])

    plt.legend(loc='upper center');
# Plotting the Series from the sample datasets

plt.figure(figsize=(16,8))



for k, v in enumerate(train_1_sampleT_site2.columns):

    plt.subplot(2, 3, k + 1)

    plt.title( str(v.split(".org")[0])+".org"+"\n"+str(v.split(".org")[1]) )

    train_1_sampleT_site2[v].plot()



plt.tight_layout();
# Plotting the Series from the sample datasets at the same graph

plt.figure(figsize=(15,8))



for v in train_1_sampleT_site2.columns:

    plt.plot(train_1_sampleT_site2[v])

    plt.legend(loc='upper center');
train_1_sampleT_site2.columns[4]
# List of the Wikipedia Article country codes

countries
# Creating a sample dataset from the train dataset for countries having "de" code

train_1_sample_de = train_1[train_1['Country'] == countries[2]].drop(['Site','Country','Agent'], axis=1).sample(6, random_state=42)



# Transposing the sample dataset to have Date Time at the index

train_1_sampleT_de = train_1_sample_de.drop('Page', axis=1).T

train_1_sampleT_de.columns = train_1_sample_de.Page.values
# Plotting the Series from the sample dataset

plt.figure(figsize=(16,8))



for k, v in enumerate(train_1_sampleT_de.columns):

    plt.subplot(2, 3, k + 1)

    plt.title( str(v.split(".org")[0])+".org"+"\n"+str(v.split(".org")[1]) )

    train_1_sampleT_de[v].plot()



plt.tight_layout();
# Plotting the Series from the sample datasets at the same graph

plt.figure(figsize=(15,8))



for v in train_1_sampleT_de.columns:

    plt.plot(train_1_sampleT_de[v])

    plt.legend(loc='upper center');
# Import Prophet library

from fbprophet import Prophet
# Picked up one Time Series for the prophet modeling

train_1_sampleT.columns[1]
# Creating a dataframe for the Time Series from the train_1 samples dataset

ds = pd.Series(train_1_sampleT.index)

y = pd.Series(train_1_sampleT.iloc[:,1].values)

frame = { 'ds': ds, 'y': y }

df = pd.DataFrame(frame)

df.head()
df.plot();
# Instantiate and fit the Prophet model with no hyperparameters at all

m = Prophet()

m.fit(df);
# Make dataframe for the future predictions to the next 60 days

# By default it will also include the dates from the history

# In summary it will have 550 + 60 days (610)

future = m.make_future_dataframe(periods=60)

future.tail()
# Predicting the values from the future dataframe

forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
forecast.shape
# The forecast object here is a new dataframe that includes a column yhat with the forecast, 

# as well as columns for components and uncertainty intervals

forecast.head()
# Plotting the forecast by calling the Prophet.plot method and passing in the forecast dataframe

fig1 = m.plot(forecast)
# Plotting the forecast components by calling the Prophet.plot_components method

# By default it includes the trend and seasonality of the time series

fig2 = m.plot_components(forecast)
# Plotting both the Actual values and Predict values at the same graph for comparison

plt.figure(figsize=(15, 7))

plt.plot(df.y)                  # Actual values in default blue color

plt.plot(forecast.yhat, "g");   # Predicted values in green color
forecast['yhat'].tail()
# Setting the floor value to 0 and the capacity to a lower value in the future

df['cap'] = 500

df['floor'] = 0.0

future['cap'] = 500

future['floor'] = 0.0



# Instantiating prophet 'logistic' growth mode, then fitting and predicting future values

m = Prophet(growth='logistic')

forecast = m.fit(df).predict(future)



# Plotting both the forecast predictions and components

fig1 = m.plot(forecast)

fig2 = m.plot_components(forecast)
# Instantiate prophet with default seasonality parameters, fitting and predicting the future

# Plotting both the forecast and its components

# I will keep the default growth='linear' by now instead of 'logistic'

m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)

forecast = m.fit(df).predict(future)

fig1 = m.plot(forecast)

fig2 = m.plot_components(forecast)
# Plotting both the Actual values and Predict values at the same graph for comparison

plt.figure(figsize=(15, 7))

plt.plot(df.y)                  # Actual values in default blue color

plt.plot(forecast.yhat, "g");   # Predicted values in green color
# Checking the locations of the significant changepoints

from fbprophet.plot import add_changepoints_to_plot

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
# Increasing the 'changepoint_range' parameter from default 80% to 90%

m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True,

            changepoint_range=0.9)

forecast = m.fit(df).predict(future)

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
deltas = m.params['delta'].mean(0)

fig = plt.figure(facecolor='w', figsize=(10, 6))

ax = fig.add_subplot(111)

ax.bar(range(len(deltas)), deltas, facecolor='#0072B2', edgecolor='#0072B2')

ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)

ax.set_ylabel('Rate change')

ax.set_xlabel('Potential changepoint')

fig.tight_layout()
# Changing the changepoint_range back to 80% since I don't want to make the trend more negative

# Also increasing the changepoint_prior_scale from default 0.05 to 0.7

# By default, changepoint_prior_scale parameter is set to 0.05, andi ncreasing it will make the trend more flexible

m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True,

            changepoint_range=0.8, changepoint_prior_scale=0.7)

forecast = m.fit(df).predict(future)

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
deltas = m.params['delta'].mean(0)

fig = plt.figure(facecolor='w', figsize=(10, 6))

ax = fig.add_subplot(111)

ax.bar(range(len(deltas)), deltas, facecolor='#0072B2', edgecolor='#0072B2')

ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)

ax.set_ylabel('Rate change')

ax.set_xlabel('Potential changepoint')

fig.tight_layout()
# Plotting both the Actual values and Predict values at the same graph for comparison

plt.figure(figsize=(15, 7))

plt.plot(df.y)                  # Actual values in default blue color

plt.plot(forecast.yhat, "g");   # Predicted values in green color
train_1_sampleT.columns[1]
"_es." in train_1_sampleT.columns[1]
from datetime import date

import holidays



# Select country

es_holidays = holidays.Spain(years = [2015,2016,2017])

es_holidays = pd.DataFrame.from_dict(es_holidays, orient='index')

es_holidays = pd.DataFrame({'holiday': 'Spain', 'ds': es_holidays.index})
es_holidays.head()
# Instantiate prophet with seasonality, changepoints and holidays parameters

m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True,

            changepoint_range=0.8, changepoint_prior_scale=0.7,

            holidays=es_holidays)

m.add_country_holidays(country_name='ES')

# Fitting and predicting the future

forecast = m.fit(df).predict(future)

# Plotting both the forecast and its components

fig1 = m.plot(forecast)

fig2 = m.plot_components(forecast)
# Instantiate prophet with seasonality, changepoints and holidays parameters

m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True,

            changepoint_range=0.8, changepoint_prior_scale=0.7,

            holidays=es_holidays,

            interval_width=0.95,

            mcmc_samples=0)

m.add_country_holidays(country_name='ES')

# Fitting and predicting the future

forecast = m.fit(df).predict(future)

# Plotting both the forecast and its components

fig1 = m.plot(forecast)

fig2 = m.plot_components(forecast)
deltas = m.params['delta'].mean(0)

fig = plt.figure(facecolor='w', figsize=(10, 6))

ax = fig.add_subplot(111)

ax.bar(range(len(deltas)), deltas, facecolor='#0072B2', edgecolor='#0072B2')

ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)

ax.set_ylabel('Rate change')

ax.set_xlabel('Potential changepoint')

fig.tight_layout()
plt.figure(figsize=(15, 7))

plt.plot(df.y)

plt.plot(forecast.yhat, "g");
from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



fig = plot_plotly(m, forecast)  # This returns a plotly Figure

py.iplot(fig)
m.params
def smape(y_true, y_pred):

    denominator = (np.abs(y_true) + np.abs(y_pred))

    diff = np.abs(y_true - y_pred) / denominator

    diff[denominator == 0] = 0.0

    return 200 * np.mean(diff)



# Source: http://shortnotes.herokuapp.com/how-to-implement-smape-function-in-python-149
smape_single_page = smape(df.y, forecast.yhat)

smape_single_page
from fbprophet.diagnostics import cross_validation
# horizon: forecast horizon

# initial: size of the initial training period

# period: spacing between cutoff dates

#

# Here we do cross-validation to assess prediction performance on a horizon of 60 days, 

# starting with 130 days of training data in the first cutoff and then making predictions every 60 days

# On this 610 days time series, this corresponds to 8 total forecasts



cv_results = cross_validation(m, initial='360 days', period='30 days', horizon='60 days')
smape_baseline = smape(cv_results.y, cv_results.yhat)

smape_baseline
train_1_all = train_1.drop(['Page','Site','Country','Agent'], axis=1).T

train_1_all.columns = train_1.Page.values

train_1_all.shape
train_1_all.head()
# Filling up NaN values with 0 visits to avoid breaking the model fit

train_1_all.fillna(0, inplace=True)



# Selecting a few series to run the Prophet model against

num_series = 10

train_1_sample = train_1_all.sample(num_series, axis=1, random_state=42)
# Plotting the Series from the sample datasets at the same graph

plt.figure(figsize=(15,8))



for v in train_1_sample.columns:

    plt.plot(train_1_sample[v])

    plt.legend(loc='upper center');
%%time



smape_partial = 0



for k, v in enumerate(train_1_sample.columns):

    ds = pd.Series(train_1_sample.index)

    y = pd.Series(train_1_sample.iloc[:,k].values)

    frame = { 'ds': ds, 'y': y }

    df = pd.DataFrame(frame)

    m_partial = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)

    forecast = m_partial.fit(df).predict(future)

    smape_partial += smape(df.y, forecast.yhat)



smape_average = smape_partial / len(train_1_sample.columns)

smape_average
# train_1_sampleT.columns[1]+"_"+"2017-01-01"

# train_1_sampleT.columns[1]+"_"+"2017-01-01" in list(key_1.Page.values)