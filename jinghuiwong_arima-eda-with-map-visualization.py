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
# Run this code for the first time, to install the libraries

# import sys

# !{sys.executable} -m pip install folium

# !{sys.executable} -m pip install plotly
import pandas as pd

import numpy as np





# remove unnecessary columns

df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

df = df.drop(['SNo','Last Update'], axis=1)

df.ObservationDate = df.ObservationDate.apply(pd.to_datetime)

df.sort_values(by='ObservationDate',ascending=False)
df.ObservationDate.unique()  # timeframe is from 22nd Jan to 25th Feb with no missing days

df.ObservationDate.isnull().any()  # no missing values for Observation date



df[['Confirmed','Deaths','Recovered']].isnull().any()  # no missing values for ['Confirmed','Deaths','Recovered']

df['Country/Region'].isnull().any()  # no missing values for Country/Region



df['Province/State'].isnull().any()  # missing values for Province/State
df[df['Confirmed'] < 0 ]  # no invalid / negative cases

df[df['Deaths'] < 0 ]  # no invalid / negative values

df[df['Recovered'] < 0 ]  # no invalid / negative values





# rename countries and provinces

df["Country/Region"].replace({"Iran (Islamic Republic of)": "Iran", "Viet Nam":"Vietnam"}, inplace=True)



df[df['Province/State'].isnull()]['Country/Region'].unique()  # list of countries without provinces/state
df['Country/Region'].replace({"Taipei and environs": "Taiwan"}, inplace=True)

df[~df['Province/State'].isnull()]['Country/Region'].unique() # list of countries with provinces/state

# the 2 lists of countries with and without provinces/state are mutually exclusive. No incorrect entries or errors in country names
df.groupby(['Country/Region','Province/State']).size()  # list of provinces for each country
import plotly.graph_objects as go



global_df = df.groupby('ObservationDate').sum()

global_df['mortality rate'] = round(global_df['Deaths'] / global_df['Confirmed'],4)*100

fig = go.Figure()

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Scatter(x=global_df.index, 

                         y=global_df['mortality rate'],

                         mode='lines+markers',

                         name='Mortality Rate',

                         line=dict(color='red', width=2)))





fig.update_layout(

    title="Global Mortality Rate",

    xaxis_title="Mortality Rate",

    yaxis_title="Date",

    font=dict(

        family="Arial",

        size=16,

        color="white"

    ))



    

fig.show()
global_df = df.groupby('ObservationDate').sum()

fig = go.Figure()

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Scatter(x=global_df.index, 

                         y=global_df['Deaths'],

                         mode='lines+markers',

                         name='Deaths',

                         line=dict(color='red', width=2)))

fig.add_trace(go.Scatter(x=global_df.index, 

                         y=global_df['Confirmed'],

                         mode='lines+markers',

                         name='Confirmed',

                         line=dict(color='blue', width=2)))

fig.add_trace(go.Scatter(x=global_df.index, 

                         y=global_df['Recovered'],

                         mode='lines+markers',

                         name='Recovered',

                         line=dict(color='green', width=2)))



fig.update_layout(

    title="Global Number of Confirmed/Death/Recovered cases",

    xaxis_title="Number of cases",

    yaxis_title="Date",

    font=dict(

        family="Arial",

        size=16,

        color="white"

    ))

fig.show()
latest_df = df[df.ObservationDate == '2020-03-10']

country_df = latest_df.groupby('Country/Region').sum()

fig = go.Figure()

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Bar(

    y=country_df.index,

    x=country_df.Confirmed,

    name='Confirmed',

    orientation='h',

    marker=dict(

        color='rgba(246, 78, 139, 0.6)',

        line=dict(color='rgba(246, 78, 139, 1.0)', width=3)

    )

))

fig.add_trace(go.Bar(

    y=country_df.index,

    x=country_df.Deaths,

    name='Deaths',

    orientation='h',

    marker=dict(

        color='rgba(58, 71, 80, 0.6)',

        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)

    )

))



fig.update_layout(barmode='stack')

fig.update_layout(

    title="Number of Confirmed/Death/Recovered cases for each country",

    yaxis_title="Country names",

    xaxis_title="Number of cases",

    font=dict(

        family="Arial",

        size=10,

        color="white"

    ))

fig.show()
latest_df = df[(df.ObservationDate == '2020-03-10') & (df['Country/Region'] == 'Mainland China')]

latest_df = latest_df.groupby('Province/State').sum()



fig = go.Figure()

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Bar(

    y=latest_df.index,

    x=latest_df.Confirmed,

    name='Confirmed',

    orientation='h',

    marker=dict(

        color='rgba(246, 78, 139, 0.6)',

        line=dict(color='rgba(246, 78, 139, 1.0)', width=3)

    )

))

fig.add_trace(go.Bar(

    y=latest_df.index,

    x=latest_df.Deaths,

    name='Deaths',

    orientation='h',

    marker=dict(

        color='rgba(58, 71, 80, 0.6)',

        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)

    )

))



fig.update_layout(barmode='stack')

fig.update_layout(

    title="Number of Confirmed/Death/Recovered cases in various provinces across China",

    yaxis_title="Province names",

    xaxis_title="Number of cases",

    font=dict(

        family="Arial",

        size=12,

        color="white"

    ))

fig.show()
country_unique_df = df.groupby('ObservationDate')['Country/Region'].nunique()

country_unique_df = pd.DataFrame({'ObservationDate':country_unique_df.index, 'Country Number':country_unique_df.values})

country_unique_df





fig = go.Figure()

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Scatter(x=country_unique_df.ObservationDate, 

                         y=country_unique_df['Country Number'],

                         mode='lines+markers',

                         name='Number of unique countries infected with COVID-19',

                         line=dict(color='red', width=2)))





fig.update_layout(

    title="Number of unique countries infected with COVID-19",

    yaxis_title="Number of Countries",

    xaxis_title="Date",

    font=dict(

        family="Arial",

        size=16,

        color="white"

    ))

fig.show()
import folium

from folium import plugins



confirmed_df = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

confirmed_df

latest_confirmed_df = confirmed_df[['Province/State', 'Country/Region', 'Lat', 'Long', '3/10/20']]



m = folium.Map(location=[10, -20], zoom_start=2.6)

use_colours = ['orange','#d6604d','#b2182b','#67001f']



for lat, lon, value, country, province in zip(latest_confirmed_df['Lat'], latest_confirmed_df['Long'], latest_confirmed_df['3/10/20'], latest_confirmed_df['Country/Region'], latest_confirmed_df['Province/State']):

    if not province:

        popup = ('<strong>Country</strong>: ' + str(country).capitalize() + '<br>' '<strong>Confirmed Cases</strong>: ' + str(value) + '<br>')

    else:

        popup = ('<strong>Country</strong>: ' + str(country).capitalize() + '<br>' '<strong>Province</strong>: ' + str(province).capitalize() + '<br>' '<strong>Confirmed Cases</strong>: ' + str(value) + '<br>')        

    if value >= 5000:

        color = use_colours[3]

    elif value >= 1000:

        color = use_colours[2]

    elif value >= 500:

        color = use_colours[1]

    else:

        color = use_colours[0]

    folium.CircleMarker([lat, lon],

                        radius=10,

                        popup = popup,

                        color=color,

                        fill_color=color,

                        fill_opacity=0.7 ).add_to(m)

minimap = plugins.MiniMap()

m.add_child(minimap)

m

path='covid19map.html'

m.save(path)
confirmed_df = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

date_list = list(confirmed_df.columns)[4:]

final_date_list = []

for i in date_list:

    k = i.split('/')

    if len(k[1])==1:

        final_date_list.append('2020' + '-' + '0' + k[0] + '-' + '0' + k[1])

    else:

        final_date_list.append('2020' + '-' + '0' + k[0] + '-' + k[1])



final_date_list = ['Province/State','Country/Region','Lat', 'Long'] + final_date_list

confirmed_df.columns = final_date_list

confirmed_df
from datetime import datetime

import matplotlib.pylab as plt

from statsmodels.tsa.stattools import adfuller

from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.stattools import acf, pacf
confirmed_df = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

confirmed_df = confirmed_df.drop(labels=['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1)

confirmed_df = confirmed_df.T

confirmed_df['confirmed'] = confirmed_df.sum(axis=1)

confirmed_df = confirmed_df['confirmed']



train_size = int(len(confirmed_df) * 0.95)

train, test = confirmed_df[0:train_size], confirmed_df[train_size:]

plt.plot(train)

plt.xlabel('Date')

plt.ylabel('Cumulative number of confirmed cases')

plt.xticks(fontsize=8, rotation=90)

plt.title('Cumulative number of confirmed cases between January to March')

plt.show()
def test_stationarity(timeseries):

    #Determing rolling statistics

    rolmean = timeseries.rolling(window=7).mean()

    rolstd = timeseries.rolling(window=7).std()



    #Plot rolling statistics:

    

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.xticks(fontsize=10, rotation=90)

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)
test_stationarity(train)

# Since this dataset has a cumulative number of confirmed cases,

# both Rolling Mean and Rolling Standard Deviation are not constant.

# Hence dataset is not stationary
def difference(dataset, interval=1):

    index = list(dataset.index)

    diff = list()

    for i in range(interval, len(dataset)):

        value = dataset.values[i] - dataset.values[i - interval]

        diff.append(value)

    return (diff)
# Difference will remove the cumulative number of confirmed cases and only present the number of confirmed cases pe

diff = difference(train)



plt.plot(diff)

plt.xlabel('Number of days')

plt.ylabel('Number of confirmed cases')

plt.title('Differencing: Subtracting previous observation from the current observation')

plt.show()

# Differencing is a popular and widely used data transform for making time series data stationary.

# Differencing is performed by subtracting the previous observation from the current observation.
X = diff

result = adfuller(X)

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():

    print('\t%s: %.3f' % (key, value))

    

# We carry out Dickey-Fuller test, to test for stationarity in the dataset. 

# The results show that te test static value -4.480739 is smaller than the critical value at 5% of -2.924

# We can reject null hypothesis and conclude that differenced series is stationary.

# At least one level of differencing is required.
diff = difference(train)

diff_df = pd.DataFrame(diff)

train1 = pd.DataFrame(train)

train1['date'] = train1.index

diff_df.index = train1.date[1:]

diff_df



test_stationarity(diff_df)

# Since this dataset has a cumulative number of confirmed cases,

# both Rolling Mean and Rolling Standard Deviation are not constant.

# Hence dataset is not stationary
from statsmodels.tsa.seasonal import seasonal_decompose

diff = difference(train)

diff_df = pd.DataFrame(diff)

train1 = pd.DataFrame(train)

train1['date'] = train1.index

diff_df.index = train1.date[1:]



diff_df1 = pd.DataFrame(diff_df)

diff_df1.reset_index(inplace=True)

diff_df1['date'] = pd.to_datetime(diff_df1['date'])

diff_df1.index = diff_df1['date']

diff_df1 = diff_df1.drop(columns=['date'],axis=1)





decomposition = seasonal_decompose(diff_df1)





trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid



plt.subplot(411)

plt.plot(diff, label='Original')

plt.legend(loc='best')

plt.subplot(412)

plt.plot(trend, label='Trend')

plt.legend(loc='best')

plt.subplot(413)

plt.plot(seasonal,label='Seasonality')

plt.legend(loc='best')

plt.subplot(414)

plt.plot(residual, label='Residuals')

plt.legend(loc='best')

plt.show()



# taking the differenced dataset, there is still some trends of seasonality
from statsmodels.tsa.stattools import acf, pacf



lag_acf = acf(diff, nlags=20)

lag_pacf = pacf(diff, nlags=20, method='ols')



plt.subplot(121) 

plt.plot(lag_acf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(diff)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(diff)),linestyle='--',color='gray')

plt.title('Autocorrelation Function')

plt.show()



plt.subplot(122)

plt.plot(lag_pacf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(diff)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(diff)),linestyle='--',color='gray')

plt.title('Partial Autocorrelation Function')

plt.show()

plt.tight_layout()
import warnings

from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error

from math import sqrt



# evaluate an ARIMA model for a given order (p,d,q) and return RMSE

def evaluate_arima_model(X, arima_order):

    # prepare training dataset

    X = X.astype('float32')

    train_size = int(len(X) * 0.50)

    train, test = X[0:train_size], X[train_size:]

    history = [x for x in train]

    # make predictions

    predictions = list()

    for t in range(len(test)):

        model = ARIMA(history, order=arima_order)

        model_fit = model.fit(disp=0)

        yhat = model_fit.forecast()[0]

        predictions.append(yhat)

        history.append(test[t])

    # calculate out of sample error

    rmse = sqrt(mean_squared_error(test, predictions))

    return rmse



# evaluate combinations of p, d and q values for an ARIMA model

def evaluate_models(dataset, p_values, d_values, q_values):

    dataset = dataset.astype('float32')

    best_score, best_cfg = float("inf"), None

    for p in p_values:

        for d in d_values:

            for q in q_values:

                order = (p,d,q)

                try:

                    rmse = evaluate_arima_model(dataset, order)

                    if rmse < best_score:

                        best_score, best_cfg = rmse, order

                    print('ARIMA%s RMSE=%.3f' % (order,rmse))

                except:

                    continue

    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))



# evaluate parameters

diff = difference(train)

diff_df = pd.DataFrame(diff)



p_values = range(0,8)

d_values = range(0, 4)

q_values = range(0, 8)

warnings.filterwarnings("ignore")

evaluate_models(diff_df.values, p_values, d_values, q_values)
# evaluate an ARIMA model for a given order (p,d,q) and return AIC

def evaluate_arima_model_aic(X, arima_order):

    # prepare training dataset

    X = X.astype('float32')

    train_size = int(len(X) * 0.50)

    train, test = X[0:train_size], X[train_size:]

    history = [x for x in train]

    # make predictions

    predictions = list()

    for t in range(len(test)):

        model = ARIMA(history, order=arima_order)

        model_fit = model.fit(disp=0)

        yhat = model_fit.forecast()[0]

        predictions.append(yhat)

        history.append(test[t])

    # calculate out of sample error

    return model_fit.aic



# evaluate combinations of p, d and q values for an ARIMA model

def evaluate_models_aic(dataset, p_values, d_values, q_values):

    dataset = dataset.astype('float32')

    best_score, best_cfg = float("inf"), None

    for p in p_values:

        for d in d_values:

            for q in q_values:

                order = (p,d,q)

                try:

                    aic = evaluate_arima_model(dataset, order)

                    if aic < best_score:

                        best_score, best_cfg = aic, order

                    print('ARIMA%s AIC=%.3f' % (order,aic))

                except:

                    continue

    print('Best ARIMA%s AIC=%.3f' % (best_cfg, best_score))



# evaluate parameters

diff = difference(train)

diff_df = pd.DataFrame(diff)



p_values = range(0,8)

d_values = range(0, 4)

q_values = range(0, 8)

warnings.filterwarnings("ignore")

evaluate_models_aic(diff_df.values, p_values, d_values, q_values)
best_cfg = (0, 1, 0)

history = [float(x) for x in diff_df.values]

model = ARIMA(history, order=best_cfg)

model_fit = model.fit(disp=0)

print(model_fit.summary())
diff_test = difference(test)

diff_df_test = pd.DataFrame(diff_test)

test1 = pd.DataFrame(test)

test1['date'] = test1.index

diff_df_test.index = test1.date[1:]

diff_df_test





# walk-forward validation

best_cfg = (0, 1, 0)

history = [float(x) for x in diff_df.values]

predictions = list()

for i in range(len(diff_df_test)):

    # predict

    model = ARIMA(history, order=best_cfg)

    model_fit = model.fit(disp=0)

    yhat = model_fit.forecast()[0]

    predictions.append(yhat)

    # observation

    obs = diff_df_test.values[i]

    history.append(obs)

# errors

residuals = [diff_df_test.values[i]-predictions[i] for i in range(len(diff_df_test))]

residuals = pd.DataFrame(residuals)

plt.figure()

plt.subplot(211)

residuals.hist(ax=plt.gca())

plt.subplot(212)

residuals.plot(kind='kde', ax=plt.gca())

plt.show()
from statsmodels.tsa.arima_model import ARIMAResults

from scipy.stats import boxcox

from sklearn.metrics import mean_squared_error

from math import exp

from math import log

import numpy



history = [float(x) for x in diff_df.values]

# predict



def invert_difference(test, diff):

    pred = list()

    value = test[0]

    for i in range(0, len(diff)):

        value += float(diff.values[i])

        pred.append(value)

    return numpy.array(pred)





model = ARIMA(history, order=(0,1,0))

model_fit = model.fit()



yhat = model_fit.forecast(steps=1)[0]

yhat = yhat + test.values[0]

print('Predicted next day volume on 12th March = %i' % (yhat))