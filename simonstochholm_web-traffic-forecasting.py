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
from fbprophet import Prophet
train = pd.read_csv("/kaggle/input/web-traffic-time-series-forecasting/train_1.csv.zip")
train.head()
all_pages = train['Page']
first_page = all_pages[0]

first_page
train_allT = train.set_index('Page').T.reset_index().rename(columns={'index':'Date'})

train_allT.head()
df = pd.DataFrame(train_allT, columns = ['Date',first_page]) 

df = df.rename(columns={'Date':'ds', first_page:'y'})
df.head()
m = Prophet()

m.fit(df)
future = m.make_future_dataframe(periods=60)

future.head()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
result = forecast['yhat']
result
sub_result = result[-60:]
sub_result
sub_result.shape
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
key = pd.read_csv("/kaggle/input/web-traffic-time-series-forecasting/key_1.csv.zip")
key
listOfIds = key.index[key['Page'].str.contains(first_page)].values
listOfIds.shape
sub = pd.read_csv("/kaggle/input/web-traffic-time-series-forecasting/sample_submission_1.csv.zip")
sub
sub['Visits'].loc[listOfIds.min():listOfIds.max()] = sub_result.values.round()

#sub['Visits'].loc[0:59] = sub_result.values.round()
sub
def my_function():

    for page in all_pages:

        #print(page)

        df = pd.DataFrame(train_allT, columns = ['Date',page]) 

        df = df.rename(columns={'Date':'ds', page:'y'})

        m = Prophet()

        m.fit(df)

        future = m.make_future_dataframe(periods=60)

        forecast = m.predict(future)

        result = forecast['yhat']

        sub_result = result[-60:]

        listOfIds = key.index[key['Page'].str.contains(page)].values

        sub['Visits'].loc[listOfIds.min():listOfIds.max()] = sub_result.values.round()

        #print(sub)
#my_function()
#sub.to_csv('submission.csv', index=False)
import numpy as np

import pandas as pd



print('Reading data...')

key_1 = pd.read_csv('/kaggle/input/web-traffic-time-series-forecasting/key_2.csv.zip')

train_1 = pd.read_csv('/kaggle/input/web-traffic-time-series-forecasting/train_2.csv.zip')

ss_1 = pd.read_csv('/kaggle/input/web-traffic-time-series-forecasting/sample_submission_2.csv.zip')



print('Preprocessing...')

# train_1.fillna(0, inplace=True)



print('Processing...')

ids = key_1.Id.values

pages = key_1.Page.values



print('key_1...')

d_pages = {}

for id, page in zip(ids, pages):

    d_pages[id] = page[:-11]

   

    



print('train_1...')

pages = train_1.Page.values

# visits = train_1['2016-12-31'].values # Version 1 score: 60.6

# visits = np.round(np.mean(train_1.drop('Page', axis=1).values, axis=1)) # Version 2 score: 64.8

# visits = np.round(np.mean(train_1.drop('Page', axis=1).values[:, -14:], axis=1)) # Version 3 score: 52.5

# visits = np.round(np.mean(train_1.drop('Page', axis=1).values[:, -7:], axis=1)) # Version 4 score: 53.7

# visits = np.round(np.mean(train_1.drop('Page', axis=1).values[:, -21:], axis=1)) # Version 5, 6 score: 51.3

# visits = np.round(np.mean(train_1.drop('Page', axis=1).values[:, -28:], axis=1)) # Version 7 score: 51.1

# visits = np.round(np.median(train_1.drop('Page', axis=1).values[:, -28:], axis=1)) # Version 8 score: 47.1 

# visits = np.round(np.median(train_1.drop('Page', axis=1).values[:, -35:], axis=1)) # Version 9 score: 46.6

# visits = np.round(np.median(train_1.drop('Page', axis=1).values[:, -42:], axis=1)) # Version 10 score: 46.3

# visits = np.round(np.median(train_1.drop('Page', axis=1).values[:, -49:], axis=1)) # Version 11 score: 46.2

# visits = np.nan_to_num(np.round(np.nanmedian(train_1.drop('Page', axis=1).values[:, -49:], axis=1))) # Version 12 score: 45.7

visits = np.nan_to_num(np.round(np.nanmedian(train_1.drop('Page', axis=1).values[:, -56:], axis=1))) # scorer 41.8 #find medianen de sidste 56 dage og skift nan ud med 0



d_visits = {}

for page, visits_number in zip(pages, visits):

    d_visits[page] = visits_number

    # for hver page i pages og visit i visits gem antal visits på page



print('Modifying sample submission...') # læs submissionfilen ind

ss_ids = ss_1.Id.values

ss_visits = ss_1.Visits.values
d_visits
d_pages
ss_ids




for i, ss_id in enumerate(ss_ids):

    ss_visits[i] = d_visits[d_pages[ss_id]] #sæt første ss_id i d_pages-listen for at finde page-navn. sæt så page-navn i d_visits for at finde tal-værdien, 

    #som gemmes i stedet for ss_tal-værdien inkrementalt.



print('Saving submission...')

subm = pd.DataFrame({'Id': ss_ids, 'Visits': ss_visits})

subm.to_csv('submission.csv', index=False)
import warnings

warnings.filterwarnings('ignore')
print('Pre-processing and feature engineering train data...')

train_flattened = pd.melt(train[list(train.columns[-49:])+['Page']], id_vars='Page', var_name='date', value_name='Visits')

train_flattened['date'] = train_flattened['date'].astype('datetime64[ns]')

train_flattened['weekend'] = ((train_flattened.date.dt.dayofweek) // 5 == 1).astype(float)

# Visualisation

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(50, 8))

mean_group = train_flattened[['Page','date','Visits']].groupby(['date'])['Visits'].mean()

plt.plot(mean_group)

plt.title('Time Series - Average')

plt.show()

plt.close()
times_series_means =  pd.DataFrame(mean_group).reset_index(drop=False)
df_date_index = times_series_means[['date','Visits']].set_index('date')
from statsmodels.tsa.stattools import adfuller

# Run Dicky-Fuller test

result = adfuller(df_date_index)



# Print test statistic

print(result[0])



# Print p-value

print(result[1])
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Create figure

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))

 

# Plot the ACF of savings on ax1

plot_acf(df_date_index, zero=False, ax=ax1, lags=10)



# Plot the PACF of savings on ax2

plot_pacf(df_date_index, zero=False, ax=ax2, lags=10)



plt.show()

plt.close()
# Create empty list to store search results

order_aic_bic=[]



# Loop over p values from 0-2

for p in range(3):

  # Loop over q values from 0-2

    for q in range(3):

        try:

            # create and fit ARMA(p,q) model

            model = SARIMAX(df_date_index, order=(p,0,q), seasonal_order=(1,2,0,7))

            results = model.fit()

           



            # Append order and results tuple

            order_aic_bic.append((p,q, results.aic, results.bic))

            print(p,q,results.aic, results.bic)

            

        except:

            print(p, q, None, None)
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Create and fit model

model = SARIMAX(df_date_index, order=(2,0,1), trend='c')

results = model.fit()



# Create the 4 diagostics plots

results.plot_diagnostics()

plt.show()

plt.close()



# Print summary

print(results.summary())

# Import seasonal decompose

from statsmodels.tsa.seasonal import seasonal_decompose



# Perform additive decomposition

decomp = seasonal_decompose(df_date_index, 

                            freq=7)



# Plot decomposition

decomp.plot()

plt.show()

plt.close()
!pip install pmdarima
import pmdarima as pm
# Create auto_arima model

model1 = pm.auto_arima(df_date_index,

                      seasonal=True, m=7,

                      d=0, D=1, 

                 	  max_p=2, max_q=2,

                      trace=True,

                      error_action='ignore',

                      suppress_warnings=True)

                       

# Print model summary

print(model1.summary())
# Import model class

from statsmodels.tsa.statespace.sarimax import SARIMAX



# Create model object

model = SARIMAX(df_date_index, 

                order=(2,0,1), 

                seasonal_order=(1,1,1,7), 

                trend='c')

# Fit model

results = model.fit()
# Plot common diagnostics

results.plot_diagnostics()

plt.show()

plt.close()
# Create forecast object

forecast_object = results.get_forecast(steps=90)



# Extract prediction mean

mean = forecast_object.predicted_mean



# Extract the confidence intervals

conf_int = forecast_object.conf_int()



# Extract the forecast dates

dates = mean.index


df_date_index.index = pd.to_datetime(df_date_index.index)
# Print last predicted mean

print(mean.iloc[-1])



# Print last confidence interval

print(conf_int.iloc[-1])
## Validating Forecast

pred = results.get_prediction(start=pd.to_datetime('2016-12-01'), dynamic=False)

pred_ci = pred.conf_int()

ax = df_date_index['2016':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')

ax.set_ylabel('Sales')

plt.legend()
y_forecasted = pred.predicted_mean

y_truth = df_date_index['2016-10-01':]

mse = ((y_forecasted - y_truth) ** 2).mean()

print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

#The MSE is a measure of the quality of an estimator — it is always non-negative, 

#and the smaller the MSE, the closer we are to finding the line of best fit.
pred_uc = results.get_forecast(steps=100)

pred_ci = pred_uc.conf_int()

ax = df_date_index.plot(label='observed', figsize=(14, 7))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')

ax.set_ylabel('Sales')

plt.legend()