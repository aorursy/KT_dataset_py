# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# lists all files in input directory

import pandas as pd

import numpy as np

import fbprophet

from fbprophet import Prophet

from fbprophet.plot import plot_plotly

from fbprophet.plot import add_changepoints_to_plot

import datetime

import os

import holidays

import seaborn as sns

from sklearn import metrics

from sklearn.metrics import mean_squared_error

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# function for seaborn linear regression

def plot_regression(original,forecast):

    regression_data=list(zip(original,forecast))

    regression_df=pd.DataFrame(regression_data,\

                                 columns=['original','forecast'])

    sns.jointplot('original','forecast',data=regression_df,kind='reg')

    print('the root mean square error is:',\

          np.sqrt(metrics.mean_squared_error(original, forecast)))
df=pd.read_csv('/kaggle/input/population-time-series-data/POP.csv')

ind = int(len(df)*0.8)

df.head(3)
df = df[['date','value']]

df.columns=['ds','y']

df['ds']=df['ds'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))

train_set,test_set = df.iloc[:ind],df.iloc[ind:]

train_set.head()
holidays_df = pd.DataFrame([],columns = ['ds','holiday'])

dates, names = [],[]

for date, name in sorted(holidays.US(years=np.arange(1952,2021)).items()):

    dates.append(date)

    names.append(name)

holidays_df['ds'] = dates

holidays_df['holiday'] = names

holidays_df.head()
yearly_seasonality = list(range(0,20))

holidays_prior_scale = [0.01,0.025, 0.05, 0.1,0.5,0.75, 1, 2, 5, 7, 10]

changepoint_prior_scale = [0.01,0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.5, 5]

trainrmse, testrmse, params = [],[],[]

for ys in yearly_seasonality:

    for hps in holidays_prior_scale:

        for cps in changepoint_prior_scale:

            model = Prophet(holidays=holidays_df, yearly_seasonality=ys, holidays_prior_scale=hps, changepoint_prior_scale=cps)

            model.fit(train_set)

            future = model.make_future_dataframe(periods=len(test_set),freq='1M')

            future.tail()

            forecast=model.predict(future)

            train_original=train_set['y'].values.tolist()

            train_forecast=forecast['yhat'].iloc[:ind].values.tolist()

            test_original=test_set['y'].values.tolist()

            test_forecast=forecast['yhat'].iloc[ind:].values.tolist()

            train_rmse = np.sqrt(metrics.mean_squared_error(train_original, train_forecast))

            test_rmse = np.sqrt(metrics.mean_squared_error(test_original, test_forecast))

            trainrmse.append(train_rmse)

            testrmse.append(test_rmse)

            parameters = '{},{},{}'.format(str(ys), str(hps), str(cps))

            params.append(parameters)

min_error = test_rmse.index(min(test_rmse))

print(params[min_error])
params=params[min_error].split(',')

yearly_seasonality=params[0]

holidays_prior_scale=params[1]

changepoint_prior_scale=params[2]
# instantiate the model

model = Prophet(holidays=holidays_df, yearly_seasonality=yearly_seasonality, changepoint_prior_scale=changepoint_prior_scale, holidays_prior_scale=holidays_prior_scale)
# fit the model to the training data

model.fit(train_set)
# create future dataframe

future = model.make_future_dataframe(periods=len(test_set),freq='1M')

future.tail()
forecast=model.predict(future)
forecast.head()
fig = model.plot(forecast)

# adds changepoints (red dotted lines show a change in trend)

a = add_changepoints_to_plot(fig.gca(), model, forecast)
# plotting components

components = model.plot_components(forecast)
# values.tolist() takes a dataframe column and makes into a normal python list

train_original=train_set['y'].values.tolist()

train_forecast=forecast['yhat'].iloc[:ind].values.tolist()
# the plot_regression function bascially makes a new dataframe from the lists (train_original, train_forecast) above

plot_regression(train_original, train_forecast)
# values.tolist() takes a dataframe column and makes into a normal python list

test_original=test_set['y'].values.tolist()

test_forecast=forecast['yhat'].iloc[ind:].values.tolist()
plot_regression(test_original, test_forecast)