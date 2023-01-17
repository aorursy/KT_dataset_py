import pandas as pd

import numpy as np

from fbprophet import Prophet

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/for-simple-exercises-time-series-forecasting/Miles_Traveled.csv')
df.head()
df.info()
# Renaming the columns of dataframe as per the convention of Prophet library

df.columns = ['ds', 'y']

df.head()
# Conveting the type of ds column to datetime format

df['ds'] = pd.to_datetime(df['ds'])

df.head()
# Plotting to see the actual behaviour of datasets

df.plot(x='ds', y='y')
# finding the number of rows in the dataset

len(df)
# Splitting the dataset into train and test (test is for one year(12 months -> 12 rows))

train = df.iloc[:576]

test = df.iloc[576:]
# Create an instance of Prophet

m = Prophet()

# Fit the training data

m.fit(train)

# Create a future dataframe 

future = m.make_future_dataframe(periods=12, freq='MS') # for daily data no need to specify freq

# making predictions

forecast = m.predict(future)
forecast.shape
forecast.iloc[-12:,]
forecast.tail()
# Plotting the predicted values against the original value

ax = forecast.plot(x='ds', y='yhat', label='Predictions', legend=True, figsize=(12,8))

test.plot(x='ds', y='y', label='True Test Data', legend=True, ax=ax, xlim=('2018-01-01', '2019-01-01'))
# Calculate the errors

from statsmodels.tools.eval_measures import rmse
predictions = forecast.iloc[-12:]['yhat']
predictions
test['y']
# Calculating rmse values

rmse(predictions, test['y'])
test.mean()
from fbprophet.diagnostics import cross_validation, performance_metrics

from fbprophet.plot import plot_cross_validation_metric
# initial training period

initial = 5 * 365

initial = str(initial) + ' days' # as Prophet requires string code



# period length for which we are gonna perform cross validation

period = 5 * 365

period = str(period) + ' days' # as Prophet requires string code



# horizon of prediction for each fold

# we'll forecast one year ahead

horizon = 365

horizon = str(horizon) + ' days'
df_cv = cross_validation(m, initial=initial, period=period, horizon=horizon)
df_cv.head()
len(df_cv)
performance_metrics(df_cv)
df.head(2)
df.tail(2)
plot_cross_validation_metric(df_cv, metric='rmse');