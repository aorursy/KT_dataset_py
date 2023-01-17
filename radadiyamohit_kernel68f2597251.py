# importing some library which help to perform vasic operations on dataset.

import pandas as pd

import numpy as np

from scipy import stats

import statsmodels.api as sm

import matplotlib.pyplot as plt



%matplotlib inline
# reading the dataset using pandas library

df = pd.read_csv('../input/into-the-future/train.csv')
# printing first five rows in out dataset

df.head()
# It is used for calculating some statistical data like percentile, mean and std of the numerical values of the Series or DataFrame

df.describe()
# It is used to print a concise summary of our DataFrame.

df.info()
# we not need to convert time to datetime because our dataset already in same format which i want.

df['time'] = pd.to_datetime(df['time'])



# deleting the unusable column from our dataset.

df.drop('id',axis=1,inplace=True)



# set_index is used to set the DataFrame index using existing columns.

df.set_index('time',inplace=True)
# printing last five rows of our dataset.

df.tail()
# printing the shape of our dataset

print(df.shape)



# graphical representation of feature_1 in our dataset same thing will do in feature_2.

plt.plot(df['feature_1'])
plt.plot(df['feature_2'])
from fbprophet import Prophet



import logging

logging.getLogger().setLevel(logging.ERROR)
data = df.reset_index()

data.tail(n=3)
data2 = data[['time','feature_2']].reset_index()

data2.drop('index',axis=1,inplace=True)

data2.columns = ['ds', 'y']

data2.tail(3)
# train test

prediction_size = 30

train_df2 = data2[:-prediction_size]

train_df2.tail()
# initialize Prophet

m = Prophet()



# point towards dataframe

m.fit(train_df2)
# set future prediction window of prediction_size

future = m.make_future_dataframe(periods=435, freq='10S')

#future = m.make_future_dataframe(periods=prediction_size)



# preview our data -- note that Prophet is only showing future dates (not values), as we need to call the prediction method still

future.tail(n=3)
forecast = m.predict(future)



# This will printing last five rows after forecasting

forecast.tail(3)
# If you want to show only usefull columns then..

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)

fig1
fig2 = m.plot_components(forecast)

fig2
print(', '.join(forecast.columns))
def make_comparison_dataframe(historical, forecast):

    """Join the history with the forecast.

    

       The resulting dataset will contain columns 'yhat', 'yhat_lower', 'yhat_upper' and 'y'.

    """

    return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical.set_index('ds'))
cmp_df = make_comparison_dataframe(data2, forecast[534:563])

cmp_df.tail(n=3)

len(cmp_df)
fcast = forecast[534:563]['yhat']

fcast.head()
def calculate_forecast_errors(df, prediction_size):

    # Make a copy

    df = df.copy()

    

    # Now we calculate the values of e_i and p_i according to the formulas given in the article above.

    df['e'] = df['y'] - df['yhat']

    df['p'] = 100 * df['e'] / df['y']

    

    # Recall that we held out the values of the last `prediction_size` days

    # in order to predict them and measure the quality of the model. 

    

    # Now cut out the part of the data which we made our prediction for.

    predicted_part = df[-prediction_size:]

    

    # Define the function that averages absolute error values over the predicted part.

    error_mean = lambda error_name: np.mean(np.abs(predicted_part[error_name]))

    

    # Now we can calculate MAPE and MAE and return the resulting dictionary of errors.

    return {'MAPE': error_mean('p'), 'MAE': error_mean('e')}

# MAE(mean absolute error) is absolute error is the absolute value of the difference between the forecasted value and the actual value.

# MAPE(mean absolute percentage error) is allows us to compare forecasts of different series in different scales.

for err_name, err_value in calculate_forecast_errors(cmp_df, prediction_size).items():

    print(err_name, err_value)
from plotly.offline import init_notebook_mode, iplot

from plotly import graph_objs as go



# Initialize plotly

init_notebook_mode(connected=True)
def show_forecast(cmp_df, num_predictions, num_values, title):

    """Visualize the forecast."""

    

    def create_go(name, column, num, **kwargs):

        points = cmp_df.tail(num)

        args = dict(name=name, x=points.index, y=points[column], mode='lines')

        args.update(kwargs)

        return go.Scatter(**args)

    

    lower_bound = create_go('Lower Bound', 'yhat_lower', num_predictions,

                            line=dict(width=0),

                            marker=dict(color="gray"))

    upper_bound = create_go('Upper Bound', 'yhat_upper', num_predictions,

                            line=dict(width=0),

                            marker=dict(color="gray"),

                            fillcolor='rgba(68, 68, 68, 0.3)', 

                            fill='tonexty')

    forecast = create_go('Forecast', 'yhat', num_predictions,

                         line=dict(color='rgb(31, 119, 180)'))

    actual = create_go('Actual', 'y', num_values,

                       marker=dict(color="red"))

    

    # In this case the order of the series is important because of the filling

    data = [lower_bound, upper_bound, forecast, actual]



    layout = go.Layout(yaxis=dict(title='features'), title=title, showlegend = False)

    fig = go.Figure(data=data, layout=layout)

    iplot(fig, show_link=False)



show_forecast(cmp_df, prediction_size, 100, 'Visualization')
test = pd.read_csv('../input/into-the-future/test.csv')

len(test)
d = forecast[594:]['yhat']

len(d)
# generating new dataframe for final outcome

final = pd.DataFrame()



final['id'] = test['id']

final['feature_2'] = list(d)
# printing first five rows of our predicted data

final.head()
final.tail()
# finally save that final dataframe into the CSV file using "to_csv" function.

final.to_csv("/kaggle/working/solution.csv", index=False)