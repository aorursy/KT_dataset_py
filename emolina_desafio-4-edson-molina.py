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
# Importando as bibliotecas

import pandas as pd

import numpy as np

from scipy import stats

import statsmodels.api as sm

from fbprophet import Prophet

from plotly.offline import init_notebook_mode, iplot

from plotly import graph_objs as go



# Initialize plotly

init_notebook_mode(connected=True)



import matplotlib.pyplot as plt

%matplotlib inline



import logging

logging.getLogger().setLevel(logging.ERROR)
# Carregando os dados

train = pd.read_csv("/kaggle/input/web-traffic-time-series-forecasting/train_1.csv")

keys = pd.read_csv("/kaggle/input/web-traffic-time-series-forecasting/key_1.csv")

ss = pd.read_csv("/kaggle/input/web-traffic-time-series-forecasting/sample_submission_1.csv")
train.head()
train.info()
# Drop Page column

X_train = train.drop(['Page'], axis=1)

X_train.head()
y = X_train.as_matrix()[0]

df = pd.DataFrame({ 'ds': X_train.T.index.values, 'y': y})
df.head()
# %load solutions/solution_08.py

df['ds'] = pd.to_datetime(df['ds']).dt.date
prediction_size = 30

train_df = df[:-prediction_size]

train_df.tail(n=3)
# %load solutions/solution_09.py



m = Prophet()

m.fit(train_df)

future = m.make_future_dataframe(periods=prediction_size)

forecast = m.predict(future)
forecast.tail(n=3)
m.plot(forecast)
# %load solutions/solution_10.py

m.plot_components(forecast)
print(', '.join(forecast.columns))
def make_comparison_dataframe(historical, forecast):

    """Join the history with the forecast.

    

       The resulting dataset will contain columns 'yhat', 'yhat_lower', 'yhat_upper' and 'y'.

    """

    return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical.set_index('ds'))
cmp_df = make_comparison_dataframe(df, forecast)

cmp_df.tail()
def calculate_forecast_errors(df, prediction_size):

    """Calculate MAPE and MAE of the forecast.

    

       Args:

           df: joined dataset with 'y' and 'yhat' columns.

           prediction_size: number of days at the end to predict.

    """

    

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
for err_name, err_value in calculate_forecast_errors(cmp_df, prediction_size).items():

    print(err_name, err_value)
def show_forecast(cmp_df, num_predictions, num_values, title):

    """Visualize the forecast."""

    

    def create_go(name, column, num, **kwargs):

        points = cmp_df.tail(num)

        args = dict(name=name, x=points.index, y=points[column], mode='lines')

        args.update(kwargs)

        return go.Scatter(**args)

    

    lower_bound = create_go('Lower Bound', 'yhat_lower', num_predictions,

                            line=dict(width=0),

                            marker=dict(color="blue"))

    upper_bound = create_go('Upper Bound', 'yhat_upper', num_predictions,

                            line=dict(width=0),

                            marker=dict(color="black"),

                            fillcolor='rgba(68, 68, 68, 0.3)', 

                            fill='tonexty')

    forecast = create_go('Forecast', 'yhat', num_predictions,

                         line=dict(color='rgb(31, 119, 180)'))

    actual = create_go('Actual', 'y', num_values,

                       marker=dict(color="red"))

    

    # In this case the order of the series is important because of the filling

    data = [lower_bound, upper_bound, forecast, actual]



    layout = go.Layout(yaxis=dict(title='Posts'), title=title, showlegend = False)

    fig = go.Figure(data=data, layout=layout)

    iplot(fig, show_link=False)



show_forecast(cmp_df, prediction_size, 100, 'New Traficc WIKI')
keys.head()
y_pred = pd.DataFrame()

y_pred['ID_code'] = keys ['Id']
y_pred.head()
y_pred.to_csv("sample_submission_1.csv",index=False)