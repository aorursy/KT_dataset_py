import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

from scipy import stats

import statsmodels.api as sm

import matplotlib.pyplot as plt



%matplotlib inline
df = pd.read_csv('../input/medium_posts.csv')
df = df[['published', 'url']].dropna().drop_duplicates()
df['published'] = pd.to_datetime(df['published'])
df.sort_values(by=['published']).head(n=3)
df = df[(df['published'] > '2012-08-15') & (df['published'] < '2017-06-26')].sort_values(by=['published'])

df.head(n=3)
df.tail(n=3)
aggr_df = df.groupby('published')[['url']].count()

aggr_df.columns = ['posts']
aggr_df.head(n=3)
daily_df = aggr_df.resample('D').apply(sum)

daily_df.head(n=3)
from plotly.offline import init_notebook_mode, iplot

from plotly import graph_objs as go



# Initialize plotly

init_notebook_mode(connected=True)
def plotly_df(df, title=''):

    """Visualize all the dataframe columns as line plots."""

    common_kw = dict(x=df.index, mode='lines')

    data = [go.Scatter(y=df[c], name=c, **common_kw) for c in df.columns]

    layout = dict(title=title)

    fig = dict(data=data, layout=layout)

    iplot(fig, show_link=False)
plotly_df(daily_df, title='Posts on Medium (daily)')
weekly_df = daily_df.resample('W').apply(sum)
plotly_df(weekly_df, title='Posts on Medium (weekly)')
daily_df = daily_df.loc[daily_df.index >= '2015-01-01']

daily_df.head(n=3)
from fbprophet import Prophet



import logging

logging.getLogger().setLevel(logging.ERROR)
df = daily_df.reset_index()

df.columns = ['ds', 'y']

df.tail(n=3)
prediction_size = 30

train_df = df[:-prediction_size]

train_df.tail(n=3)
m = Prophet()

m.fit(train_df);
future = m.make_future_dataframe(periods=prediction_size)

future.tail(n=3)
forecast = m.predict(future)

forecast.tail(n=3)
m.plot(forecast);
m.plot_components(forecast);
print(', '.join(forecast.columns))
def make_comparison_dataframe(historical, forecast):

    """Join the history with the forecast.

    

       The resulting dataset will contain columns 'yhat', 'yhat_lower', 'yhat_upper' and 'y'.

    """

    return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical.set_index('ds'))
cmp_df = make_comparison_dataframe(df, forecast)

cmp_df.tail(n=3)
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



    layout = go.Layout(yaxis=dict(title='Posts'), title=title, showlegend = False)

    fig = go.Figure(data=data, layout=layout)

    iplot(fig, show_link=False)



show_forecast(cmp_df, prediction_size, 100, 'New posts on Medium')
def inverse_boxcox(y, lambda_):

    return np.exp(y) if lambda_ == 0 else np.exp(np.log(lambda_ * y + 1) / lambda_)
train_df2 = train_df.copy().set_index('ds')
train_df2['y'], lambda_prophet = stats.boxcox(train_df2['y'])

train_df2.reset_index(inplace=True)
m2 = Prophet()

m2.fit(train_df2)

future2 = m2.make_future_dataframe(periods=prediction_size)

forecast2 = m2.predict(future2)
for column in ['yhat', 'yhat_lower', 'yhat_upper']:

    forecast2[column] = inverse_boxcox(forecast2[column], lambda_prophet)
cmp_df2 = make_comparison_dataframe(df, forecast2)

for err_name, err_value in calculate_forecast_errors(cmp_df2, prediction_size).items():

    print(err_name, err_value)
show_forecast(cmp_df, prediction_size, 100, 'No transformations')

show_forecast(cmp_df2, prediction_size, 100, 'Box–Cox transformation')