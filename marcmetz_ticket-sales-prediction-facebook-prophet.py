# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Additional imports

from fbprophet import Prophet
"""

# Los Angeles event

df_LA = pd.read_csv(

    '../input/orders_order_LA.csv',

    usecols=['created', 'total_gross', 'event_id'],

    parse_dates=['created'],

)



df_LA = df_LA.rename(columns={'created': 'ds'})

df_LA['ds'] = df_LA['ds'] - pd.DateOffset(months=3)

# df_LA['ds'] = df_LA['ds'].dt.tz_convert(None)  # Remove timezone to avoid ValueError in Prophet 

df_LA.head()

"""
# max_total_gross_LA = 71000  # Will be calculated in v2
"""

# Prepare df['y']

df_LA['total_gross_cumsum'] = df_LA['total_gross'].cumsum()

df_LA['y'] = df_LA['total_gross_cumsum']/(max_total_gross_LA*100)  # Transform to % of max_total_gross in Cents (*100)

df_LA.head()

"""
# Wolfsburg event

df_WOB = pd.read_csv(

    '../input/orders_order_WOB.csv',

    usecols=['created', 'total_gross', 'event_id'],

    parse_dates=['created'],

)



# Prepare df['ds']

df_WOB = df_WOB.rename(columns={'created': 'ds'})

# df_WOB['ds'] = df_WOB['ds'].dt.tz_convert(None)  # Remove timezone to avoid ValueError in Prophet 

df_WOB.head()
max_total_gross_WOB = 90000  # Will be calculated in v2
# Prepare df['y']

df_WOB['total_gross_cumsum'] = df_WOB['total_gross'].cumsum()

df_WOB['y'] = df_WOB['total_gross_cumsum']/(max_total_gross_WOB*100)  # Transform to % of max_total_gross in Cents (*100)

df_WOB.head()
# Combine DataFrames (Wolfsburg & LA event)

# ignore_index: If True, do not use the index values along the concatenation axis.

# df = pd.concat((df_LA, df_WOB), ignore_index=True)

df = df_WOB

df.head()
# Load ticket data

def load_ticket_data():

    df = pd.read_csv(

        '../input/tickets_ticket_WOB.csv',

        usecols=['created', 'start_at'],

        parse_dates=['created', 'start_at'],

    )

    return df



tickets = load_ticket_data()

tickets.head()
tickets['created'] = tickets['created'].dt.date

tickets['start_at'] = tickets['start_at'].dt.date



# Note: We use to_datetime in the next cell to convert python objects to a datetime objects
# Treat ticket releases as holidays

ticket_start_at = pd.DataFrame({

  'holiday': 'ticket_release',

  'ds': pd.to_datetime(tickets['start_at']),

  'lower_window': 0,

  'upper_window': 1,  # Extend the holiday out to [upper_window] days around the date

})

ticket_peaks = ticket_start_at



"""

Further improvements:

What if ticket released right away (no start_at)



ticket_created = pd.DataFrame({

  'holiday': 'ticket_release',

  'ds': pd.to_datetime(tickets['created']),

  'lower_window': 0,

  'upper_window': 1,  # Extend the holiday out to [upper_window] days around the date

})

ticket_peaks = pd.concat((ticket_start_at, ticket_created))

"""
# Fit algorithm 

periods = 137



m = Prophet(growth='linear', holidays=ticket_peaks)

df['cap'] = 1



m.fit(df)



# Try help(m)
future = m.make_future_dataframe(periods=periods)

future['cap'] = 1

future.tail()



# Try help(m.make_future_dataframe)
forecast = m.predict(future)

forecast.tail()
# Show ticket release effect

forecast[forecast['ticket_release'].abs() > 0][['ds', 'ticket_release']][-10:]
predicted_date = '2019-06-25 13:34:45.400553'

yhat_percentage = forecast[['yhat']].loc[forecast['ds'].isin([predicted_date])]

yhat = yhat_percentage * max_total_gross_WOB

yhat
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# Python

fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
# Cross validation

from fbprophet.diagnostics import cross_validation



horizon = '14 days'  # Approx two weeks

df_cv = cross_validation(m, horizon=horizon)

df_cv.head()
from fbprophet.diagnostics import performance_metrics

df_p = performance_metrics(df_cv)

df_p.head()
from fbprophet.plot import plot_cross_validation_metric

fig = plot_cross_validation_metric(df_cv, metric='mape')