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
import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fbprophet import Prophet

	

# Default value of display.max_rows is 10 i.e. at max 10 rows will be printed.

# Set it None to display all rows in the dataframe

pd.set_option('display.max_rows', None)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")

test_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")

submission_csv = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/submission.csv")

train_df.head()
train_df['Date']=pd.to_datetime(train_df.Date)

gr_df_conf_cases = train_df.groupby(['Date'])['ConfirmedCases'].sum().reset_index() 



gr_df_conf_cases.set_index('Date')

gr_df_conf_cases = gr_df_conf_cases.astype({"ConfirmedCases": float})

gr_df_conf_cases.rename(columns={'ConfirmedCases':'y','Date':'ds'}, inplace=True)

m=Prophet(daily_seasonality=False)

m.fit(gr_df_conf_cases)

future = m.make_future_dataframe(periods=120)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)



# Below would be the prediction for confirmed cases 
m.plot_components(forecast);
train_df['Date']=pd.to_datetime(train_df.Date)

gr_df_fatal_cases = train_df.groupby(['Date'])['Fatalities'].sum().reset_index() 



gr_df_fatal_cases.set_index('Date')

gr_df_fatal_cases = gr_df_fatal_cases.astype({"Fatalities": float})

gr_df_fatal_cases.rename(columns={'Fatalities':'y','Date':'ds'}, inplace=True)

m=Prophet(daily_seasonality=False)

m.fit(gr_df_fatal_cases)

future = m.make_future_dataframe(periods=120)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)



# Below would be the prediction for Fatal cases 
m.plot_components(forecast);