import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set()

from statsmodels.graphics.tsaplots import plot_acf

import plotly.offline as py
import plotly.graph_objs as go
from plotly import tools
py.init_notebook_mode(connected=True)

#############

import statsmodels.api as sm
from fbprophet import Prophet

#############

import os
print(os.listdir("../input"))
dtype_dict={"id":np.uint32,
            "store_nbr":np.uint8,
            "item_nbr":np.uint32,
            "unit_sales":np.float32,
            "perishable":np.uint8,
            "class":np.uint32,
            "cluster":np.uint8,
            "transactions":np.uint32
           }

iter_csv = pd.read_csv("../input/train.csv", iterator=True, chunksize=40000000, usecols=["date", "store_nbr", "item_nbr", "unit_sales"], dtype=dtype_dict, parse_dates=['date'])
df = pd.concat([chunk[chunk['store_nbr'] == 44] for chunk in iter_csv])
#df = pd.read_csv("output.csv", usecols=["date", "store_nbr", "item_nbr", "unit_sales"], dtype=dtype_dict, parse_dates=['date'])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["week"] = df["date"].dt.week
df["day_week"] = df["date"].dt.weekday_name
df["day_month"] = df["date"].dt.day

df_2013 = df[df["year"] == 2013]

items = pd.read_csv("../input/items.csv", dtype=dtype_dict)
#items["family"] = items["family"].map(lambda x: x.lower().replace("i", "I"))

stores = pd.read_csv("../input/stores.csv", dtype=dtype_dict) 
holidays = pd.read_csv("../input/holidays_events.csv", dtype=dtype_dict, parse_dates=['date'])
transactions = pd.read_csv("../input/transactions.csv", dtype=dtype_dict, parse_dates=['date'])
oil = pd.read_csv("../input/oil.csv", dtype=dtype_dict, parse_dates=['date'])

'ok'
df_sample = df.sample(1000 * 1000)
df_sample = pd.merge(left=df_sample, right=items, how="left", on="item_nbr")

sales = df_sample.groupby('date').size()
x_sliding_window = pd.DataFrame(data={'x1':sales.values[:-1], 'x_date': sales.keys()[1:]})
x_sliding_window["x_year"] = x_sliding_window["x_date"].dt.year
x_sliding_window["x_month"] = x_sliding_window["x_date"].dt.month
x_sliding_window["x_week"] = x_sliding_window["x_date"].dt.week
x_sliding_window["x_day_week"] = x_sliding_window["x_date"].dt.weekday #Monday=0, Sunday=6
x_sliding_window["x_day_month"] = x_sliding_window["x_date"].dt.day
y_sliding_window = pd.DataFrame(data={'y':sales.values[1:]})

'ok'
# y_sliding_window

y_sliding_window['y'].plot('line')
# armax = sm.tsa.ARMA(y_sliding_window.values, order=(1, 1), exog=x_sliding_window.values).fit()
x_sliding_window
df = pd.DataFrame(data={'ds':sales.keys(), 'y':np.log(sales.values) })

holidays_values = pd.read_csv("../input/holidays_events.csv")['date'].values

holidays = pd.DataFrame({
  'holiday': 'holidays',
  'ds': pd.to_datetime(holidays_values),
  'lower_window': 0,
  'upper_window': 5,
})

m1 = Prophet(holidays=holidays).fit(df)
# m1 = Prophet().fit(df)
future1 = m1.make_future_dataframe(periods=365)
forecast1 = m1.predict(future1)
m1.plot(forecast1);
# pd.read_csv("../input/holidays_events.csv")['date']

'ok'
model2 = Prophet(mcmc_samples=500).fit(df)
model2.plot_components(forecast1)
from fbprophet.diagnostics import cross_validation
# from fbprophet.diagnostics.plot_cross_validation_metric import plot_cross_validation_metric
# fig = plot_cross_validation_metric(df_cv, metric='mape')

df_cv = cross_validation(m1, initial='730 days', period='180 days', horizon = '365 days')
df_cv

import fbprophet as fbprophet
# from fbprophet.diagnostics import performance_metrics
# df_p = performance_metrics(df_cv)
# df_p

fbprophet.__version__




















































