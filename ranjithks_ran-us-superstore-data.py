import pandas as pd

import numpy as np



import warnings

warnings.filterwarnings("ignore")
PATH = '/kaggle/input/superstore'

DATASET = 'US Superstore data.xls'
"""

from io import BytesIO

from urllib.request import urlopen

from zipfile import ZipFile



with urlopen(DATASET_URL) as zipresp:

    with ZipFile(BytesIO(zipresp.read())) as zfile:

        zfile.extractall(PATH)

"""
%ls -lrt
df = pd.read_excel(f'{PATH}/{DATASET}', date_parser=['TransactionDateTime', 'Ship Date'])
df.sample(10)
df.drop('Row ID', axis=1, inplace=True)
df.shape
df.info()
df['NetPrice'] = (df['Sales'] - df['Profit']) / df['Quantity']
df.sample(5)
df.rename(columns={'Order ID': 'TransactionID', 'Order Date': 'TransactionDateTime', 'Ship Date': 'ShipDate', 'Ship Mode': 'ShipMode', 'Sub-Category': 'SubCategory', 'Postal Code':'PostalCode', 'Customer ID': 'CustomerID', 'Customer Name': 'CustomerName', 'Product ID': 'SKU', 'Product Name': 'ProductName'}, inplace=True)
df.info()
df.describe()
df['SKU'].unique()
df['SKU'].nunique()
df['TransactionDateTime'].min(), df['TransactionDateTime'].max()
from datetime import datetime



today = datetime.today()

offset = today - df.TransactionDateTime.max()

print(offset.days)
#df['TransactionDateTime'] = df['TransactionDateTime'] + pd.DateOffset(offset.days)

#df['ShipDate'] = df['ShipDate'] + pd.DateOffset(offset.days)
#df['TransactionDateTime'].min(), df['TransactionDateTime'].max()
df.loc[:, 'Country'].unique()
df.loc[:, 'State'].unique()
df.loc[:, 'City'].unique()
states = df.loc[:, 'State'].unique()

stateCities = dict()

for state in states:

  stateCities[state] = df.loc[df.loc[:, 'State'] == state, 'City'].nunique()

  

print(stateCities)

df.loc[:, 'State'].value_counts()
df.loc[:, 'ProductName'].value_counts()
df.loc[df.loc[:, 'TransactionID'] == 'CA-2014-148040']
df['Year'] = df['TransactionDateTime'].dt.year

df['Month'] = df['TransactionDateTime'].dt.strftime('%Y-%b')

df['Week'] = df['TransactionDateTime'].dt.week

df['DayOfWeek'] = df['TransactionDateTime'].dt.dayofweek

df['Day'] = df['TransactionDateTime'].dt.day
df.sample(5)
import plotly.express as px
px.scatter(data_frame=df, x='TransactionDateTime', y='Sales', size='Sales', color='Category')
thresholdSales = 2000

df.drop(df[df['Sales'] > thresholdSales].index, inplace=True)
px.scatter(data_frame=df, x='TransactionDateTime', y='Sales', size='Sales', color='Category')
import plotly.graph_objects as go
df_sales_per_week = df.groupby(['Year', 'Week'])['Sales'].sum().reset_index()
df_sales_per_week.head()
df_sales_per_week.loc[:, 'Year'].unique()
for year in df_sales_per_week.loc[:, 'Year'].unique():

  df_plot = df_sales_per_week.loc[df_sales_per_week.loc[:, 'Year'] == year]

  x = df_plot.loc[:, 'Week']

  y = df_plot.loc[:, 'Sales']

  fig = go.Figure(go.Bar(x=x, y=y))

  fig.add_trace(go.Line(x=x, y=y))

  fig.show()
fig = px.line(df_sales_per_week, x="Week", y="Sales", color='Year', title="Weekly Sales")

fig.show()
df_sales_per_month = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
fig = px.line(df_sales_per_month, x="Month", y="Sales", title="Monthly Sales")

fig.show()
N = 5

topN_products = df['SKU'].value_counts()[:N].index
df_sales_per_product_id = df.groupby(['Year', 'SKU', 'Month'])['Sales'].count().reset_index()
df_sales_per_product_id
df_plot = df_sales_per_product_id.loc[df_sales_per_product_id.loc[:, 'SKU'].isin(topN_products)]
fig = px.bar(df_plot.loc[df_plot.loc[:, 'Year'] == 2014], x="Month", y="Sales", color='SKU', title="Monthly SKU Sales")

fig.show()
fig = px.bar(df_plot.loc[df_plot.loc[:, 'Year'] == 2015], x="Month", y="Sales", color='SKU', title="Monthly SKU Sales")

fig.show()
fig = px.bar(df_plot.loc[df_plot.loc[:, 'Year'] == 2016], x="Month", y="Sales", color='SKU', title="Monthly SKU Sales")

fig.show()
fig = px.bar(df_plot.loc[df_plot.loc[:, 'Year'] == 2017], x="Month", y="Sales", color='SKU', title="Monthly SKU Sales")

fig.show()
df.info()
df['Category'].unique()
df['SubCategory'].unique()
from fbprophet import Prophet
# set the uncertainty interval to 95% (the Prophet default is 80%)

my_model = Prophet(interval_width=0.95)
df_Train = df.loc[:, ['TransactionDateTime', 'Sales']]
df_Train.info()
df_Train.sort_values(by='TransactionDateTime', ascending=True, inplace=True)
df_Train['TransactionDateTime'][:500].value_counts()
df_Train = df_Train.set_index('TransactionDateTime')

daily_df = df_Train.resample('D').mean()

df_Train.reset_index()

d_df = daily_df.reset_index().dropna()
d_df['Sales'] = np.log(d_df['Sales'])
d_df.columns = ['ds', 'y']
d_df.head()
d_df['ds'].min(), d_df['ds'].max()
m = Prophet()

m.fit(d_df)
future = m.make_future_dataframe(periods=12, freq="m")

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast, uncertainty=True, figsize=(30,6))
fig2 = m.plot_components(forecast, figsize=(30,12))
from fbprophet.plot import plot_plotly



fig = plot_plotly(m, forecast, figsize=(1600, 900))

fig.show()
forecast
d_df
out_df = forecast[['ds', 'yhat']]

out_df.set_index('ds', inplace=True)
in_df = d_df.copy()

in_df.set_index('ds', inplace=True)



out_df['y'] = in_df['y']

out_df = out_df.reset_index()
metrics_df = out_df[out_df['ds'] <= df['TransactionDateTime'].max()]

out_df
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error



mae = mean_absolute_error(y_true=metrics_df['y'], y_pred=metrics_df['yhat'])

mse = mean_squared_error(y_true=metrics_df['y'], y_pred=metrics_df['yhat'])

msle = mean_squared_log_error(y_true=metrics_df['y'], y_pred=metrics_df['yhat'])



print(f'Prophet model has MAE {mae:0.4f} - MSE {mse:0.4f} - MSLE {msle:0.4f}')
del out_df, in_df, metrics_df
df.sample(5)
df['SKU'].nunique()
df['SKU'].value_counts()
thresholdNProducts = 10

indexNProducts = (df['SKU'].value_counts() >= thresholdNProducts).sum()

print(indexNProducts)
df['SKU'].value_counts()[:indexNProducts].index
topNProducts = list(df['SKU'].value_counts()[:indexNProducts].index)
df_temp = df[['TransactionDateTime', 'SKU', 'Quantity']]

minDate = df['TransactionDateTime'].min()

maxDate = df['TransactionDateTime'].max()

df_emptyDateIndex = pd.DataFrame(index=pd.date_range(start=minDate, end=maxDate, freq="7d"), columns=['Quantity']).fillna(0)

df_emptyDateIndex.index.rename("TransactionDateTime", inplace=True)

df_topNProducts = dict()

for sku in topNProducts:

  df_empty_temp = df_emptyDateIndex.copy()

  df_sku_temp = df_temp[df_temp['SKU'] == sku].groupby('TransactionDateTime')['Quantity'].sum().reset_index().set_index('TransactionDateTime')

  df_topNProducts[sku] = pd.concat([df_sku_temp, df_empty_temp], join='outer')

  
df_temp[df_temp['SKU'] == 'OFF-PA-10001970']['Quantity'].sum()
df_sku_temp = df_temp[df_temp['SKU'] == 'OFF-PA-10001970'].groupby('TransactionDateTime')['Quantity'].sum().reset_index().set_index('TransactionDateTime')

df_sku_temp
df_empty_temp = df_emptyDateIndex.copy()

df_empty_temp
df_empty_temp.update(df_sku_temp)

df_empty_temp.combine_first(df_sku_temp)

df_empty_temp['Quantity'].sum()
df_tt = df_topNProducts['OFF-PA-10001970']

print(df_tt[df_tt['Quantity'] > 0])

print(df_tt['Quantity'].sum())
def prophetModel(df, period, frequency):

  m = Prophet()

  #m.add_country_holidays(country_name='US')

  m.fit(df)

  future = m.make_future_dataframe(periods=period, freq=frequency)

  forecast = m.predict(future)

  return forecast
df_toModel = df_topNProducts['OFF-PA-10001970'].reset_index().dropna()

df_toModel.columns = ['ds', 'y']

#df_toModel['y'] = np.log(df_toModel['y'])

#df_toModelTrain = df_toModel[:int(df_toModel.shape[0] * 0.8)]

df_toModelTrain = df_toModel

df_forecastProduct = prophetModel(df_toModelTrain, 2, "w") # 21 days

df_forecastProduct[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(50)
forecastNProducts = dict()

period = 2

frequency = "w"

for key in df_topNProducts.keys():

  df_toModel = df_topNProducts[key].reset_index().dropna()

  df_toModel.columns = ['ds', 'y']

  df_forecastProduct = prophetModel(df_toModel, period, frequency) # for 3 weeks

  forecastNProducts[key] = df_forecastProduct.tail()
print(forecastNProducts['OFF-PA-10001970'])
df.loc[df.loc[:, 'SKU'] == 'OFF-PA-10000587']
df.info()
df11 = df.groupby('SKU')['ProductName', 'NetPrice'].max().reset_index()

df11['Quantity'] = np.random.choice([2, 5], df11.shape[0])
df11.info()
#df11.to_csv('inventory.csv')
#df.to_csv('tranactions.csv')
import fbprophet as prophet

prophet.__version__

import pickle
df_Products = df.groupby(['SKU', 'ProductName'])['CustomerID'].max().reset_index()[['SKU', 'ProductName']]

df_Products.sample(5)
df_predictedAll = pd.DataFrame()



for key in forecastNProducts.keys():

  df_prod = forecastNProducts[key][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

  df_prod['SKU'] = key

  df_predictedAll = df_predictedAll.append(df_prod)
df_predictedAll = df_predictedAll.merge(df_Products, on='SKU')

df_predictedAll.sample(5)
df_predictedAll.drop_duplicates(subset=['ds', 'SKU'], keep='last', inplace=True)
df_predictedAll.to_csv('predictions.csv', index=False)
df.columns