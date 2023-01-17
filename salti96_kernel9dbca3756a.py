import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
data_path = '../input/competitive-data-science-predict-future-sales'
sales_train = pd.read_csv(f'{data_path}/sales_train.csv')
sales_train.head(10)
items = pd.read_csv(f'{data_path}/items.csv')
items.head()
item_categories = pd.read_csv(f'{data_path}/item_categories.csv')
item_categories.head()
shops = pd.read_csv(f'{data_path}/shops.csv')
shops.head()
all_data = pd.merge(sales_train, items, on='item_id', how='inner')
all_data = pd.merge(all_data, shops, on='shop_id', how='inner')
all_data = pd.merge(all_data, item_categories, on='item_category_id', how='inner')
all_data.head()
monthly_sales = sales_train.groupby(['date_block_num'])['item_cnt_day'].sum()
monthly_sales.index = pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
monthly_sales = monthly_sales.reset_index()
monthly_sales.head()
monthly_sales.columns = ['ds', 'y']

model = Prophet(yearly_seasonality=True)
model.fit(monthly_sales)
future = model.make_future_dataframe(periods=12, freq='MS')
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(4)
forecast
model.plot(forecast, xlabel='Month', ylabel='Sales');
monthly_sales.ds.values
model.plot_components(forecast);
dates = pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')

# For each store, lets count monthly sales
monthly_shop_sales = sales_train.groupby(["date_block_num","shop_id"])["item_cnt_day"].sum()
monthly_shop_sales = monthly_shop_sales.unstack(level=1)

# becase a store might not sell all items, this might result in NaN, we replace it with 0
monthly_shop_sales = monthly_shop_sales.fillna(0)
monthly_shop_sales.index = dates
monthly_shop_sales = monthly_shop_sales.reset_index()
monthly_shop_sales.head(10)
def forecast_shop(shop_id):
    if shop_id < 0 or shop_id > 59:
        print('Shop not found')
        return
    shop_data = pd.concat([monthly_shop_sales.iloc[:,0], monthly_shop_sales.iloc[:, shop_id+1]], axis = 1)
    shop_data.columns = ['ds', 'y']
    model = Prophet()
    model.fit(shop_data)
    future = model.make_future_dataframe(periods=1, freq='MS')
    return model, model.predict(future)
model3, shop3_forecast = forecast_shop(3)
model3.plot(shop3_forecast);
model3.plot_components(shop3_forecast);
monthly_sales = sales_train.groupby(['shop_id', 'item_id', 'date_block_num'])['item_cnt_day'].sum()
# # monthly_sales.fillna(0, inplace=True)
# monthly_sales = monthly_sales.unstack(level=-1).fillna(0)
# monthly_sales = monthly_sales.T
# monthly_sales.index = dates
# monthly_sales = monthly_sales.reset_index()
monthly_sales.head(10)
def forecast_product(shop_id, product_id):
    if shop_id < 0 or shop_id > 59:
        print('Shop not found')
        return

    product_data = monthly_sales[shop_id].iloc[:, product_id]
    product_data = pd.concat([monthly_shop_sales.iloc[:,0], monthly_shop_sales.iloc[:, shop_id+1]], axis = 1)
    product_data.columns = ['ds', 'y']
    model = Prophet()
    model.fit(product_data)
    future = model.make_future_dataframe(periods=1, freq='MS')
    return model, model.predict(future)
monthly_sales = sales_train.groupby(['date_block_num'])['item_cnt_day'].sum()
monthly_sales.index = pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
monthly_sales = monthly_sales.reset_index()
monthly_sales.columns = ['ds', 'y']
monthly_sales.head()
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
from sklearn.metrics import mean_squared_error as mse

def rmsle(y, yhat):
    y, yhat = np.log(y), np.log(yhat)
    return np.sqrt(mse(y, yhat))
model = ARIMA(monthly_sales.y.values ,order=(0,0,0))
model_fit = model.fit(disp=0)

residuals = pd.DataFrame(model_fit.resid)
residuals.plot();
series = monthly_sales.y.values
train_size = int(len(series) * 0.5)
train, test = series[:train_size], series[train_size:]

history = [x for x in train]
preds = []

for t in range(len(test)):
    model = ARIMA(history, order=(1,1,0)).fit(disp=0)
    yhat = model.forecast()[0]
    preds.append(yhat)
    history.append(test[t])
    
print(rmsle(test, preds))

plt.figure(figsize=(8, 5))
plt.plot(list(train) + preds, color='r');
plt.plot(monthly_sales.y, color='b');
series = monthly_sales.y.values
history = list(series)

for t in range(12):
    model = ARIMA(history, order=(5,1,0)).fit(disp=0)
    yhat = model.forecast()[0]
    history.append(yhat)

    
plt.figure(figsize=(8, 5))
plt.plot(history,color='r');
plt.plot(series,color='b');
model_fit.summary()
residuals.plot(kind='kde');
residuals.describe()
