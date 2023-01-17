import numpy as np # linear algebra
import pandas as pd
import random as rd # generating random numbers
import datetime # manipulating date formats
# Viz
import matplotlib.pyplot as plt # basic plotting
import seaborn as sns # for prettier plots
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# TIME SERIES
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
sales = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
item_cat = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
sample_sub = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')


print('sales:', sales.shape, 'items:', items.shape, 'item_cats:', item_cat.shape, 'shops:', shops.shape, 'sample submission', sample_sub.shape)
sales['date'] = pd.to_datetime(sales['date'])
print(sales.info())
sales.head(5)
# looking for outliers
sns.boxplot(x=sales.item_cnt_day)
sns.boxplot(x=sales.item_price)
# remove it
sales = sales[(sales.item_price < 100000) & (sales.item_price > 0)]
sales = sales[(sales.item_cnt_day < 1001) & (sales.item_cnt_day > 0)]
# get shops ID and item ID
test_shops = test.shop_id.unique()
sales = sales[sales.shop_id.isin(test_shops)]
test_items = test.item_id.unique()
train = sales[sales.item_id.isin(test_items)]
# aggregation for the monthly sales
total_sales = sales.groupby(["date_block_num"])["item_cnt_day"].sum()
total_sales.head(20)
fig = px.line(total_sales, labels={
    'date_block_num': "Month",
    'value': "Sales"
    })
fig.show()
res = sm.tsa.seasonal_decompose(total_sales.values, period=12)
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(y=res.seasonal,
                    mode='lines',
                    name='seasonal'))
fig.add_trace(go.Scatter(y=res.trend,
                    mode='lines',
                    name='trend'),
                    secondary_y=True)
fig.add_trace(go.Scatter(y=res.resid,
                    mode='lines',
                    name='residual'))
fig.add_trace(go.Scatter(y=res.observed,
                    mode='lines',
                    name='observed'),
                    secondary_y=True)
fig.update_layout(title={
                    'text': "Original"
})

fig.show()

adf = sm.tsa.stattools.adfuller(total_sales)
print('Dickey-Fuller results:', adf[1:])
total_sales_no_trend = total_sales - total_sales.shift(1)

res = sm.tsa.seasonal_decompose(total_sales_no_trend[1:], period=12)
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(y=res.trend,
                    mode='lines',
                    name='trend'),
                    secondary_y=True)
fig.add_trace(go.Scatter(y=res.observed,
                    mode='lines',
                    name='observed'),
                    secondary_y=True)
fig.add_trace(go.Scatter(y=res.seasonal,
                    mode='lines',
                    name='seasonal'))
fig.update_layout(title={
                    'text': "After removing the trend"
                    })

fig.show()

adf = sm.tsa.stattools.adfuller(total_sales_no_trend[1:])
print('Dickey-Fuller results:', adf[1:])
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot

ts = sales.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.index=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
ts=ts.reset_index()
ts.head()
ts.columns=['ds','y']
# build model
model = Prophet( yearly_seasonality=True) #instantiate Prophet with only yearly seasonality as our data is monthly 
model.fit(ts) #fit the model with your dataframe
# make forecast
# predict for five months in the furure and MS - month start is the frequency
future = model.make_future_dataframe(periods = 5, freq = 'MS')  
# now lets make the forecasts
forecast = model.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
fig = model.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), model, forecast)
fig = model.plot_components(forecast)
test.insert(loc=3, column='date_block_num', value=34)
# test = pd.merge(test, shops, on=['shop_id'], how='left')
test = pd.merge(test, items, on=['item_id'], how='left')
test['date_block_num'] = 34

test = test.drop(['item_name', 'ID'], axis=1)
print(f'Shape of test data: {test.shape}')
test.head(5)
# merge with shops and items
train = pd.merge(sales, shops, on=['shop_id'], how='left')
train = pd.merge(train, items, on=['item_id'], how='left')
train = pd.merge(train, item_cat, on=['item_category_id'], how='left')
train = train.groupby(['shop_id', 'item_id', 'date_block_num', 'item_category_id'])['item_cnt_day'].sum().rename('item_cnt_month').reset_index()
train['item_cnt_month'] = (train['item_cnt_month'].fillna(0).clip(0,20))

# add test
df = pd.concat([train, test], ignore_index=True, sort=False, keys=['shop_id', 'item_id', 'date_block_num'])
df.fillna(0, inplace=True)

print(f'Shape of training data: {df.shape}')
def generate_lag(train, months, lag_column):
    for month in months:
        # Speed up by grabbing only the useful bits
        train_shift = train[['date_block_num', 'shop_id', 'item_id', lag_column]].copy()
        train_shift.columns = ['date_block_num', 'shop_id', 'item_id', lag_column+'_lag_'+ str(month)]
        train_shift['date_block_num'] += month
        train = pd.merge(train, train_shift, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    return train

df = generate_lag(df, [1, 2, 3], 'item_cnt_month')
df.fillna(0, inplace=True)
X_train = df[df.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = df[df.date_block_num < 33]['item_cnt_month']
X_valid = df[df.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = df[df.date_block_num == 33]['item_cnt_month']
X_test = df[df.date_block_num == 34].drop(['item_cnt_month'], axis=1)
X_train.head(5)
import lightgbm as lgb
import optuna
import sklearn.metrics

from sklearn.metrics import mean_squared_error as rmse

feature_name = X_train.columns.tolist()

lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_valid, Y_valid, reference=lgb_train)
def objective(trial):
        
    param = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        'learning_rate':0.001,
        "num_leaves": trial.suggest_int("num_leaves", 50, 150),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.8, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    model = lgb.train(param, 
                      lgb_train,
                      valid_sets=[lgb_train,lgb_eval],
                      early_stopping_rounds=15,
                      verbose_eval=1)
    
    y_pred = model.predict(X_valid)
    accuracy = rmse(Y_valid, y_pred)

    return accuracy
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)
 
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
opt_params = study.best_trial.params
print(f'optimal trial parameters\n{opt_params}')
x = {"objective": "regression",
     "metric"   : "rmse",
     "verbosity": -1,
     "boosting_type": "gbdt"}

opt_params.update(x)
opt_params
evals_result = {} 

model = lgb.train(opt_params,
                  lgb_train,
                  valid_sets=[lgb_train,lgb_eval],
                  evals_result=evals_result,
                  early_stopping_rounds=100,
                  verbose_eval=1,
                  )
lgb.plot_importance(model, max_num_features=10, importance_type='gain')
# submission
Y_test = model.predict(X_test[feature_name]).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv('submission.csv', index=False)