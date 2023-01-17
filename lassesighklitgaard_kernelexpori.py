import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import warnings

from statsmodels.tsa.stattools import adfuller

import gc

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.statespace.sarimax import SARIMAX 

from statsmodels.tsa.arima_model import ARMA 

import seaborn as sns

warnings.filterwarnings("ignore")
calendar = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')

train_sales = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')

sample_submission = pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')

sell_prices = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')
calendar.head()
train_sales.head()
sell_prices.head()
sample_submission.head()
sample_submission.tail()
wi3_sales = train_sales.loc[train_sales['store_id'] == 'WI_3']
plt.figure(figsize=(12, 4))

for d in wi3_sales['dept_id'].unique():

    store_sales = wi3_sales.loc[wi3_sales['dept_id'] == d]

    store_sales.iloc[:, 6:].sum().rolling(30).mean().plot(label=d)

plt.title('WI_3 sales by category, rolling mean 30 days')

plt.legend(loc=(1.0, 0.5));
wi_3_prices = sell_prices.loc[sell_prices['store_id'] == 'WI_3']

wi_3_prices['dept_id'] = wi_3_prices['item_id'].apply(lambda x: x[:-4])
wi_3_prices
plt.figure(figsize=(12, 6))

for d in wi_3_prices['dept_id'].unique():

    small_df = wi_3_prices.loc[wi_3_prices['dept_id'] == d]

    grouped = small_df.groupby(['wm_yr_wk'])['sell_price'].mean()

    plt.plot(grouped.index, grouped.values, label=d)

plt.legend(loc=(1.0, 0.5))

plt.title('WI_3 mean sell prices by dept');


train_sales_dept = train_sales.groupby(['dept_id']).sum() #group sales by department

train_sales_item = train_sales.groupby(['item_id']).sum() #group sales by item_id

train_sales_cat = train_sales.groupby(['cat_id']).sum().T #group sales by category

train_sales_cat['day'] = train_sales_cat.index



train_sales_store = train_sales.groupby(['store_id']).sum()

train_sales_state_id = train_sales.groupby(['state_id']).sum()



data_calendar = calendar.iloc[:, [0,2,3,4,5,6,7]]



#Merge data_calendar columns related to days of the week, month and year.

train_sales_cat = pd.merge(data_calendar, train_sales_cat, how = 'inner', left_on='d', right_on='day')

train_sales_cat_final = train_sales_cat.iloc[:,[7,8,9]]

train_sales_cat_final.index = train_sales_cat['date']

train_sales_cat_final.index = pd.to_datetime(train_sales_cat_final.index)

train_sales_cat_final.parse_dates =train_sales_cat_final.index

train_sales_cat_final.head(10)
train_sales_cat_final.plot()
sns.heatmap(train_sales_cat_final[['FOODS','HOBBIES','HOUSEHOLD']].corr(), annot = True,  cbar=True)
train_sales_cat_final_monthly = train_sales_cat_final.iloc[:,[0,1,2]].resample('M').sum()[2:-1] 
train_sales_cat_final_monthly.plot()
plot_acf(train_sales_cat_final_monthly.FOODS, lags=12)
plot_pacf(train_sales_cat_final_monthly.FOODS, lags=12)
result = adfuller(train_sales_cat_final['FOODS'])

print(result)
train_sales_cat_final['FOODS'].plot()
model = ARMA(train_sales_cat_final['FOODS'], order=(1,3))

result = model.fit()
predict = result.predict()

predict.plot()
result.summary()