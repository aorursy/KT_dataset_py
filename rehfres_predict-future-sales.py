# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

# import datetime

import matplotlib.pyplot as plt

# import seaborn as sns

from sklearn.preprocessing import LabelEncoder

import gc


gc.collect() 

sales=pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

item_cats=pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

items=pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

sample_sub=pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")

shops=pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

test=pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
# Якутск Орджоникидзе, 56

sales.loc[sales.shop_id == 0, 'shop_id'] = 57

test.loc[test.shop_id == 0, 'shop_id'] = 57

# Якутск ТЦ "Центральный"

sales.loc[sales.shop_id == 1, 'shop_id'] = 58

test.loc[test.shop_id == 1, 'shop_id'] = 58

# Жуковский ул. Чкалова 39м²

sales.loc[sales.shop_id == 10, 'shop_id'] = 11

test.loc[test.shop_id == 10, 'shop_id'] = 11
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'

shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])

shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'

shops['city_code'] = LabelEncoder().fit_transform(shops['city'])

shops = shops[['shop_id','city_code']]



item_cats['split'] = item_cats['item_category_name'].str.split('-')

item_cats['type'] = item_cats['split'].map(lambda x: x[0].strip())

item_cats['type_code'] = LabelEncoder().fit_transform(item_cats['type'])

# if subtype is nan then type

item_cats['subtype'] = item_cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

item_cats['subtype_code'] = LabelEncoder().fit_transform(item_cats['subtype'])

item_cats = item_cats[['item_category_id','type_code', 'subtype_code']]
# items.iloc[22154]
sales = sales[sales.item_price<100000]

sales = sales[sales.item_cnt_day<1001]
median = sales[(sales.shop_id==32)&(sales.item_id==2973)&(sales.date_block_num==4)&(sales.item_price>0)].item_price.median()

sales.loc[sales.item_price<0, 'item_price'] = median
#date formats are fine

# sales['date'].str.contains('^\d\d.\d\d.\d{4}$', regex=True).value_counts() == sales['date'].count()

# sales['date'] = pd.to_datetime(sales['date'], dayfirst=True)

sales[sales['item_cnt_day']==1].head(20)

# sales['item_cnt_day'] = sales['item_cnt_day'].astype('int32')

# print(sales.dtypes)

# sales = sales.convert_dtypes()

sales.dtypes
sales[(sales['shop_id'] == 25)&(sales['item_id']==22154)].item_price.mean()
# sales[(sales['shop_id'] == 25)&(sales['item_id']==22154)]

# # sales.mode

# # sales[sales['item_id']==22154]

# sales[(sales['shop_id'] == 55)&(sales['item_id']==1)]

sales['revenue'] = sales['item_price']*sales['item_cnt_day']

# dfsSales.tail(50)

sales[(sales['shop_id'] == 2)&(sales['item_id']==1523)]

# sales[(sales['shop_id'] == 55)&(sales['item_id']==1)&(sales['date_block_num']==15)]
sales
# sales.groupby()

# sales2 = sales[sales['date_block_num'] >= 21]

# sales2

sales2 = sales

# sales2
salesPrices = sales2[['shop_id', 'item_id', 'item_price']].groupby(['item_id', 'shop_id'], as_index=False).agg({'item_price': 'mean'})

salesPrices.rename({'item_price':'item_shop_price_mean'}, axis=1, inplace=True)

salesPrices
salesPrices[(salesPrices['shop_id'] == 25)&(salesPrices['item_id']==2564)]
salesMulti = sales2.groupby(['item_id', 'shop_id', 'date_block_num'], as_index=False).agg({'item_price': 'mean', 'item_cnt_day': 'sum', 'revenue': 'sum'})

salesMulti.rename({'item_cnt_day':'item_cnt_month'}, axis=1, inplace=True)

salesMulti
salesMulti.describe()
# uniqueItemAndShop = sales.drop([]).groupby(['item_id', 'shop_id'], as_index=False).agg({'item_price': 'mean', 'item_cnt_day': 'sum'})

uniqueItemAndShops = sales[['item_id', 'shop_id']].drop_duplicates()



dfs = []

for i in range(0, 34):

    df = uniqueItemAndShops.copy()

    df['date_block_num'] = i

    dfs.append(df)

dfs = pd.concat(dfs)

# dfs[(dfs['shop_id'] == 25)&(dfs['item_id']==22154)]

# dfs = dfs.sort_values(by=['item_id', 'shop_id']).reset_index(drop=True)

# dfs.head(35)



print('before:', dfs.shape)

random_items = items.item_id.sample(frac=0.3).to_numpy()

dfs = dfs[dfs.item_id.isin(random_items)]

print('after:', dfs.shape)



dfs[(dfs['shop_id'] == 2)&(dfs['item_id']==1523)].sort_values('date_block_num')#.iloc[-1]
# uniqueItemAndShops.head(20).apply(lambda x : print(x), axis=1)#.reset_index()

dfsSales = dfs.merge(salesMulti, on=['item_id', 'shop_id', 'date_block_num'], how='left')

dfsSales = dfsSales.sort_values(by=['item_id', 'shop_id']).reset_index(drop=True)

dfsSales[['item_cnt_month', 'revenue']] = dfsSales[['item_cnt_month', 'revenue']].fillna(value=0)

dfsSales = dfsSales.merge(items[['item_id', 'item_category_id']], on='item_id')

dfsSales.head(35)



# dfsSales[(dfsSales['shop_id'] == 25)&(dfsSales['item_id']==22154)]
dfsSales[(dfsSales['shop_id'] == 2)&(dfsSales['item_id']==1523)].sort_values('date_block_num')
# salesCats = dfsSales[['shop_id', 'date_block_num', 'item_category_id', 'item_cnt_month']].groupby(['shop_id', 'item_category_id', 'date_block_num'], as_index=False).agg({'item_cnt_month': 'mean'})

# salesCats.rename({'item_cnt_month':'shop_item_category_cnt_month_mean'}, axis=1, inplace=True)

# salesCats



salesCats = sales2.merge(items[['item_id', 'item_category_id']], on='item_id')

# salesCats[(salesCats['shop_id'] == 2)&(salesCats['item_category_id']==2)&(salesCats['date_block_num']==1)].item_cnt_day.sum()

salesCats = salesCats[['shop_id', 'date_block_num', 'item_category_id', 'item_cnt_day']].groupby(['shop_id', 'item_category_id', 'date_block_num'], as_index=False).agg({'item_cnt_day': 'sum'})

salesCats.rename({'item_cnt_day':'shop_item_category_cnt_month_mean'}, axis=1, inplace=True)

salesCats.head(35)





sales2[(sales2['shop_id'] == 2)&(sales2['item_id']==1523)].sort_values('date_block_num')

# salesCats[(salesCats['shop_id'] == 2)&(salesCats['item_category_id']==21)].sort_values('date_block_num')
# testDf = dfsSales[(dfsSales['shop_id'] == 55)&(dfsSales['item_id']==492)].reset_index(drop=True)

# testDf

# y = testDf['item_cnt_month']

# y
# !pip install sktime

# !pip install pmdarima

# from sktime.utils.plotting.forecasting import plot_ys

# from sktime.forecasting.model_selection import temporal_train_test_split

# from sktime.performance_metrics.forecasting import smape_loss
# fig, ax = plot_ys(y)

# ax.set(xlabel="Months", ylabel="Sales per month");
# y_train, y_test = temporal_train_test_split(y)

# fh = np.arange(len(y_test)) + 1

# print(y_train.shape[0], y_test.shape[0], fh)
# from sktime.forecasting.arima import AutoARIMA

# from sklearn.metrics import mean_squared_error



# forecaster = AutoARIMA(sp=12, suppress_warnings=True)

# forecaster.fit(y_train)

# y_pred = forecaster.predict(fh)

# plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"]);

# mean_squared_error(y_test, y_pred)

# from sktime.forecasting.compose import DirectRegressionForecaster

# from sktime.regression.compose import TimeSeriesForestRegressor



# forecaster = DirectRegressionForecaster(regressor=TimeSeriesForestRegressor)

# forecaster.fit(y_train, fh=fh)

# y_pred = forecaster.predict(fh)

# plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"]);

# smape_loss(y_test, y_pred)

# from sklearn.neighbors import KNeighborsRegressor



# regressor = KNeighborsRegressor(n_neighbors=1)

# from sklearn.model_selection import GridSearchCV

# from sktime.forecasting.compose import RecursiveRegressionForecaster

# from sklearn.ensemble import RandomForestRegressor



# # tuning the 'n_estimator' hyperparameter of RandomForestRegressor from scikit-learn

# regressor_param_grid = {"n_estimators": [100, 200, 300]}

# forecaster_param_grid = {"window_length": [5,10,15,20,25]}



# # create a tunnable regressor with GridSearchCV

# regressor = GridSearchCV(RandomForestRegressor(), param_grid=regressor_param_grid)

# forecaster = RecursiveRegressionForecaster(regressor, window_length=15)



# cv = SlidingWindowSplitter(initial_window=int(len(y_train) * 0.5))

# gscv = ForecastingGridSearchCV(forecaster, cv=cv, param_grid=forecaster_param_grid)

# from sktime.forecasting.trend import PolynomialTrendForecaster

# from sktime.transformers.single_series.detrend import Detrender



# # liner detrending

# forecaster = PolynomialTrendForecaster(degree=1)

# transformer = Detrender(forecaster=forecaster)

# yt = transformer.fit_transform(y_train)



# # internally, the Detrender uses the in-sample predictions of the PolynomialTrendForecaster

# forecaster = PolynomialTrendForecaster(degree=1)

# fh_ins = -np.arange(len(y_train)) # in-sample forecasting horizon

# y_pred = forecaster.fit(y_train).predict(fh=fh_ins)



# plot_ys(y_train, y_pred, yt, labels=["y_train", "Fitted linear trend", "Residuals"]);

sales[sales.item_id == 492]#.groupby(['item_id', 'shop_id'], as_index=False).agg({'item_price': 'mean', 'item_cnt_month': 'sum'})
dfsSalesFull = dfsSales[['item_id', 'shop_id', 'item_cnt_month']]

dfsSalesFull

dfsSalesFull = dfsSalesFull.groupby(['item_id', 'shop_id'], as_index=False).sum()

dfsSalesFull.rename({'item_cnt_month':'item_cnt_all'}, axis=1, inplace=True)

dfsSalesFull = dfsSalesFull.sort_values(by='item_cnt_all', ascending=False).reset_index(drop=True)

dfsSalesFull.iloc[30:40]

# dfsSalesFull[(dfsSalesFull['shop_id'] == 31)&(dfsSalesFull['item_id']==20949)]
dfsSalesFull
# salesMulti[(salesMulti['shop_id'] == 25)&(salesMulti['item_id']==22154)]
# salesMulti.reindex('date_block_num')

# salesMulti.xs('shop_id').set_index('date_block_num')
# salesMulti.loc[salesMulti['item_id'] == 32]

# salesMulti.xs((25, 22154), level=('shop_id', 'item_id'))
# salesMulti.groupby(level=0).mean()
# dfn = dfsSales[['item_cnt_day', 'item_id']].shift(1)

# dfn[dfn['item_id']==22154].head()

# dfsSales[dfsSales['item_id']==22154].head()

gc.collect() 
# dfsSales[dfsSales.item_category_id == 21]

# dfsSales[(dfsSales['shop_id'] == 26)&(dfsSales['item_id']==1347)].sort_values('date_block_num')
dfsSales2 = dfsSales.copy()

dfsSales2 = dfsSales2.merge(dfsSalesFull, on=['item_id', 'shop_id'])

dfsSales2 = dfsSales2.merge(salesPrices, on=['item_id', 'shop_id'])

dfsSales2 = dfsSales2.merge(salesCats, on=['shop_id', 'item_category_id', 'date_block_num'], how="left")

dfsSales66 = dfsSales2.copy()

dfsSales2 = dfsSales2.sort_values(by=['item_id', 'shop_id', 'date_block_num'])

dfsSales2



for i in range(1, 13):

#     dfsSales2['item_cnt_month, t-' + str(i)] = dfsSales2['item_cnt_month'].shift(i)

    dfsSales2[['item_cnt_month, t-' + str(i), 'shop_id, t-' + str(i), 'shop_item_category_cnt_month_mean, t-' + str(i)]] = dfsSales2[['item_cnt_month', 'shop_id', 'shop_item_category_cnt_month_mean']].shift(i)

#     dfsSales2[['item_cnt_month, t-' + str(i), 'shop_id, t-' + str(i)]] = dfsSales2[['item_cnt_month', 'shop_id']].shift(i)

    dfsSales2['item_price delta, t-' + str(i)] = (dfsSales2['item_price'].shift(i) - dfsSales2['item_shop_price_mean']) / dfsSales2['item_shop_price_mean']

#     dfsSales2['item_price delta, t-' + str(i)] = (dfsSales2['item_price'].shift(i) - dfsSales2['item_shop_price_mean']) / dfsSales2['item_shop_price_mean']

#     dfsSales2['shop_id, t-' + str(i)] = dfsSales2['shop_id'].shift(i)



# dfsSales2['shop_id, t-33'] = dfsSales2['shop_id'].shift(33)

dfsSales2['item_price delta'] = (dfsSales2['item_price'] - dfsSales2['item_shop_price_mean']) / dfsSales2['item_shop_price_mean']

dfsSales2[(dfsSales2['shop_id'] == 2)&(dfsSales2['item_id']==3327)]

# dfnew = dfsSales.shift(1)

# dfnew

dfsSales2

# dfsSales2[(dfsSales2['shop_id'] == 2)&(dfsSales2['item_id']==1523)].sort_values('date_block_num')

# dfsSales2['item_price, t-1'].describe()


dfsSales2[(dfsSales2['shop_id'] == 2)&(dfsSales2['item_id']==1523)].sort_values('date_block_num')
# dfsSales2[dfsSales2['shop_id'] == 25]

dfsSales2[(dfsSales2['shop_id'] == 55)&(dfsSales2['item_id']==1)].sort_values('date_block_num')
for i in range(1, 13):

    dfsSales2.loc[dfsSales2['shop_id'] != dfsSales2['shop_id, t-' + str(i)], ['item_cnt_month, t-' + str(i), 'item_price delta, t-' + str(i), 'shop_item_category_cnt_month_mean, t-' + str(i)]] = np.nan

dfsSales2[(dfsSales2['shop_id'] == 2)&(dfsSales2['item_id']==3327)]

# dfsSales2[(dfsSales2['shop_id'] == 55)&(dfsSales2['item_id']==1)].sort_values('date_block_num')
# dfsSales2 = dfsSales2.filter(regex='^((?!shop_id, t-).)*$', axis=1)

# dfsSales2

dfsSales2[(dfsSales2['shop_id'] == 2)&(dfsSales2['item_id']==1523)].sort_values('date_block_num')
dfsSales3 = dfsSales2.copy()

dfsSales3['done?'] = False

for i in range(12, 0, -1):

    dfsSales3.loc[dfsSales3['item_cnt_month, t-' + str(i)] > 0, 'done?'] = True

    dfsSales3.loc[(dfsSales3['done?'] == False)&(dfsSales3['item_cnt_month, t-' + str(i)] == 0), 'item_cnt_month, t-' + str(i)] = np.nan

dfsSales3 = dfsSales3.drop('done?', axis=1)

dfsSales3.head(50)

# dfsSales4 = dfsSales3[dfsSales3['item_price'] > 0]

# dfsSales4[]

dfsSales4 = dfsSales3

dfsSales4[(dfsSales4['shop_id'] == 4)&(dfsSales4['item_id']==3835)]

# dfsSales4
dfsSales4

dfsSales4.memory_usage(index=True).sum()
# dfsSales4[(dfsSales4.date_block_num < 12)&(dfsSales4['item_cnt_month, t-12'] > 0)]

dfsSales4[(dfsSales4['shop_id'] == 2)&(dfsSales4['item_id']==1523)].sort_values('date_block_num')
dfsSales4[(dfsSales4['shop_id'] == 4)&(dfsSales4['item_id']==3423)]
# # range(1, 13)

# for i in range(22, 34):

#     print(i)

    

# print('\n')

    

# for i in range(1, 13):

#     print(33 - i)



# for i in range(12, 0, -1):

#     print(i)
# df = dfsSales4.merge(items[['item_id', 'item_category_id']], on='item_id')

# df = dfsSales4.merge(salesCats, on=['shop_id', 'item_category_id'])

# df = df.drop(['item_id', 'shop_id'], axis=1)

# df = df.drop('revenue', axis=1)

df = dfsSales4#[dfsSales4.date_block_num > 20]



# del dfsSales

# del dfsSales2

# del dfsSales3

# del dfsSales4

# del sales

# del items

# del sample_sub

# del test

# del shops

# del item_cats

# del sales2

# del salesPrices

# del salesMulti

# del dfs

# del dfsSalesFull



# gc.collect()

df.dtypes
from xgboost import XGBRegressor

from xgboost import plot_importance
X_train = df[df.date_block_num < 12].drop(['item_cnt_month'], axis=1)

Y_train = df[df.date_block_num < 12]['item_cnt_month']

X_valid = df[(df.date_block_num >= 12)&(df.date_block_num < 24)].drop(['item_cnt_month'], axis=1)

Y_valid = df[(df.date_block_num >= 12)&(df.date_block_num < 24)]['item_cnt_month']

X_test = df[df.date_block_num == 33].drop(['item_cnt_month'], axis=1)
print(

    X_train.shape,

    Y_train.shape,

    X_valid.shape,

    Y_valid.shape,

    X_test.shape

    )

X_test
X_train.head(35)
Y_train.head(35)
X_valid.head(35)
Y_valid.head(35)
X_valid[(X_valid['shop_id'] == 2)&(X_valid['item_id']==2921)].sort_values('date_block_num')
model = XGBRegressor(

    max_depth=8,

    n_estimators=1000,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,    

    seed=42)



model.fit(

    X_train, 

    Y_train, 

    eval_metric="rmse", 

    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 

    verbose=True, 

    early_stopping_rounds = 10)





# import pickle



# filename = 'finalized_model.sav'

# # pickle.dump(model, open(filename, 'wb'))



# model = pickle.load(open(filename, 'rb'))

# result = model.score(X_valid, Y_valid)

# print(result)
# plot_features(model, (10,14))

plt.style.use('dark_background')



fig, ax = plt.subplots(1,1,figsize=(10,20))

plot_importance(booster=model, ax=ax)
from xgboost import plot_tree



# fig = plt.figure()

# plot_tree(model, num_trees=4)



plot_tree(model, num_trees=2)

fig = plt.gcf()

fig.set_size_inches(150, 100)

fig.savefig('tree.png')
# data = X_valid.join(Y_valid)

# data

# set = data[(data['shop_id'] == 31)&(data['item_id']==20949)&(data.date_block_num == 33)]

# set



# set = df[(df['shop_id'] == 31)&(df['item_id']==20949)&(df.date_block_num == 33)]

# set

display(df[(df['shop_id'] == 2)&(df['item_id']==1523)].sort_values('date_block_num'))

# df[(df['shop_id'] == 2)&(df['item_id']==1523)].sort_values('date_block_num').iloc[-1]

# df[(df['shop_id'] == 54)&(df['item_id']==22152)][['shop_id', 'item_id', 'shop_item_category_cnt_month_mean', 'date_block_num', 'item_cnt_month', 'item_cnt_month, t-1', 'item_cnt_month, t-2', 'item_cnt_month, t-3', 'item_cnt_month, t-4', 'item_cnt_month, t-5', 'item_cnt_month, t-6', 'item_cnt_month, t-7', 'item_cnt_month, t-8', 'item_cnt_month, t-9', 'item_cnt_month, t-10', 'item_cnt_month, t-11', 'item_cnt_month, t-12']]
# df[(df.item_cnt_month < 20)&(df.item_cnt_month > 10)&(df['item_cnt_month, t-1'] > 10)&(df['item_cnt_month, t-1'] > 10)]

# df[(df.item_cnt_month < 20)&(df.item_cnt_month > 10)&(df['item_cnt_month, t-1'] > 10)&(df['item_cnt_month, t-1'] > 10)&(df.date_block_num == 33)]

# df[pd.isnull(df.shop_item_category_cnt_month_mean)]

# dfnext = df[(df.item_category_id == 40)&(df.shop_id == 54)&(df.date_block_num >= 28)].copy()

# dfnext.loc[dfnext.index[2], 'shop_item_category_cnt_month_mean'] = 3

# dfnext.shop_item_category_cnt_month_mean.isnull().all()

# df[(df.item_category_id == 40)&(df.shop_id == 54)&(df.date_block_num >= 28)].iloc[2].shop_item_category_cnt_month_mean

# df[(df.item_category_id == 40)&(df.shop_id == 54)&(df.date_block_num >= 28)]

# df[(df.item_category_id == 40)&(df.shop_id == 54)&(df.date_block_num >= 28)].shop_item_category_cnt_month_mean.isnull().all()#&(pd.isnull(df.shop_item_category_cnt_month_mean))]
# x_set = set.drop(['item_cnt_month'], axis=1)

# y_set = set.item_cnt_month

# x_set, y_set
# y_pred = model.predict(x_set)

# y_pred, y_set
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from IPython.display import display



# mean_squared_error(y_pred, y_set)
def predictAndPlot(shop_id, item_id):

    set = df[(df['shop_id'] == shop_id)&(df['item_id']==item_id)&(df.date_block_num >= 24)].sort_values(by=['item_id', 'shop_id', 'date_block_num'], ascending=True)

    display(set[['date_block_num', 'item_cnt_month', 'item_cnt_month, t-1', 'item_cnt_month, t-2', 'item_cnt_month, t-3', 'item_cnt_month, t-4', 'item_cnt_month, t-5', 'item_cnt_month, t-6', 'item_cnt_month, t-7', 'item_cnt_month, t-8', 'item_cnt_month, t-9', 'item_cnt_month, t-10', 'item_cnt_month, t-11', 'item_cnt_month, t-12']])

#     print(set.iloc[-1])

    last_row = set.iloc[-1]

    historical_data = []

    for i in range(12, 0, -1):

        historical_data.append([last_row.date_block_num - i, last_row['item_cnt_month, t-' + str(i)]])

    display(historical_data)

    historical_data = pd.DataFrame(historical_data, columns=['date_block_num', 'item_cnt_month_historical'])

    

#     display(historical_data)

    x_set = set.drop(['item_cnt_month'], axis=1)

    y_set = set[['date_block_num', 'item_cnt_month']]

    display(x_set, y_set)



    y_pred = model.predict(x_set)

    # print(y_pred)

    y_pred = np.clip(y_pred, 0, None)

    y_pred, y_set

    # print(y_pred)



    print('mae:', mean_absolute_error(y_pred, y_set.item_cnt_month))

    print(y_pred, '\n', y_set.item_cnt_month.to_numpy())

    

    y_df = pd.DataFrame({'date_block_num': y_set.date_block_num, 'item_cnt_month_pred': y_pred, 'item_cnt_month_actual': y_set.item_cnt_month})

    y_df = historical_data.merge(y_df, on='date_block_num', how="left")

    display(y_df)

    plt.figure(predictAndPlot.iterator)

    predictAndPlot.iterator += 1

    y_df.plot(x='date_block_num')

#     plt.plot(y_pred, label="prediction")

#     plt.plot(y_set.reset_index(drop=True), label='actual')

#     plt.legend()

    plt.show()

predictAndPlot.iterator = 1
dfToPlot = df[(df.item_cnt_month > 0)&(df['item_cnt_month, t-1'] > 0)&(df['item_cnt_month, t-1'] > 0)&(df.date_block_num >= 24)] #df.item_cnt_month < 20)&

dfToPlot = dfToPlot[df.item_category_id == 21].head(20)

uniqueItemAndShops = dfToPlot[['item_id', 'shop_id']].drop_duplicates()

print(uniqueItemAndShops.shape)

for index, row in uniqueItemAndShops.iterrows():

    predictAndPlot(shop_id=row.shop_id, item_id=row.item_id)