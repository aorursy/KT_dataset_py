import pandas as pd
import numpy as np
import datetime 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import ensemble, metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
pd.options.display.float_format = '{:,.3f}'.format
parser = lambda date:pd.to_datetime(date,format='%d.%m.%Y')
sales = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv",parse_dates = ['date'],date_parser = parser)
items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")
item_cat = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")
shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")
test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")

print(sales.shape,test.shape)
sales.head()
print (sorted(sales['shop_id'].unique()))
print (sorted(test["shop_id"].unique()))
sales_group = sales.groupby(['date_block_num','shop_id','item_id'])
sales_group.mean()

sales_monthly = pd.DataFrame(sales_group.agg({'item_cnt_day':'sum'})).reset_index()
sales_monthly.columns = ['date_block_num','shop_id','item_id','item_cnt']
print (sales_monthly[['item_cnt']])

sales_monthly['item_cnt'].clip(0,20,inplace=True)
sales_monthly
item_grp = item_cat['item_category_name'].apply(lambda x: str(x).split(' ')[0])
item_cat['item_group'] = pd.Categorical(item_grp).codes

items = pd.merge(items, item_cat.loc[:,['item_category_id','item_group']], on=['item_category_id'], how='left')

items
grp = sales_monthly.groupby(['shop_id','item_id'])
train_shop = grp.agg({'item_cnt':'mean'}).reset_index()
train_shop.columns = ['shop_id','item_id','cnt_mean_shop']
train_shop
## By shop,item_group
train_cat_monthly = pd.merge(sales_monthly, items, on=['item_id'], how='left')
grp = train_cat_monthly.groupby(['shop_id', 'item_group'])
train_shop_cat = grp.agg({'item_cnt':['mean']}).reset_index()
train_shop_cat.columns = ['shop_id','item_group','cnt_mean_shop_cat']
print(train_shop_cat.loc[:,['cnt_mean_shop_cat']].describe())

train_shop_cat
city = shops.shop_name.apply(lambda x: str.replace(x, '!', '')).apply(lambda x: x.split(' ')[0])
shops['city'] = pd.Categorical(city).codes

shops
# Price mean by month,shop,item
train_price = sales_group['item_price'].mean().reset_index()
price = train_price[~train_price['item_price'].isnull()]

# last price by shop,item
last_price = price.drop_duplicates(subset=['shop_id', 'item_id'], keep='last').drop(['date_block_num'], axis=1)
pred_price_set = test.copy()
pred_price_set.drop('ID', axis=1)
reg = ensemble.ExtraTreesRegressor(n_estimators=25, n_jobs=-1, max_depth=15, random_state=42)
reg.fit(last_price[['shop_id','item_id']], last_price['item_price'])
pred_price_set['item_price'] = reg.predict(pred_price_set[['shop_id','item_id']])

test_price = pd.concat([last_price, pred_price_set], join='inner')
pred_price_set.drop(['ID'], axis=1, inplace=True)
pred_price_set

test_price
price_max = price.groupby(['item_id']).max()['item_price'].reset_index()
price_max.rename(columns={'item_price':'item_max_price'}, inplace=True)
price_max.head()
train_price_disc = pd.merge(price, price_max, on=['item_id'], how='left')
train_price_disc['discount_rate'] = 1 - (train_price_disc['item_price'] / train_price_disc['item_max_price'])
train_price_disc.drop('item_max_price', axis=1, inplace=True)
train_price_disc
test_price_disc = pd.merge(pred_price_set, price_max, on=['item_id'], how='left')
test_price_disc.loc[test_price_disc['item_max_price'].isnull(), 'item_max_price'] = test_price_disc['item_price']
test_price_disc['discount_rate'] = 1 - (test_price_disc['item_price'] / test_price_disc['item_max_price'])
test_price_disc.drop('item_max_price', axis=1, inplace=True)
test_price_disc
def mergeFeature(df): 
  df = pd.merge(df, items, on=['item_id'], how='left').drop('item_group', axis=1)
  df = pd.merge(df, item_cat, on=['item_category_id'], how='left')
  df = pd.merge(df, shops, on=['shop_id'], how='left')

  df = pd.merge(df, train_shop, on=['shop_id','item_id'], how='left')
  df = pd.merge(df, train_shop_cat, on=['shop_id','item_group'], how='left')
  
  df.drop(['shop_name','item_name','item_category_name','item_group'], axis=1, inplace=True)
  df.fillna(0.0, inplace=True)
  return df
train_set = sales_monthly[sales_monthly['date_block_num'] >= 10]

train_set = pd.merge(train_set, train_price_disc, on=['date_block_num','shop_id','item_id'], how='left')
train_set = mergeFeature(train_set)

X_train = train_set.drop(['item_cnt'], axis=1)
Y_train = train_set['item_cnt']
print(Y_train)
# X_train.loc[X_train['discount_rate'].isnull()]
X_train.head()
test_set = test.copy()
test_set['date_block_num'] = 34

test_set = pd.merge(test_set, test_price_disc, on=['shop_id','item_id'], how='left')
test_set = mergeFeature(test_set)

price_mean = X_train['item_price'].mean()
discount_mean = X_train['discount_rate'].mean()
print(price_mean)
X_test = test_set.drop(['ID'], axis=1)
# X_test['item_price'].fillna(price_mean, inplace=True)
# X_test['discount_rate'].fillna(discount_mean, inplace=True)
assert(X_train.columns.isin(X_test.columns).all())
X_test
X_train
X_test
rfr = RandomForestRegressor().fit(X_train, Y_train.ravel())

rmse_rfr = np.sqrt(mean_squared_error(Y_train, rfr.predict(X_train)))
print('RFR RMSE: %.4f' % rmse_rfr)

y_pred = rfr.predict(X_test)

result = pd.DataFrame({
    "ID": test["ID"],
    "item_cnt_month": y_pred
})
result.head(30)
from sklearn import linear_model, preprocessing
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

params={'learning_rate': 0.05,
        'objective':'regression',
        'metric':'rmse',
        'num_leaves': 64,
        'verbose': 1,
        'random_state':42,
        'bagging_fraction': 1,
        'feature_fraction': 1
       }

folds = GroupKFold(n_splits=6)
oof_preds = np.zeros(X_train.shape[0])
sub_preds = np.zeros(X_test.shape[0])

for fold_, (trn_, val_) in enumerate(folds.split(X_train, Y_train, X_train['date_block_num'])):
    trn_x, trn_y = X_train.iloc[trn_], Y_train[trn_]
    val_x, val_y = X_train.iloc[val_], Y_train[val_]

    reg = lgb.LGBMRegressor(**params, n_estimators=3000)
    reg.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=50, verbose=500)
    
    oof_preds[val_] = reg.predict(val_x.values, num_iteration=reg.best_iteration_)
    sub_preds += reg.predict(X_test.values, num_iteration=reg.best_iteration_) / folds.n_splits
result = pd.DataFrame({
    "ID": test["ID"],
    "item_cnt_month": sub_preds
})
result.to_csv("submission.csv", index=False)
result.head(20)
result.to_csv("Hack2.csv", index=False)