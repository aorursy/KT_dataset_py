import numpy as np 

import pandas as pd 



import os

print(os.listdir("../input"))



from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics



import math

import matplotlib.pyplot as plt
train = pd.read_csv('../input/sales_train.csv')

test = pd.read_csv('../input/test.csv')

submission = pd.read_csv('../input/sample_submission.csv')

items = pd.read_csv('../input/items.csv')

item_cats = pd.read_csv('../input/item_categories.csv')

shops = pd.read_csv('../input/shops.csv')
train.head()
test.head()
train_agg = train.drop(['date', 'item_price'], axis=1)

train_agg.describe()
# Sum up item_cnt_day grouped by shop_id and item_id and date_block_num to get unique rows of items sold per month. 



df = train_agg.groupby(["shop_id", "item_id", "date_block_num"])



monthly = df.aggregate({"item_cnt_day":np.sum}).fillna(0)

monthly.reset_index(level=["shop_id", "item_id", "date_block_num"], inplace=True)

monthly = monthly.rename(columns={ monthly.columns[3]: "item_cnt_month" })
monthly.describe()
monthly['item_id'].value_counts()/34
test['item_id'].value_counts()
monthly['shop_id'].loc[monthly['item_id'] == 5822].value_counts().sort_index()
test['shop_id'].loc[test['item_id'] == 5822].value_counts().sort_index()
monthly['shop_id'].value_counts()
test['shop_id'].value_counts()
monthly.describe()
test.describe()
train_simple = monthly.drop('date_block_num', axis=1)

#shuffle rows

train_simple = train_simple.sample(frac=1).reset_index(drop=True)



X_simple = train_simple[['shop_id', 'item_id']]

y_simple = train_simple['item_cnt_month']
def split_vals(a,n): return a[:n].copy(), a[n:].copy()



n_valid = 214200

n_trn = len(train_simple) - n_valid

X_train, X_valid = split_vals(X_simple, n_trn)

y_train, y_valid = split_vals(y_simple, n_trn)
plt.scatter(X_valid.iloc[:100,1], y_valid[:100], color='black')
m = RandomForestRegressor(n_estimators=1, n_jobs=-1)

%time m.fit(X_train, y_train)
def rmse(x,y): return math.sqrt(((x-y)**2).mean())



def print_score(m):

    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),

                m.score(X_train, y_train), m.score(X_valid, y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)



print_score(m)
plt.scatter(X_valid.iloc[:100,1], m.predict(X_valid)[:100], color='black')
m_2 = RandomForestRegressor(n_estimators=100, n_jobs=-1)

%time m_2.fit(X_train, y_train)

%time print_score(m_2)
preds = np.stack([t.predict(X_valid) for t in m_2.estimators_])
plt.plot([metrics.mean_squared_error(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(100)]);
X_simple = train_simple[['shop_id', 'item_id']]

y_simple = train_simple['item_cnt_month'].clip(0,20)
n_valid = 214200

n_trn = len(train_simple) - n_valid

X_train, X_valid = split_vals(X_simple, n_trn)

y_train, y_valid = split_vals(y_simple, n_trn)
m = RandomForestRegressor(n_estimators=1, n_jobs=-1)

%time m.fit(X_train, y_train)

print_score(m)
m_2 = RandomForestRegressor(n_estimators=100, n_jobs=-1)

%time m_2.fit(X_train, y_train)

print_score(m_2)
preds = np.stack([t.predict(X_valid) for t in m_2.estimators_])

plt.plot([metrics.mean_squared_error(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(100)]);
pd.DataFrame(m_2.predict(X_valid)).describe()
train_td = monthly.sort_values(by=["date_block_num"])

valid_td = monthly[monthly["date_block_num"] == 33]



X_train = train_td[['shop_id', 'item_id']]

y_train = train_td['item_cnt_month'].clip(0,20)

X_valid = valid_td[['shop_id', 'item_id']]

y_valid = valid_td['item_cnt_month'].clip(0,20)
m_3 = RandomForestRegressor(n_estimators=60, n_jobs=-1)

%time m_3.fit(X_train, y_train)

print_score(m_3)
preds = np.stack([t.predict(X_valid) for t in m_3.estimators_])

plt.plot([metrics.mean_squared_error(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(60)]);
import xgboost as xgb

param = {'max_depth':12,  # originally 10

         'subsample':1,  # 1

         'min_child_weight':0.5,  # 0.5

         'eta':0.3,

         'num_round':1000, 

         'seed':0,  # 1

         'silent':0,

         'eval_metric':'rmse',

         'early_stopping_rounds':100

        }



progress = dict()

xgbtrain = xgb.DMatrix(X_train, y_train)

watchlist  = [(xgbtrain,'train-rmse')]

m_4 = xgb.train(param, xgbtrain)
preds = m_4.predict(xgb.DMatrix(X_valid))



rmse = np.sqrt(mean_squared_error(preds, y_valid))

print(rmse)
new_submission = pd.merge(month_sum, test, how='right', left_on=['shop_id','item_id'], right_on = ['shop_id','item_id']).fillna(0)

new_submission.drop(['shop_id', 'item_id'], axis=1)

new_submission = new_submission[['ID','item_cnt_month']]
new_submission['item_cnt_month'] = new_submission['item_cnt_month'].clip(0,20)

new_submission.describe()
new_submission.to_csv('previous_value_benchmark.csv', index=False)