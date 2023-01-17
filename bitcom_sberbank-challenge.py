%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
prefix = '../input/'
transactions = pd.read_csv(prefix + 'transactions/transactions.csv')
train_transactions = transactions[transactions.amount < 0].copy()
train_transactions['day'] = train_transactions.tr_datetime.apply(lambda dt: dt.split()[0]).astype(int)
from pylab import rcParams
rcParams['axes.xmargin'] = 0.02
rcParams['axes.ymargin'] = 0.02
def draw_customer_code(customer, mcc, predicted=None):
    table = train_transactions[(train_transactions.customer_id == customer) \
                               & (train_transactions.mcc_code == mcc)] \
            [['day', 'amount']].groupby('day').sum()
    plt.figure(figsize=(15,7))
    plt.plot(table.index, np.log1p(-table.values.flatten()))
    if predicted is not None:    
        plt.plot(table.index, np.log1p(predicted))
    
    
# for code in train_transactions.mcc_code.unique():
# draw_customer_code(7391977, 4814)
test_transactions = pd.DataFrame(columns=train_transactions.mcc_code.unique(), 
                                 index=np.arange(1, 31) + train_transactions.day.max())
test_transactions = test_transactions.unstack().reset_index().dropna(axis=1)
test_transactions.columns = ['mcc_code', 'day']
train_grid = pd.DataFrame(columns=train_transactions.mcc_code.unique(), 
                          index=train_transactions.day.unique())
train_grid = train_grid.unstack().reset_index().dropna(axis=1)
train_grid.columns = ['mcc_code', 'day']
for tr_table in [train_transactions, test_transactions, train_grid]:
    tr_table['week_num'] = tr_table['day'] // 7
    tr_table['week_day'] = tr_table['day'] % 7
    tr_table['month_num'] = tr_table['day'] // 30
    tr_table['month_day'] = tr_table['day'] % 30
train_transactions = \
    pd.merge(train_grid,
             train_transactions.groupby(['day', 'week_num', 'week_day', 'month_num', 'month_day', 'mcc_code'])[['amount']]\
                 .sum().reset_index(),
             how='left').fillna(0)
def copies():
    return train_transactions.copy(), test_transactions.copy()
def do_month_shift(train, test):
    shifted = train.copy()
    shifted['month_num'] += 1
    
    group_columns = ['mcc_code', 'month_num']
    shifted = shifted[group_columns + ['amount']].groupby(group_columns, as_index=False).mean()
    shifted.rename(columns={'amount': 'month_shift_mean'}, inplace=True)
    
    train = pd.merge(train, shifted, on=group_columns, how='left').fillna(0)
    test = pd.merge(test, shifted, on=group_columns, how='left').fillna(0)
    
    return train, test
def do_days_shift(train, test, days, week=False):
    concat = []
    
    group_columns = ['mcc_code', 'day']
    for day_shift in range(7 if week else 1, days + 1, 7 if week else 1):
        x = train[group_columns + ['amount']].copy()
        x['day'] += day_shift
        concat.append(x)
        
    
    week_str = 'week_' if week else ' '
    shifted = pd.concat(concat, ignore_index=True).groupby(group_columns, as_index=False).mean()
    shifted.rename(columns={'amount': '{}_{}days_shift_mean'.format(days, week_str)}, inplace=True)
    
    train = pd.merge(train, shifted, on=group_columns, how='left').fillna(0)
    test = pd.merge(test, shifted, on=group_columns, how='left').fillna(0)
    
    return train, test
def do_zero_days_before(train, test):
    s = train
    s['zero_before'] = s.amount == 0
    group_columns = ['mcc_code', 'day']
    r = (s[group_columns + ['zero_before']].groupby(group_columns).sum().groupby(level=[0]).cumsum() - 
        s[group_columns + ['zero_before']].groupby(group_columns).sum())
    r.reset_index(inplace=True)
    r.zero_before /= r.day
    r.fillna(0, inplace=True)
    train.drop(columns='zero_before', inplace=True)
    
    train = pd.merge(train, r, on=group_columns, how='left').fillna(0)
    test = pd.merge(test, r[r.day == r.day.max()].drop(columns='day'), on=['mcc_code'], how='left')
    
    return train, test
reverse_by_sum = {}
def change_mcc_by_total_amount(train, test):
    by_sum = {}
    global reverse_by_sumer
    reverse_by_sum = {}
    
    def calc_maps():
        x = train_transactions[['mcc_code', 'amount']].groupby('mcc_code', as_index=False).sum()
        for idx, code in enumerate(np.array(sorted(zip(x.mcc_code, x.amount), key=lambda s: s[1], reverse=True))[:, 0].astype(int)):
            by_sum[code] = idx + 1
            reverse_by_sum[idx + 1] = code

    calc_maps()
    for df in (train, test):
        df['mcc_by_total_amount'] = df.mcc_code.map(by_sum)
    return train, test
reverse_by_nonzero = {}
def change_mcc_by_nonzero(train, test):
    by_nonzero = {}
    global reverse_by_nonzero
    reverse_by_nonzero = {}
    
    def calc_maps():
        x = train_transactions[['mcc_code', 'amount']].groupby('mcc_code', as_index=False).agg(np.nonzero)
        for idx, code in enumerate(np.array(sorted(zip(x.mcc_code, x.amount.map(lambda x: - x[0].shape[0] / 457)), key=lambda s: s[1]))[:, 0].astype(int)):
            by_nonzero[code] = idx + 1
            reverse_by_nonzero[idx + 1] = code

    
    calc_maps()
    for df in (train, test):
        df['mcc_by_nonzero'] = df.mcc_code.map(by_nonzero)
    return train, test
def PipelineData(train, test):
#     train, test = do_month_shift(train, test)
    train, test = do_zero_days_before(train, test)
    train, test = do_days_shift(train, test, 7)
    train, test = do_days_shift(train, test, 30)
    train, test = do_days_shift(train, test, 90)
    train, test = do_days_shift(train, test, 180)
    train, test = do_days_shift(train, test, 7, week=True)
    train, test = do_days_shift(train, test, 30, week=True)
    train, test = do_days_shift(train, test, 90, week=True)
    train, test = do_days_shift(train, test, 180, week=True)
    
    return train, test
train, test = copies()

train, test = change_mcc_by_total_amount(train, test)
# train, test = change_mcc_by_nonzero(train, test)

train.amount = np.log1p(-train.amount)

train, test = PipelineData(train, test)
# for day_shift in [-1, 0, 1]:
#     for month_shift in train_transactions.month_num.unique()[1:]:
#         train_shift = train_transactions.copy()
#         train_shift['month_num'] += month_shift
#         train_shift['month_day'] += day_shift
#         train_shift['amount_day_{}_{}'.format(day_shift, month_shift)] = np.log1p(-train_shift['amount'])
#         train_shift = train_shift[['month_num', 'month_day', 'mcc_code', 'amount_day_{}_{}'.format(day_shift, month_shift)]]

#         train_transactions = pd.merge(train_transactions, train_shift, 
#                                       on=['month_num', 'month_day', 'mcc_code'], how='left').fillna(0)
#         test_transactions = pd.merge(test_transactions, train_shift, 
#                                      on=['month_num', 'month_day', 'mcc_code'], how='left').fillna(0)
dummy = ['mcc_code', 'week_day']
train = pd.get_dummies(train, columns=dummy)
test = pd.get_dummies(test, columns=dummy)
# train = train_transactions
# test = test_transactions
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn import metrics
from sklearn.metrics import mean_squared_error

from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

import sys
def make_cv(X, n=1, days=30, b=457, start=0):
    cv = []
    a = b - days * n
    for i in range(a, b - 1, days):
        cv.append((
            X[X.day.isin(range(start, i))].index.values.astype(int),
            X[X.day.isin(range(i, i + days))].index.values.astype(int)            
#             X.day.isin(range(i)),
#             X.day.isin(range(i, i + days))
        ))
    return cv
start = 187
c = train.columns.difference(['amount'])
X = train[c]
y = train['amount']


cv = make_cv(train, start=start)
clf = XGBRegressor(n_estimators=1000, 
                   max_depth=7, min_child_weight=2, 
                   subsample=1, learning_rate=0.02, 
                   gamma=0.1, colsample=0.7, booster='gbtree', 
                   seed=33, reg_alpha=26
                  )
clf = LinearRegression()
# clf = XGBRegressor(n_estimators=140, learning_rate=0.1)
# clf = XGBRegressor(n_estimators=140, max_depth=7, min_child_weight=2, subsample=1, learning_rate=0.1, gamma=0.1, colsample=0.7)
# %%time
# cv = make_cv(train, start=187)

# pr = -cross_val_score(clf, X, y, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')
# pr = np.sqrt(pr)
# print(pr)
# print(np.mean(pr))
# clf.fit(X, y)
sorted(list(zip(clf.feature_importances_, X.columns)), reverse=True)
tr_idx, tst_idx = cv[0]
X_train, X_test, y_train, y_test = X.iloc[tr_idx], X.iloc[tst_idx], y.iloc[tr_idx], y.iloc[tst_idx]

# for d in range(10, 2):
#     clf = CatBoostRegressor(n_estimators=130, silent=True, learning_rate=0.1, eval_metric='RMSE', 
#                         loss_function='RMSE', depth=d
#                            )
#     clf.fit(X_train, y_train, eval_set=(X_test, y_test))
#     results.append(unpack(clf))
def unpack(clf):
    print(clf.best_iteration_)
    c = []
    for i in clf.best_score_.values():
        c.append(list(i.values())[0])
    return c[0], c[1]
clf = CatBoostRegressor(n_estimators=1000, silent=True, learning_rate=0.01, eval_metric='RMSE', 
                        loss_function='RMSE', depth=9)
# clf = CatBoostRegressor(n_estimators=110, silent=True, learning_rate=0.1, eval_metric='RMSE', 
#                         loss_function='RMSE', depth=9)
# clf.fit(X_train, y_train, plot=True, eval_set=(X_test, y_test))
clf = XGBRegressor(n_estimators=2000, learning_rate=0.0035, seed=33, max_depth=3, min_child_weight=4, gamma=0.5,
                  colsample_bytree=0.9, subsample=0.6, reg_alpha=0.13)
# clf = XGBRegressor(n_estimators=60, learning_rate=0.1, seed=33)
# from numpy import loadtxt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# # load data

# model = clf
# eval_set = [(X_train, y_train), (X_test, y_test)]
# model.fit(X_train, y_train, eval_metric=["rmse"], eval_set=eval_set, verbose=True, early_stopping_rounds=20)
# # make predictions for test data
# y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]
# # evaluate predictions
# accuracy = metrics.mean_squared_error(y_test, predictions)
# print("RMSE: %.2f%%" % (accuracy))
# # retrieve performance metrics
# results = model.evals_result()
# epochs = len(results['validation_0']['rmse'])
# x_axis = range(0, epochs)
# # plot log loss
# fig, ax = plt.subplots()
# start = 0
# ax.plot(x_axis[start:], results['validation_0']['rmse'][start:], label='Train')
# ax.plot(x_axis[start:], results['validation_1']['rmse'][start:], label='Test')
# ax.legend()
# plt.ylabel('Rmse')
# plt.title('XGBoost Rmse')
# plt.show()
# fig, ax = plt.subplots()
# start = 500
# ax.plot(x_axis[start:], results['validation_0']['rmse'][start:], label='Train')
# ax.plot(x_axis[start:], results['validation_1']['rmse'][start:], label='Test')
# ax.legend()
# plt.ylabel('Rmse')
# plt.title('XGBoost Rmse')
# plt.show()
# clf = XGBRegressor(n_estimators=600, learning_rate=0.01, seed=33, max_depth=3, min_child_weight=4,
#                   colsample_bytree=1, subsample=0.7, colsample_bylevel=0.9, reg_alpha=1.2)
from sklearn.model_selection import GridSearchCV
tr = train[c][train['day'] >= start]
clf.fit(train[c][train['day'] >= start], train[train['day'] >= start].amount)
test_transactions['volume'] = np.expm1(clf.predict(test[c]))
test_transactions['id'] = test_transactions[['mcc_code', 'day']].apply(lambda x: '-'.join(map(str, x)), axis=1)
test_transactions[['id', 'volume']].to_csv('baseline.csv', index=False)