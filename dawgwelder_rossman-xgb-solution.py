# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("."))

# Any results you write to the current directory are saved as output.
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
df_store = pd.read_csv('../input/store.csv')
df = pd.read_csv('../input/train.csv', low_memory=False)

df = df.merge(df_store, on='Store')
df_test = pd.read_csv('../input/test.csv', low_memory=False)
df_test.head()
df.head(5)
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df.Date.apply(lambda dt: dt.month)
df['Year'] = df.Date.apply(lambda dt: dt.year)
df['WeekOfYear'] = df.Date.apply(lambda dt: dt.weekofyear)
df['Day'] = df.Date.apply(lambda dt: dt.day)

df['isMonthEnd'] = df.Date.apply(lambda dt: dt.is_month_end)
df['isMonthStart'] = df.Date.apply(lambda dt: dt.is_month_start)
df['isQuarterEnd'] = df.Date.apply(lambda dt: dt.is_quarter_end )
df['isQuarterStart'] = df.Date.apply(lambda dt: dt.is_quarter_start)
df['isYearEnd'] = df.Date.apply(lambda dt: dt.is_year_end)
df['isYearStart'] = df.Date.apply(lambda dt: dt.is_year_start)
features = []
for feat in df.columns.drop('Sales'):
    if df[feat].dtype == np.float64 or df[feat].dtype == np.int64:
        features.append(feat)
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20));
df_sample = df.sample(frac=0.05)

for idx, feature in enumerate(features):
    df_sample.plot(feature, "Sales", subplots=True, kind="scatter", ax=axes[idx // 4, idx % 4]);
import gc 

del df_sample
gc.collect()
# сильно выраженная линейная корреляция между Customers и Sales (а так же Open/Promo). Но прогнозирование количества покупателей - отдельная задача
# Promo2 возможно не столь хороша в целом

df[df.columns.drop('Sales')].corrwith(df.Sales)
# Тип магазина "b" почти в два раза увеличивает продажи 

df.groupby('StoreType')['Sales'].mean()
sns.distplot(df.Sales[df.Sales > 0])
df.info()
# нет пропущенных данных для Promo2, если нет значений для ~SinceWeek или ~SinceYear
df[(pd.isnull(df.Promo2SinceWeek) | pd.isnull(df.Promo2SinceYear)) & df.Promo2 != 0]
df['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
df['CompetitionOpenSinceYear'].fillna(0, inplace=True)
df['Promo2SinceWeek'].fillna(0, inplace=True)
df['Promo2SinceYear'].fillna(0, inplace=True)
df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
df['CompetitionDistance'] = np.log(df.CompetitionDistance) + 1
df.sample(frac=.001).plot('CompetitionDistance', "Sales", subplots=True, kind="scatter")
# учитывается лишь один магазин конкурентов поблизости, так что нет возможности ввести дополнительный предиктор о количестве конкурентов
df.groupby('Store')['CompetitionDistance'].unique().apply(lambda l: 1 if len(l) > 1 else 0).sum()
# большинство магазинов закрыто по праздникам, да и весомой разницы между ними в продажах не оказалось
df['StateHoliday'] = df['StateHoliday'].replace(0, '0')
df['Holiday'] = df.StateHoliday.apply(lambda x: 0 if x == '0' else 1)

df.drop('StateHoliday', axis=1, inplace=True)
df = df.sort_values(by='Date')
df.drop('Date', axis=1, inplace=True)
df = df[(df['Open'] != 0) & (df['Sales'] != 0)]
df.drop('Open', axis=1, inplace=True)
# можно интерпретировать как категориальную фичу 

df.PromoInterval.value_counts()
df['isMonthEnd'] = df['isMonthEnd'].astype(int)
df['isMonthStart'] = df['isMonthStart'].astype(int)
df['isQuarterEnd'] = df['isQuarterEnd'].astype(int)
df['isQuarterStart'] = df['isQuarterStart'].astype(int)
df['isYearEnd'] = df['isYearEnd'].astype(int)
df['isYearStart'] = df['isYearStart'].astype(int)
# competition open time (in months)
df['CompetitionOpen'] = 12 * (df.Year - df.CompetitionOpenSinceYear) + \
        (df.Month - df.CompetitionOpenSinceMonth)
    
# Promo open time
df['PromoOpen'] = 12 * (df.Year - df.Promo2SinceYear) + \
        (df.WeekOfYear - df.Promo2SinceWeek) / 4.0

df = pd.get_dummies(df, columns=['DayOfWeek', 'StoreType', 'Assortment','PromoInterval'], dummy_na=True)
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score, make_scorer

def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w


def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe


def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe
# from sklearn.ensemble import RandomForestRegressor

# rfr = RandomForestRegressor(n_estimators=300, criterion='mae', max_depth=12, n_jobs=-1, verbose=True)
# rfr.fit(X_train.values, np.log(y_train.values) + 1)

# y_hat = rfr.predict(X_test.values)
# y_hat = np.exp(y_hat) - 1

# print(f'MAE: {mae(y_test, y_hat)}')
# print(f'RMSPE: {rmspe(y_hat, y_test)}')
import xgboost as xgb
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def train(index, train, hp_selection=False):
    train_store = train[index]
    X = train_store[train_store.columns.drop(['Sales', 'Store', 'Customers'])]
    y = train_store['Sales']

    train_size = int(X.shape[0]*.99)
    print(f'Regressor for {index} store\nTraining on {X.shape[0]} samples')
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]

    xtrain = xgb.DMatrix(X_train, np.log(y_train.values) + 1)
    xtest = xgb.DMatrix(X_test, np.log(y_test.values) + 1)
    
    if hp_selection:
        def score(params):
            num_round = 200
            model = xgb.train(params, xtrain, num_round, feval=rmspe_xg)
            predictions = model.predict(xtest)
            score = rmspe(y=y_test, yhat=predictions)
            return {'loss': score, 'status': STATUS_OK}

        def optimize(trials):
            space = {
                     'n_estimators' : hp.quniform('n_estimators', 1, 1000, 1),
                     'eta' : hp.quniform('eta', 0.2, 0.825, 0.025),
                     'max_depth' : hp.choice('max_depth', np.arange(1, 14, dtype=int)),
                     'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),
                     'subsample' : hp.quniform('subsample', 0.7, 1, 0.05),
                     'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
                     'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
                     'eval_metric': 'rmse',
                     'objective': 'reg:linear',
                     'nthread': 4,
                     'silent' : 1
                     }

            best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=250)
            return best
        
        trials = Trials()
        best_opts = optimize(trials)
        best_opts['silent'] = 1
    else:
        best_opts = {'colsample_bytree': 0.7, 
                  'eta': 0.625, 
                  'gamma': 0.8, 
                  'max_depth': 6,
                  'eval_metric': 'rmse',
                  'min_child_weight': 6.0, 
                  'n_estimators': 8.0,  # 585
                  'silent': 1,
                  'nthread': 4,
                  'subsample': 0.95}
        
    watchlist = [(xtrain, 'train'), (xtest, 'eval')]
    num_round = 10000
    regressor = xgb.train(best_opts, xtrain, num_round, watchlist, feval=rmspe_xg,
                          verbose_eval=10, early_stopping_rounds=50)
    print("Validating")
    train_probs = regressor.predict(xtest)
    indices = train_probs < 0
    train_probs[indices] = 0
    error = rmspe(np.exp(train_probs) - 1, y_test.values)
    print('error', error)
    regressor = xgb.train(best_opts, xtest, 10, feval=rmspe_xg, xgb_model=regressor)
    return regressor
# params = {'colsample_bytree': 0.7000000000000001, 
#           'eta': 0.625, 
#           'gamma': 0.8, 
#           'max_depth': 6,
#           'eval_metric': 'rmse',
#           'min_child_weight': 6.0, 
#           'n_estimators': 8.0,  # 585
#           'silent': 1,
#           'subsample': 0.9500000000000001}


# watchlist = [(xtrain, 'train'), (xtest, 'eval')]
# num_round = 10000
# xgb_regressor = xgb.train(params, xtrain, num_round, watchlist, feval=rmspe_xg,
#                           verbose_eval=10, early_stopping_rounds=50)
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 20));
# xgb.plot_importance(xgb_regressor, axes)
# print("Validating")
# train_probs = xgb_regressor.predict(xtest)
# indices = train_probs < 0
# train_probs[indices] = 0
# error = rmspe(np.exp(train_probs) - 1, y_test.values)
# print('error', error)

# xgb_regressor = xgb.train(params, xtest, 1000, feval=rmspe_xg, xgb_model=xgb_regressor)
df_test = pd.read_csv('../input/test.csv', low_memory=False)
closed_store_ids = df_test["Id"][df_test["Open"] == 0].values

df_test = df_test.merge(df_store, on='Store')
df_test['Date'] = pd.to_datetime(df_test['Date'])
df_test['Month'] = df_test.Date.apply(lambda dt: dt.month)
df_test['Year'] = df_test.Date.apply(lambda dt: dt.year)
df_test['WeekOfYear'] = df_test.Date.apply(lambda dt: dt.weekofyear)
df_test['Day'] = df_test.Date.apply(lambda dt: dt.day)

df_test['isMonthEnd'] = df_test.Date.apply(lambda dt: dt.is_month_end).astype(int)
df_test['isMonthStart'] = df_test.Date.apply(lambda dt: dt.is_month_start).astype(int)
df_test['isQuarterEnd'] = df_test.Date.apply(lambda dt: dt.is_quarter_end ).astype(int)
df_test['isQuarterStart'] = df_test.Date.apply(lambda dt: dt.is_quarter_start).astype(int)
df_test['isYearEnd'] = df_test.Date.apply(lambda dt: dt.is_year_end).astype(int)
df_test['isYearStart'] = df_test.Date.apply(lambda dt: dt.is_year_start).astype(int)

df_test['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
df_test['CompetitionOpenSinceYear'].fillna(0, inplace=True)

df_test['Promo2SinceWeek'].fillna(0, inplace=True)
df_test['Promo2SinceYear'].fillna(0, inplace=True)

df_test['CompetitionDistance'].fillna(df_test['CompetitionDistance'].median(), inplace=True)

df_test['StateHoliday'] = df_test['StateHoliday'].replace(0, '0')
df_test['Holiday'] = df_test.StateHoliday.apply(lambda x: 0 if x == '0' else 1)

df_test.drop('StateHoliday', axis=1, inplace=True)
df_test.drop('Date', axis=1, inplace=True)

# competition open time (in months)
df_test['CompetitionOpen'] = 12 * (df_test.Year - df_test.CompetitionOpenSinceYear) + \
        (df_test.Month - df_test.CompetitionOpenSinceMonth)
    
# Promo open time
df_test['PromoOpen'] = 12 * (df_test.Year - df_test.Promo2SinceYear) + \
        (df_test.WeekOfYear - df_test.Promo2SinceWeek) / 4.0

df_test.drop(['Open'], axis=1, inplace=True)

df_test = pd.get_dummies(df_test, columns=['DayOfWeek', 'StoreType', 'Assortment','PromoInterval'], dummy_na=True)

store_grouped = dict(list(df.groupby('Store')))
test_grouped = dict(list(df_test.groupby('Store')))
submission = pd.Series(np.zeros(df_test.Id.shape))
submission.index += 1

for store in test_grouped:
    test = test_grouped[store].copy()
    ids = test['Id']
    dpred = xgb.DMatrix(test[test.columns.drop(['Id', 'Store'])]) 
    regressor = train(store, store_grouped)
    preds = regressor.predict(dpred)
    preds[preds < 0] = 0
    preds = np.exp(preds) - 1
    submission[ids] = preds

submission[closed_store_ids] = 0
submission.head()
df_submission = pd.DataFrame()
df_submission['Id'] = submission.index
df_submission['Sales'] = submission.values
df_submission
df_submission.to_csv('submission.csv', index=False)
# def score(params):
#     print("Training with params : ")
#     print(params)
#     num_round = int(params['n_estimators'])
#     model = xgb.train(params, xtrain, num_round, feval=rmspe_xg)
#     predictions = model.predict(xtest)
#     score = rmspe(y=y_test, yhat=predictions)
#     br = '-'*124
#     print(f'{br}\n\tScore of RMSPE: {score}\n{br}')
#     return {'loss': score, 'status': STATUS_OK}

# def optimize(trials):
#     space = {
#              'n_estimators' : hp.quniform('n_estimators', 1, 1000, 1),
#              'eta' : hp.quniform('eta', 0.3, 0.825, 0.025),
#              'max_depth' : hp.choice('max_depth', np.arange(1, 14, dtype=int)),
#              'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),
#              'subsample' : hp.quniform('subsample', 0.7, 1, 0.05),
#              'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
#              'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
#              'eval_metric': 'rmse',
#              'objective': 'reg:linear',
#              'nthread': 4,
#              'silent' : 1
#              }

#     best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=250)

#     print(best)
#     return best

    
# trials = Trials()
# best_opts = optimize(trials)

# print(best_opts)
# def score(params):
#     print("Training with params : ")
#     print(params)
#     num_round = 25  # int(params['n_estimators'])
#     # del params['n_estimators']
#     dtrain = xgb.DMatrix(X_train, label=y_train)
#     dvalid = xgb.DMatrix(X_test, label=y_test)
#     model = xgb.train(params, dtrain, num_round)
#     predictions = model.predict(dvalid)
#     score = mae(y_test, predictions)
#     br = '-'*130
#     print(f'{br}\n\tScore of MAE: {score}\n{br}')
#     return {'loss': score, 'status': STATUS_OK}

# def optimize(trials):
#     space = {
#              'n_estimators' : hp.quniform('n_estimators', 100, 1000, 1),
#              'eta' : hp.quniform('eta', 0.4, 0.825, 0.025),
#              'max_depth' : hp.choice('max_depth', np.arange(1, 14, dtype=int)),
#              'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),
#              'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
#              'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
#              'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
#              'eval_metric': 'mae',
#              'objective': 'reg:linear',
#              'nthread': 4,
#              'silent' : 1
#              }

#     best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=50)

#     print(best)
    
# trials = Trials()
# optimize(trials)