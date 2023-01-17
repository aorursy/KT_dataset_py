import pandas as pd

import numpy as np

import missingno as msno

import seaborn as sns

import scipy as sp

import matplotlib.pyplot as plt



import warnings 

warnings.filterwarnings('ignore')



from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_squared_error as mse

from scipy.stats import randint as sp_randint
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
for df in [train, test]:

    

    # 전체부지 - 주거공간

    df['without_living'] = df['sqft_lot'] - df['sqft_living']

    

    # 주거공간 / 전체부지

    df['living_lot_ratio'] = df['sqft_living'] / df['sqft_lot']

    

    # 주거공간 / 층수

    df['living_per_floor'] = df['sqft_living'] / df['floors']

    

    # (주거공간 / 층수)의 공간 비율

    df['living_ratio'] = df['living_per_floor'] / df['sqft_lot']

    

    # 지하공간 유무

    df['has_basement'] = np.where(df['sqft_basement'] > 0, 1, 0)

    

    # zipcode 별 sqft당 평균가격

    df['price_per_sqft-ZIP'] = df['zipcode'].replace(train.groupby('zipcode').mean().price.to_dict())

    

    # 화장실 / 침실

    df['bath_per_bed'] = df['bathrooms'] / (df['bedrooms'] + 0.01)

    

    # is_renovated

    df['is_renovated'] = np.where(df['yr_renovated'] > 0, 1, 0)

    

    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    

    qcut_count = 10

    df['qcut_long'] = pd.qcut(df['long'], qcut_count, labels=range(qcut_count))

    df['qcut_lat'] = pd.qcut(df['lat'], qcut_count, labels=range(qcut_count))

    df['qcut_long'] = df['qcut_long'].astype(int)

    df['qcut_lat'] = df['qcut_lat'].astype(int)

    

    df['grade_condition'] = df['grade'] * df['condition']

    df['sqft_total'] = df['sqft_living'] + df['sqft_lot']

    df['sqft_total_size'] = df['sqft_living'] + df['sqft_lot'] + df['sqft_above'] + df['sqft_basement']
for df in [train, test]:

    

    df['year'] = df['date'].str[:4].astype(int)

    

    df['month'] = df['date'].str[4:6].astype(int)

    

    df.drop(['date'], axis=1, inplace=True)

    

    df['yr_latest'] = df[['yr_built','yr_renovated']].apply(lambda x: x.max(), axis=1)

    

    df['yr_renovated'] = df['yr_renovated'].apply(lambda x: sp.nan if x == 0 else x)

    df['yr_renovated'] = df['yr_renovated'].fillna(df['yr_built'])

    

    del df['yr_built']

    del df['yr_renovated']

train = train[train['living_ratio'] > 0.01]

train = train.loc[train['bedrooms']<10]
train['price'] = np.log1p(train['price'].values)
skew_columns = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']



for c in skew_columns:

    train[c] = np.log1p(train[c].values)

    test[c] = np.log1p(test[c].values)
train.fillna(0, inplace=True)
from sklearn.ensemble import RandomForestRegressor



test_id = test['id']

Y_test = test.drop(['id'], axis = 1, inplace = False)
y_target = train['price']

x_data = train.drop(['price', 'id'], axis = 1, inplace = False)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_target, test_size = 0.2, random_state = 42)
forest_reg = RandomForestRegressor()
def report(results, n_top=3):

    for i in range(1, n_top + 1):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")
param_dist = {"max_depth": [7, 11, 15, 18, 21],

              "max_features": sp_randint(1, len(x_train.columns)),

              "min_samples_split": sp_randint(2, 21),

              "min_samples_leaf": sp_randint(1, 21),

              "bootstrap": [True, False],

              "random_state": [42]

             }



n_iter_search = 20

random_search = RandomizedSearchCV(forest_reg, param_distributions=param_dist,

                                   n_iter=n_iter_search)
random_search.fit(x_train, y_train)
report(random_search.cv_results_)
pred = sp.special.expm1(random_search.predict(x_test))

y_test = sp.expm1(y_test)

rf_score = (mse(y_test, pred)) ** float(0.5)

print('RMSE : {0:.3F}'.format(rf_score))
rf_pred = sp.special.expm1(random_search.predict(Y_test))
import lightgbm as lgb
y_target = train['price']

X_data = train.drop(['price', 'id'], axis = 1, inplace = False)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size = 0.2, random_state = 42)



model_lgb=lgb.LGBMRegressor(

                           learning_rate=0.001,

                           n_estimators=100000,

                           subsample=0.6,

                           colsample_bytree=0.6,

                           reg_alpha=0.2,

                           reg_lambda=10,

                           num_leaves=35,

                           silent=True,

                           min_child_samples=10,

                            

                           )



model_lgb.fit(X_train,y_train,eval_set=(X_test,y_test),verbose=0,early_stopping_rounds=1000,

              eval_metric='rmse')



lgbm_score=mse(sp.special.expm1(model_lgb.predict(X_test)),sp.special.expm1(y_test))**0.5

lgbm_pred = sp.special.expm1(model_lgb.predict(Y_test))

print("RMSE unseen : {}".format(lgbm_score))
fig, ax = plt.subplots(figsize=(10,10))

lgb.plot_importance(model_lgb, ax=ax)

plt.show()
import xgboost as xgb



test_id = test['id']

Y_test = test.drop(['id'], axis = 1, inplace = False)

y_target = train['price']

X_data = train.drop(['price', 'id'], axis = 1, inplace = False)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size = 0.2, random_state = 42)

watchlist=[(X_train,y_train),(X_test,y_test)]
Y_test = Y_test[X_test.columns]
model_xgb= xgb.XGBRegressor(tree_method='gpu_hist',

                        n_estimators=100000,

                        num_round_boost=500,

                        show_stdv=False,

                        feature_selector='greedy',

                        verbosity=0,

                        reg_lambda=10,

                        reg_alpha=0.01,

                        learning_rate=0.001,

                        seed=42,

                        colsample_bytree=0.8,

                        colsample_bylevel=0.8,

                        subsample=0.8,

                        n_jobs=-1,

                        gamma=0.005,

                        base_score=np.mean(y_target)

                        )
model_xgb.fit(X_train,y_train, verbose=False, eval_set=watchlist,

              eval_metric='rmse',

              early_stopping_rounds=1000)
xgb_score=mse(np.exp(model_xgb.predict(X_test)),np.exp(y_test))**0.5

xgb_pred=np.exp(model_xgb.predict(Y_test))



print("RMSE unseen : {}".format(xgb_score))
fig, ax = plt.subplots(figsize=(10,10))

xgb.plot_importance(model_xgb, ax=ax)

plt.show()
score=lgbm_score+rf_score+xgb_score

lgbm_ratio=lgbm_score/score

rf_ratio=rf_score/score

xgb_ratio=xgb_score/score

predict=lgbm_pred*(lgbm_ratio)+rf_pred*(rf_ratio)+xgb_pred*(xgb_ratio)

print('rf_ratio={}, lgbm_ratio={}, xgb_ratio={}'.format(rf_ratio,lgbm_ratio, xgb_ratio))

submission=pd.read_csv('../input/sample_submission.csv')

submission.loc[:,'price']=predict

submission.to_csv('submission.csv',index=False)