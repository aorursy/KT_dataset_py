import numpy as np

import pandas as pd

import xgboost as xgb

import matplotlib.pyplot as plt
df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])

df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])

df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])



df_train.head()
# This section added:  drop crazy data points

print( df_train.life_sq.max() )

df_train.drop(df_train[df_train["life_sq"] > 7000].index, inplace=True)

print( df_train.life_sq.max() )
y_train = df_train['price_doc'].values

id_test = df_test['id']



df_train.drop(['id', 'price_doc'], axis=1, inplace=True)

df_test.drop(['id'], axis=1, inplace=True)



num_train = len(df_train)

df_all = pd.concat([df_train, df_test])

# Next line just adds a lot of NA columns (becuase "join" only works on indexes)

# but somewhow it seems to affect the result

df_all = df_all.join(df_macro, on='timestamp', rsuffix='_macro')

print(df_all.shape)



# Add month-year

month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)

month_year_cnt_map = month_year.value_counts().to_dict()

df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)



# Add week-year count

week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)

week_year_cnt_map = week_year.value_counts().to_dict()

df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)



# Add month and day-of-week

df_all['month'] = df_all.timestamp.dt.month

df_all['dow'] = df_all.timestamp.dt.dayofweek



# Other feature engineering

df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)

df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)



# Remove timestamp column (may overfit the model in train)

df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)
factorize = lambda t: pd.factorize(t[1])[0]



df_obj = df_all.select_dtypes(include=['object'])



X_all = np.c_[

    df_all.select_dtypes(exclude=['object']).values,

    np.array(list(map(factorize, df_obj.iteritems()))).T

]

print(X_all.shape)



X_train = X_all[:num_train]

X_test = X_all[num_train:]
# Deal with categorical values

df_numeric = df_all.select_dtypes(exclude=['object'])

df_obj = df_all.select_dtypes(include=['object']).copy()



for c in df_obj:

    df_obj[c] = pd.factorize(df_obj[c])[0]



df_values = pd.concat([df_numeric, df_obj], axis=1)
# Convert to numpy values

X_all = df_values.values

print(X_all.shape)



X_train = X_all[:num_train]

X_test = X_all[num_train:]



df_columns = df_values.columns
xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)

dtest = xgb.DMatrix(X_test, feature_names=df_columns)
# Uncomment to tune XGB `num_boost_rounds`



#cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,

#    verbose_eval=True, show_stdv=False)

#cv_result[['train-rmse-mean', 'test-rmse-mean']].plot()

#num_boost_rounds = len(cv_result)



num_boost_round = 489
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_round)
y_pred = model.predict(dtest)



df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})



df_sub.head()
df_sub.to_csv('sub.csv', index=False)
df_test["timestamp"] = pd.to_datetime(df_test["timestamp"])

df_test["year"]  = df_test["timestamp"].dt.year

df_test["month"] = df_test["timestamp"].dt.month

df_test["yearmonth"] = 100*df_test.year + df_test.month

test_ids = pd.DataFrame({"yearmonth":df_test.yearmonth.values,

                         "id":id_test.values })

test_data = test_ids.merge(df_sub,on="id")



test_prices = test_data[["yearmonth","price_doc"]]

test_p = test_prices.groupby("yearmonth").median()

test_p.head()
import statsmodels.api as sm



df_macro["timestamp"] = pd.to_datetime(df_macro["timestamp"])

df_macro["year"]  = df_macro["timestamp"].dt.year

df_macro["month"] = df_macro["timestamp"].dt.month

df_macro["yearmonth"] = 100*df_macro.year + df_macro.month

macmeds = df_macro.groupby("yearmonth").median()



df_train["timestamp"] = pd.to_datetime(df_train["timestamp"])

df_train["year"]  = df_train["timestamp"].dt.year

df_train["month"] = df_train["timestamp"].dt.month

df_train["yearmonth"] = 100*df_train.year + df_train.month

prices = pd.DataFrame({"yearmonth":df_train.yearmonth.values,

                       "price_doc":y_train })

p = prices.groupby("yearmonth").median()



df = macmeds.join(p)
#  Adapted from code at http://adorio-research.org/wordpress/?p=7595

#  Original post was dated May 31st, 2010

#    but was unreachable last time I tried



import numpy.matlib as ml

 

def almonZmatrix(X, maxlag, maxdeg):

    """

    Creates the Z matrix corresponding to vector X.

    """

    n = len(X)

    Z = ml.zeros((len(X)-maxlag, maxdeg+1))

    for t in range(maxlag,  n):

       #Solve for Z[t][0].

       Z[t-maxlag,0] = sum([X[t-lag] for lag in range(maxlag+1)])

       for j in range(1, maxdeg+1):

             s = 0.0

             for i in range(1, maxlag+1):       

                s += (i)**j * X[t-i]

             Z[t-maxlag,j] = s

    return Z
y_macro = df.price_doc.div(df.cpi).apply(np.log).loc[201108:201506]

nobs = 47  # August 2011 through June 2015, months with price_doc data

tblags = 5    # Number of lags used on PDL for Trade Balance

mrlags = 5    # Number of lags used on PDL for Mortgage Rate

ztb = almonZmatrix(df.balance_trade.loc[201103:201506].as_matrix(), tblags, 1)

zmr = almonZmatrix(df.mortgage_rate.loc[201103:201506].as_matrix(), mrlags, 1)

columns = ['tb0', 'tb1', 'mr0', 'mr1']

z = pd.DataFrame( np.concatenate( (ztb, zmr), axis=1), y_macro.index.values, columns )

X_macro = sm.add_constant( z )
macro_fit = sm.OLS(y_macro, X_macro).fit()
test_cpi = df.cpi.loc[201507:201605]

test_index = test_cpi.index

ztb_test = almonZmatrix(df.balance_trade.loc[201502:201605].as_matrix(), tblags, 1)

zmr_test = almonZmatrix(df.mortgage_rate.loc[201502:201605].as_matrix(), mrlags, 1)

z_test = pd.DataFrame( np.concatenate( (ztb_test, zmr_test), axis=1), test_index, columns )

X_macro_test = sm.add_constant( z_test )

pred_lnrp = macro_fit.predict( X_macro_test )

pred_p = np.exp(pred_lnrp) * test_cpi
adjust = pd.DataFrame( pred_p/test_p.price_doc, columns=["adjustment"] )

adjust
combo = test_data.join(adjust, on='yearmonth')

combo['adjusted'] = combo.price_doc * combo.adjustment

adjxgb_df = pd.DataFrame()

adjxgb_df['id'] = combo.id

adjxgb_df['price_doc'] = combo.adjusted

adjxgb_df.head()
adjxgb_df.to_csv('adjusted_xgb_predicitons.csv', index=False)