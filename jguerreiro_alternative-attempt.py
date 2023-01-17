import pandas as pd

import numpy as np



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

import xgboost as xgb

from sklearn.model_selection import train_test_split



pd.set_option('display.max_columns', 500)
macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",

"micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate","income_per_cap",

              "rent_price_4+room_bus","museum_visitis_per_100_cap","apartment_build"]





train_df = pd.read_csv('../input/train.csv',parse_dates=['timestamp'])

test_df = pd.read_csv('../input/test.csv',parse_dates=['timestamp'])

#macro_df=pd.read_csv('../input/macro.csv', parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)

macro_df=pd.read_csv('../input/macro.csv',parse_dates=['timestamp'])

train_df.shape
train_df = pd.merge_ordered(train_df, macro_df, on='timestamp', how='left')

result_df = pd.merge_ordered(test_df, macro_df, on='timestamp', how='left')

combine=[train_df,result_df]

train_df.shape
train_df[train_df.full_sq<200]['life_sq'].hist(bins=20)
train_df[(train_df.full_sq<200)&(train_df.price_doc==1000000)]['life_sq'].hist(bins=20)
#Cleaning

for df in combine:

    df.ix[df.life_sq<2, 'life_sq'] = np.nan

    df.ix[df.build_year<1500,'build_year']=np.nan

    df.ix[df.max_floor<df.floor,'max_floor']=np.nan

    df.ix[df.full_sq<2,'full_sq']=df.ix[df.full_sq<2,'life_sq']

    df.ix[df.full_sq<df.life_sq,'life_sq']=np.nan

    df.ix[df.kitch_sq>df.life_sq,'kitch_sq']=np.nan

    df.ix[df.kitch_sq<2,'kitch_sq']=np.nan

    df.ix[df.floor==0,'floor']=np.nan

    df.ix[df.max_floor==0,'max_floor']=np.nan

    df.ix[df.max_floor>70,'max_floor']=np.nan

    df.ix[df.num_room==0,'num_room']=np.nan    
#New features

for df in combine:

    #df['life_pct']=df['life_sq']/df['full_sq'].astype(float)

    df['rel_kitch']=df['kitch_sq']/df['full_sq'].astype(float)

    df['rel_floor']=df['floor']/df['max_floor'].astype(float)

   

    

    
for df in combine:

    month_year = (df.timestamp.dt.month + (df.timestamp.dt.year)*100)

    month_year_cnt_map = month_year.value_counts().to_dict()

    df['month_year_cnt'] = month_year.map(month_year_cnt_map)



    week_year = (df.timestamp.dt.weekofyear + (df.timestamp.dt.year)*100)

    week_year_cnt_map = week_year.value_counts().to_dict()

    df['week_year_cnt'] = week_year.map(week_year_cnt_map)

    df['month'] = df.timestamp.dt.month

    df['dow'] = df.timestamp.dt.dayofweek

    df['rel_kitch']=df['kitch_sq']/df['full_sq'].astype(float)

    df['rel_floor']=df['floor']/df['max_floor'].astype(float)
train_df['price_sq']=train_df['price_doc']/train_df['full_sq']

train_df.shape
train_df[['full_sq','life_sq','price_doc']][train_df.life_pct<0.05]
train_df['ecology'].dtypes
train_df_numeric = train_df.select_dtypes(exclude=['object'])

train_df_obj = train_df.select_dtypes(include=['object']).copy()



for column in train_df_obj:

    train_df_obj[column] = pd.factorize(train_df_obj[column])[0]



train_df_values = pd.concat([train_df_numeric, train_df_obj], axis=1)[:24377]

test_df_values = pd.concat([train_df_numeric, train_df_obj], axis=1)[24377:]

all_df_values = pd.concat([train_df_numeric, train_df_obj], axis=1)
result_df_numeric = result_df.select_dtypes(exclude=['object'])

result_df_obj = result_df.select_dtypes(include=['object']).copy()



for column in result_df_obj:

      result_df_obj[column] = pd.factorize(result_df_obj[column])[0]



result_df_values = pd.concat([result_df_numeric, result_df_obj], axis=1)
bound=33000

X_train = train_df_values[(train_df_values.full_sq<1000)&

                          (train_df_values.price_sq > bound)

                         ].drop(['price_doc','id','timestamp','price_sq'],axis=1)

Y_train = np.log1p(train_df_values[(train_df_values.full_sq<1000)&

                          (train_df_values.price_sq > bound)

                                  ]['price_doc'].values.reshape(-1,1))

X_train.shape
X_train = train_df_values.drop(['price_doc','id','timestamp','price_sq'],axis=1)

Y_train = np.log1p(train_df_values['price_doc'].values.reshape(-1,1))

X_train.shape
X_test = test_df_values.drop(['price_doc','id','timestamp','price_sq'],axis=1)

Y_test = np.log1p(test_df_values['price_doc'].values.reshape(-1,1))

X_test.shape
bound=33000

X_all = all_df_values[(all_df_values.full_sq<1000)&

                          (all_df_values.price_sq > bound)

                         ].drop(['price_doc','id','timestamp','price_sq'],axis=1)

Y_all = np.log1p(all_df_values[(all_df_values.full_sq<1000)&

                          (all_df_values.price_sq > bound)

                                  ]['price_doc'].values.reshape(-1,1))

X_all.shape
X_all = all_df_values.drop(['price_doc','id','timestamp','price_sq'],axis=1)

Y_all = np.log1p(all_df_values['price_doc'].values.reshape(-1,1))

X_all.shape
X_result = result_df_values.drop(['id','timestamp'],axis=1)

id_test = result_df_values['id']

X_result.shape
dtrain = xgb.DMatrix(X_train[:], Y_train[:])

dtest = xgb.DMatrix(X_test, Y_test)

dall = xgb.DMatrix(X_all,Y_all)

dresult=xgb.DMatrix(X_result)
xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}

# Uncomment to tune XGB `num_boost_rounds`

#model = xgb.cv(xgb_params, dtrain, num_boost_round=200,

                  #early_stopping_rounds=30, verbose_eval=10)



model = xgb.train(xgb_params, dtrain, num_boost_round=1000,

                  verbose_eval=20, early_stopping_rounds=20, evals=[(dtrain,'train'),(dtest,'test')])
cv_result = xgb.cv(xgb_params, dall, num_boost_round=1000, early_stopping_rounds=20,

    verbose_eval=10, show_stdv=False)

cv_result[['train-rmse-mean', 'test-rmse-mean']].plot()

num_boost_rounds = len(cv_result)



num_boost_round = 489
cv_result[50:][['train-rmse-mean', 'test-rmse-mean']].plot()



num_round=model.best_iteration

print(num_round)
fig, ax = plt.subplots(1, 1, figsize=(8, 16))

xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}

# Uncomment to tune XGB `num_boost_rounds`

#model = xgb.cv(xgb_params, dtrain, num_boost_round=200,

                  #early_stopping_rounds=30, verbose_eval=10)

model = xgb.train(xgb_params, dall, num_boost_round=210,verbose_eval=20,evals=[(dall,'all')])
logy_pred=model.predict(dresult)

y_pred = np.exp(logy_pred)-1

output=pd.DataFrame(data={'price_doc':y_pred},index=id_test)
plt.hist(y_pred,bins=100)

plt.show()
output.head()
output.to_csv('output.csv',header=True)