import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import seaborn as sns
import plotly.tools as tls
from itertools import chain
import pandas as pd
import numpy as np
import re
from pandas.tseries.holiday import USFederalHolidayCalendar
import datetime
import warnings
warnings.filterwarnings('ignore')
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
market_train_df.shape, news_train_df.shape
market_train_df.head()
percent = (market_train_df.isnull().sum()/market_train_df.shape[0]).sort_values(ascending=False)
trace = go.Bar(
    x = percent.index,
    y = percent.values)
layout = dict(title = "Missing values in Market data",
              xaxis = dict(title = 'column'),
              yaxis = dict(title = 'Missing values'),
              )
data = [trace]
py.iplot(dict(data=data, layout=layout), filename='basic-line')
market_train_df.assetName.describe()
market_train_df[market_train_df['assetName'] == 'Unknown'].size
assetName_gb = market_train_df[market_train_df['assetName'] == 'Unknown'].groupby('assetCode')
unknownAssets = assetName_gb.size().reset_index('assetCode')
unknownAssets
del unknownAssets
print('Oldest date:', market_train_df['time'].min().strftime('%Y-%m-%d'))
print('Most recent date:', market_train_df['time'].max().strftime('%Y-%m-%d'))
market_train_df.columns.values
volumesByTradingYear = market_train_df.groupby(market_train_df['time'].dt.year)['volume'].sum()
trace1 = go.Bar(
    x = volumesByTradingYear.index,
    y = volumesByTradingYear.values
)

layout = dict(title = "Trading volumes by Year",
              xaxis = dict(title = 'time'),
              yaxis = dict(title = 'Volume'),
              )
data = [trace1]
py.iplot(dict(data=data, layout=layout), filename='basic-line')
volumesByTradingMonth = market_train_df.groupby(market_train_df['time'].dt.month)['volume'].sum()
trace1 = go.Bar(
    x = volumesByTradingMonth.index,
    y = volumesByTradingMonth.values
)
layout = dict(title = "Trading volumes by Month",
              xaxis = dict(title = 'Month'),
              yaxis = dict(title = 'Volume'),
              )
data = [trace1]
py.iplot(dict(data=data, layout=layout), filename='basic-line')
volumesByTradingDay = market_train_df.groupby(market_train_df['time'].dt.date)['volume'].sum()
trace1 = go.Bar(
    x = volumesByTradingDay.index,
    y = volumesByTradingDay.values
)
layout = dict(title = "Trading volumes by Date",
              xaxis = dict(title = 'Date'),
              yaxis = dict(title = 'Volume'),
              )
data = [trace1]
py.iplot(dict(data=data, layout=layout), filename='basic-line')
volumesByCode = market_train_df.groupby('assetCode')['volume'].sum()
highestVolumes = volumesByCode.sort_values(ascending=False)[:10]
trace1 = go.Pie(
    labels = highestVolumes.index,
    values = highestVolumes.values
)
layout = dict(title = "Top 10 of volumes")
data = [trace1]
py.iplot(dict(data=data, layout=layout), filename='basic-line')
var = 'volume'
data = pd.concat([market_train_df['returnsOpenNextMktres10'],market_train_df[var]],axis=1)
data.plot.scatter(x = var, y = 'returnsOpenNextMktres10', ylim = (0.8000))
var = 'close'
data = pd.concat([market_train_df['returnsOpenNextMktres10'],market_train_df[var]],axis=1)
data.plot.scatter(x = var, y = 'returnsOpenNextMktres10',ylim = (0.8000))
var = 'open'
data = pd.concat([market_train_df['returnsOpenNextMktres10'],market_train_df[var]],axis=1)
data.plot.scatter(x = var, y = 'returnsOpenNextMktres10',ylim = (0.8000))
news_train_df.head()
percent = (news_train_df.isnull().sum()/news_train_df.shape[0]).sort_values(ascending=False)
trace = go.Bar(
    x = percent.index,
    y = percent.values)
layout = dict(title = "Missing values in News data",
              xaxis = dict(title = 'column'),
              yaxis = dict(title = 'Missing values'),
              )
data = [trace]
py.iplot(dict(data=data, layout=layout), filename='basic-line')
corrmat = news_train_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
news_train_df.headline.describe()
news_train_df.assetName.describe()
news_train_df.assetCodes.describe()
news_agg_col = {
    'urgency':['min','count'],
    'takeSequence':['max'],
    'bodySize':['mean','sum','max','min'],
    'companyCount':['mean','sum','max','min'],
    'sentenceCount':['mean','sum','max','min'],
    'wordCount':['mean','sum','max','min'],
#     'marketCommentary':['mean','sum','max','min'],
    'relevance':['mean','sum','max','min'],
    'sentimentWordCount':['mean','sum','max','min'], 
    'sentimentNegative':['mean','sum','max','min'], 
    'sentimentNeutral':['mean','sum','max','min'],
    'sentimentPositive':['mean','sum','max','min'],
    'noveltyCount12H':['mean','sum','max','min'],
    'noveltyCount24H':['mean','sum','max','min'],
    'noveltyCount3D':['mean','sum','max','min'],
    'noveltyCount5D':['mean','sum','max','min'],
    'noveltyCount7D':['mean','sum','max','min'],
    'volumeCounts12H':['mean','sum','max','min'],
    'volumeCounts24H':['mean','sum','max','min'],
    'volumeCounts3D':['mean','sum','max','min'],
    'volumeCounts5D':['mean','sum','max','min'],
    'volumeCounts7D':['mean','sum','max','min']
}
def get_news_market_connect(news_df,market_df):
    news_df['assetCodes'] =  news_df['assetCodes'].str.findall("'(.*?)'")
    assertCode_expanded = list(chain(*news_df['assetCodes']))
    asset_index = news_df.index.repeat(news_df['assetCodes'].apply(len))
    assert_pd = pd.DataFrame({'_index':asset_index,'assetCode':assertCode_expanded})
    temp_news_expanded = pd.merge(assert_pd,news_df[['time','assetCodes']], left_on = '_index', right_index = True, suffixes = (['','_old']))
    temp_news_df = temp_news_expanded.copy()[['time','assetCode','_index']]
    temp_market_df = market_df.copy()[['time','assetCode']]
    temp_valid_col = pd.merge(temp_news_df,temp_market_df,on=['time','assetCode'])
    return temp_news_df,temp_market_df,temp_valid_col
#     temp_news_df.set_index(['time','assetCode'],inplace=True)
#     temp_market_df_new = temp_market_df.join(temp_news_df,on=['time','assetCode'])
#     temp_market_df_new = pd.merge(temp_market_df, temp_news_df,on=['time','assetCode'])
temp_news_df, temp_market_df,temp_valid_col = get_news_market_connect(news_train_df, market_train_df)
temp_market_df.head()
temp_news_df.head()
temp_valid_col.shape
del temp_market_df,temp_news_df
def join_market(temp_valid_col,news_train_df,market_train_df):
    new_col = sorted(list(news_agg_col.keys()))
    temp_valid = pd.merge(temp_valid_col, news_train_df[new_col],left_on='_index',right_index = True)
    temp_valid_gy = temp_valid.groupby(['time','assetCode'])
    temp_valid_agg = temp_valid_gy.agg(news_agg_col).apply(np.float32).reset_index()
    temp_valid_agg.columns = ['_'.join(col).strip() for col in temp_valid_agg.columns.values ]
    temp_valid_agg.rename(columns={'time_':'time','assetCode_':'assetCode'},inplace=True)
    if market_train_df.shape[0]> 3000000:
        train_df = pd.merge(market_train_df.tail(3000000),temp_valid_agg, on=['time','assetCode'],how = 'left')
    else:
        train_df = pd.merge(market_train_df,temp_valid_agg, on=['time','assetCode'],how = 'left')
    return temp_valid,temp_valid_agg,train_df
temp_valid,temp_valid_agg,train_df = join_market(temp_valid_col,news_train_df,market_train_df)
del market_train_df, news_train_df
temp_valid.columns
del temp_valid
temp_valid_agg.head()
import gc
def get_extra(T):
    T['diff'] = T['close']-T['open']
    T['close_to_open'] = np.abs(T['close'] / T['open'])
    
def generate_time(T):
    #check if the day is holiday
    check = USFederalHolidayCalendar()
    T['time'] = T['time'].dt.strftime('%Y-%m-%d')
    T['temp_time'] = T['time'].str.replace("UTC","")
    T['temp_time'] = pd.to_datetime(T['temp_time'],format = '%Y-%m-%d %H')
    T['day_of_year'] = T.temp_time.dt.dayofyear
    T['week_of_year'] = T.temp_time.dt.weekofyear
    T['weekday'] = T.temp_time.dt.weekday
    T['on_weekday'] = T['weekday'].apply(lambda x:1 if x > 4 else 0)
    T['year'] = T.temp_time.dt.year
    T['month'] = T.temp_time.dt.month
    T['day'] = T.temp_time.dt.day
    T['start_of year'] = T['month'].apply(lambda x:1 if x < 2  else 0)
    T['end_of year'] = T['month'].apply(lambda x:1 if x > 11 else 0)
    T['start_of_month'] = T['day'].apply(lambda x:1 if x < 6 else 0)
    T['end_of_month'] = T['day'].apply(lambda x:1 if x > 25 else 0)
    del T['temp_time']
    gc.collect()
    holidays = check.holidays(start = '2007-02-01', end = '2016-12-30').to_pydatetime()
    T['on_holiday'] = T['time'].apply(lambda x: 1 if x in holidays else 0)
    del T['time']
# def remove_col(T):
#     for f in T.columns:
#         if T[f].dtype == 'object'  or f == 'assetName':
#             del T[f]
def label_encode(series, min_count):
    c = series.value_counts()
    re_encoed = {c:i for i,c in enumerate(c.index[c >= min_count])}
    return re_encoed
def get_encoded(T):
    Code_encode = label_encode(T['assetCode'],min_count = 10)
    Name_encode = label_encode(T['assetName'],min_count = 5)
    T['assetCode'] = T['assetCode'].map(Code_encode).fillna(-1).astype(int)
    T['assetName'] = T['assetName'].map(Name_encode).fillna(-1).astype(int)
def get_x(T):
    get_encoded(T)
    get_extra(T)
    generate_time(T)
get_x(train_df)
train_df.shape
train_df.tail()
def beforeModel(T):
    y = train_df['returnsOpenNextMktres10'].clip(-1,1)
    del T['returnsOpenNextMktres10'],T['universe']
    return T, y
T, y = beforeModel(train_df)
T.shape,y.shape
def get_parameter(T,_y):
    X_train,X_test, y_train, y_test = split_df(T,_y)
    # other scikit-learn modules
    estimator = lgb.LGBMRegressor(num_leaves=31)
    param_grid = {
        'learning_rate': [0.001,0.01,0.1],
        'n_estimators': [3000,4000,6000]
    }
    gbm = GridSearchCV(estimator, param_grid, cv=3)
    gbm.fit(X_train, y_train)
    print('Best parameters found by grid search are:', gbm.best_params_)
def split_df(T,_y):
    train_x,test_x, train_y, test_y= train_test_split(T,_y,test_size=0.2,random_state=99)
    return train_x,test_x, train_y, test_y
def gbm_rmse_training(T,_y):
    X_train,X_test, y_train, y_test = split_df(T,_y)
    print('Starting training...')
    gbm = lgb.LGBMRegressor(num_leaves=120,
                            learning_rate=0.1,
                            n_estimators=5000)
    gbm.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='l1',
            early_stopping_rounds=200)
    print('Starting predicting...')
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
    # eval
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5) 
    return gbm
def rmsle(y_true, y_pred):
    return 'RMSLE', np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2))), False
def gbm_rmsle_training(T,_y):
    X_train,X_test, y_train, y_test = split_df(T,_y)
    print('Starting training ...')
    # train
    gbm = lgb.LGBMRegressor(num_leaves=120,
                        learning_rate=0.1,
                        n_estimators=10)
    gbm.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=rmsle,
            early_stopping_rounds=20)

    print('Starting predicting...')
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
    # eval
    print('The rmsle of prediction is:', rmsle(y_test, y_pred)[1])
def gbm_plot(T,_y):
    X_train,X_test, y_train, y_test = split_df(T,_y)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
    train_cols = T.columns.tolist()
    params = {
    'num_leaves': 120,
    'metric': ('l1', 'l2'),
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,#每 k 次迭代执行bagging
    'learning_rate': 0.1,
    'objective': 'regression',
    'boosting':'gbdt',
    'seed' : 1220
    }
    categorical_feature = []
    evals_result = {}  # to record eval results for plotting
    print('Starting training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=8000,
                    valid_sets=[lgb_train, lgb_test],
                    feature_name=train_cols,
                    categorical_feature=categorical_feature,
                    evals_result=evals_result,
                    verbose_eval=1000,
                    early_stopping_rounds = 200)

    print('Plotting metrics recorded during training...')
    ax = lgb.plot_metric(evals_result, metric='l1')
    plt.show()
    ax = lgb.plot_metric(evals_result, metric='l2')
    plt.show()

    print('Plotting feature importances...')
    fig,ax = plt.subplots(1,2,figsize = (14,14))
    lgb.plot_importance(gbm,ax[0])
    lgb.plot_importance(gbm,ax[1],importance_type='gain')
    fig.tight_layout()

    print('Plotting 5th tree...')  # one tree use categorical feature to split
    ax = lgb.plot_tree(gbm, tree_index=5, figsize=(25, 10), show_info=['split_gain'])
    plt.show()
    return gbm
# get_parameter(T,y)
# gbm = gbm_rmse_training(T,y)
# gbm_rmsle_training(T,y)
gbm = gbm_plot(T,y)
def make_predictions(market_obs_df,news_obs_df):
    temp_news_df,_,_ = get_news_market_connect(news_obs_df,market_obs_df)
    _,_,train_obs_df = join_market(temp_news_df,news_obs_df,market_obs_df)
    get_x(train_obs_df)
    prediction_values = np.clip(gbm.predict(train_obs_df),-1,1)
    return prediction_values
for (market_obs_df, news_obs_df, predictions_template_df) in env.get_prediction_days():
    predictions_template_df['confidenceValue'] = make_predictions(market_obs_df,news_obs_df)
    env.predict(predictions_template_df)
env.write_submission_file()
print('finished')