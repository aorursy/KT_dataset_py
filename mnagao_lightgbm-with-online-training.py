import gc
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
from kaggle.competitions import twosigmanews
pd.set_option('max_columns', 50)
env = twosigmanews.make_env()
market_train_df, news_train = env.get_training_data()
for column in market_train_df.select_dtypes(include='float64').columns:
    market_train_df[column] = market_train_df[column].astype('float16')
start = datetime(2010, 1, 1, 0, 0, 0).date()
market_train = market_train_df.loc[market_train_df['time'].dt.date >= start].reset_index(drop=True)
news_train = news_train.loc[news_train['time'].dt.date >= start].reset_index(drop=True)
def preprocess_news(news_train):
    drop_list = [
        'audiences', 'subjects', 'assetName',
        'headline', 'firstCreated', 'sourceTimestamp',
    ]
    news_train.drop(drop_list, axis=1, inplace=True)
    
    # Factorize categorical columns
    for col in ['headlineTag', 'provider', 'sourceId']:
        news_train[col], uniques = pd.factorize(news_train[col])
        del uniques
    
    # Remove {} and '' from assetCodes column
    news_train['assetCodes'] = news_train['assetCodes'].apply(lambda x: x[1:-1].replace("'", ""))
    return news_train
news_train = preprocess_news(news_train)
def unstack_asset_codes(news_train):
    codes = []
    indexes = []
    for i, values in news_train['assetCodes'].iteritems():
        explode = values.split(", ")
        codes.extend(explode)
        repeat_index = [int(i)]*len(explode)
        indexes.extend(repeat_index)
    index_df = pd.DataFrame({'news_index': indexes, 'assetCode': codes})
    del codes, indexes
    gc.collect()
    return index_df
index_df = unstack_asset_codes(news_train)
index_df.head()
def merge_news_on_index(news_train, index_df):
    news_train['news_index'] = news_train.index.copy()

    # Merge news on unstacked assets
    news_unstack = index_df.merge(news_train, how='left', on='news_index')
    news_unstack.drop(['news_index', 'assetCodes'], axis=1, inplace=True)
    return news_unstack
news_unstack = merge_news_on_index(news_train, index_df)
del news_train, index_df
gc.collect()
news_unstack.head(3)
def group_news(news_frame):
    news_frame['date'] = news_frame.time.dt.date  # Add date column
    
    aggregations = ['mean']
    gp = news_frame.groupby(['assetCode', 'date']).agg(aggregations)
    gp.columns = pd.Index(["{}_{}".format(e[0], e[1]) for e in gp.columns.tolist()])
    gp.reset_index(inplace=True)
    gp.drop(['sourceId_mean'],axis=1, inplace=True)
    # Set datatype to float32
    float_cols = {c: 'float32' for c in gp.columns if c not in ['assetCode', 'date']}
    return gp.astype(float_cols)
news_agg = group_news(news_unstack)
del news_unstack; gc.collect()
news_agg.head(3)
def process_date(df):
    df['date'] = df['time'].dt.date
    df['month'] = df['time'].dt.month
    df['dayofweek'] = df['time'].dt.dayofweek
    
    return df
def process_ma(df,columns=['open','close','volume'], windows=[10, 20, 60]):
    ma_columns = []
    ma_dev_columns = []
    drop_list = []
    for col in columns:
        for window in windows:
            ma_column = 'ma_{0}_{1}'.format(col,window)
            ma_dev_column = 'ma_dev_{0}_{1}'.format(col,window)
            ma_lag_column = 'ma_lag_{0}_{1}'.format(col,window)
            ma_lag_rate_column = 'ma_lag_rate_{0}_{1}'.format(col,window)
            std_column = 'std_{0}_{1}'.format(col,window)
            ma_columns.append(ma_column)
            ma_dev_columns.append(ma_dev_column)
            # calc moving average
            df[ma_column] = df.groupby('assetCode')[col].apply(lambda x: x.rolling(window).mean())
            # calc rate of deviation from moving average
            df[ma_dev_column] = df[col] / df[ma_column] - 1
            # calc moving std
            df[std_column] = df.groupby('assetCode')[col].apply(lambda x: x.rolling(window).std())
            
            df[ma_lag_column] = df.groupby('assetCode')[ma_column].shift()
            df[ma_lag_rate_column] = (df[ma_column] - df[ma_lag_column]) / df[ma_lag_column]
            
            drop_list.append(ma_column)
            drop_list.append(ma_lag_column)
            
    df.drop(drop_list,axis=1, inplace=True)
    return df
market_train = process_date(market_train)
market_train = process_ma(market_train)
df = market_train.merge(news_agg, how='left', on=['assetCode', 'date'])
del market_train, news_agg
gc.collect()
df.head(3)
for column in df.select_dtypes(include='float64').columns:
    df[column] = df[column].astype('float16')

for column in df.select_dtypes(include='int64').columns:
    df[column] = df[column].astype('int16')
df.dtypes
def custom_metric(date, pred_proba, num_target, universe):
    y = pred_proba*2 - 1
    r = num_target.clip(-1,1) # get rid of outliers
    x = y * r * universe
    result = pd.DataFrame({'day' : date, 'x' : x})
    x_t = result.groupby('day').sum().values
    return np.mean(x_t) / np.std(x_t)
date = df.date
num_target = df.returnsOpenNextMktres10.astype('float32')
bin_target = (df.returnsOpenNextMktres10 >= 0).astype('int8')
universe = df.universe.astype('int8')
# Drop columns that are not features
df.drop(['returnsOpenNextMktres10', 'date', 'universe', 'assetCode', 'assetName', 'time'], 
        axis=1, inplace=True)
df = df.astype('float32')  # Set all remaining columns to float32 datatype
gc.collect()
train_index, test_index = train_test_split(df.index.values, test_size=0.1)
best_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'n_estimators': 1000,
    'min_child_samples': 20,
    'colsample_bytree': 1,
    'subsample': 1.0,
    'reg_alpha': 0.1,
    'reg_lambda': 0.8,
}
lgb_train = lgb.Dataset(df.iloc[train_index], bin_target.iloc[train_index])
lgb_eval = lgb.Dataset(df.iloc[test_index], bin_target.iloc[test_index], reference=lgb_train)
clf = lgb.train(
            best_params,
            lgb_train,
            valid_sets=lgb_eval,
            early_stopping_rounds=5
        )
pred_y = clf.predict(df.iloc[test_index])
pred_y[pred_y >= 0.5] = 1
pred_y[pred_y < 0.5] = 0
print((pred_y == bin_target.iloc[test_index]).sum() / len(pred_y))
test_df_columns = ['time', 'assetCode', 'assetName', 'volume', 'close', 'open',
                   'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
                   'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
                   'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
                   'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
base_df = market_train_df[market_train_df['time'] >= '2016-10-01']
base_df = base_df[test_df_columns]
base_df['id'] = -1
base_df.shape
def calc_target(df):
    df['open_next10'] = df.groupby('assetCode')['open'].shift(-10)
    df['return_next10'] = df['open_next10'] / df['open'] - 1.0
    df['target'] = (df.return_next10 >= 0).astype('int8')
    return df
target_test_df = calc_target(market_train_df)
target_test_df.head()
def online_train(model, df, train_day_id, feats):
    best_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'n_estimators': 1000,
        'min_child_samples': 20,
        'colsample_bytree': 1,
        'subsample': 1.0,
        'reg_alpha': 0.1,
        'reg_lambda': 0.8,
    }
    
    train_df = calc_target(df)
    train_df = train_df[train_df['id'] == train_day_id]
    train_df.reset_index(drop=True)
    target_df = train_df[['target']]
    train_df = train_df[feats]
    
    model.refit(train_df, target_df['target'])
    return model
    """
    train_index, test_index = train_test_split(train_df.index.values, test_size=0.1)
    lgb_train = lgb.Dataset(train_df.iloc[train_index], target_df.iloc[train_index])
    lgb_eval = lgb.Dataset(train_df.iloc[test_index], target_df.iloc[test_index], reference=lgb_train)
    return lgb.train(
            best_params,
            lgb_train,
            init_model=model,
            learning_rates=lambda iter: 0.05 * (0.99 ** iter),
            valid_sets=lgb_eval,
            early_stopping_rounds=5
    )
    """
    
def write_submission(model, env):
    days = env.get_prediction_days()
    day_id = 0
    market_obs_df_append = None
    online_train_df_append = None
    for (market_obs_df, news_obs_df, predictions_template_df) in days:
        news_obs_df = preprocess_news(news_obs_df)
        # Unstack news
        index_df = unstack_asset_codes(news_obs_df)
        news_unstack = merge_news_on_index(news_obs_df, index_df)
        # Group and and get aggregations (mean)
        news_obs_agg = group_news(news_unstack)

        market_obs_df['id'] = day_id
        if market_obs_df_append is None:
            market_obs_df_append = base_df
            
        market_obs_df_append = pd.concat([market_obs_df_append,market_obs_df],
                                         ignore_index=True,
                                         sort=False)
        
        market_obs_process = process_date(market_obs_df_append)
        market_obs_process = process_ma(market_obs_process)
        market_obs_df = market_obs_process[market_obs_process['id']==day_id]
        # Join market and news frames
        obs_df = market_obs_df.merge(news_obs_agg, how='left', on=['assetCode', 'date'])
        del market_obs_df, news_obs_agg, news_obs_df, news_unstack, index_df
        gc.collect()
        obs_df = obs_df[obs_df.assetCode.isin(predictions_template_df.assetCode)]
        
        # Drop cols that are not features
        feats = [c for c in obs_df.columns if c not in ['date', 'assetCode', 'assetName', 'time', 'id']]
        
        # pile obs_df for online training
        if online_train_df_append is None:
            online_train_df_append = obs_df
        else:
            online_train_df_append = pd.concat([online_train_df_append, obs_df],
                                                ignore_index=True,
                                                sort=False)
        
        # online training
        if day_id >= 10:
            model = online_train(model, online_train_df_append, day_id - 10, feats)
        
        #preds = model.predict_proba(obs_df[feats])[:, 1] * 2 - 1
        preds = model.predict(obs_df[feats]) * 2 - 1
        sub = pd.DataFrame({'assetCode': obs_df['assetCode'], 'confidence': preds})
        predictions_template_df = predictions_template_df.merge(sub, how='left').drop(
            'confidenceValue', axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
        
        env.predict(predictions_template_df)
        if day_id == 59:
            market_obs_df_append.drop(
                market_obs_df_append.index[market_obs_df_append['id']==-1],
                inplace=True)
        elif day_id >= 60:
            market_obs_df_append.drop(
                market_obs_df_append.index[market_obs_df_append['id']==day_id-60],
                inplace=True)
        day_id += 1
        del obs_df, predictions_template_df, preds, sub
        gc.collect()
    env.write_submission_file()
    print('day_count',day_id)
write_submission(clf, env)
feat_importance = pd.DataFrame()
feat_importance["feature"] = df.columns
feat_importance["gain"] = clf.feature_importance(importance_type='gain')
feat_importance.sort_values(by='gain', ascending=False, inplace=True)
plt.figure(figsize=(8,10))
ax = sns.barplot(y="feature", x="gain", data=feat_importance)