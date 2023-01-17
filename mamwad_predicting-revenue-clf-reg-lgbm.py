import pandas as pd
import numpy as np
import logging
import json
from pandas.io.json import json_normalize
import datetime

import tensorflow as tf
from keras import Sequential, layers, optimizers, backend as K
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import lightgbm as lgb

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler, Normalizer, normalize
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, precision_recall_curve

from matplotlib import pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

def get_logger(fname='google_store.log', logger_name=__name__):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s >> %(message)s')

    file_handler = logging.FileHandler(fname)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info('Initializing \'{}\' logger...'.format(logger_name))
    
    return logger

def cdf(data):
    """Compute CDF for a one-dimensional array of measurements."""
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y

def plot_hist_cdf(data, x_label='', title = '', figsize=(10,5), bins=50,xlim=None):
    fig, ax1 = plt.subplots(figsize=figsize);
    ax1.hist(data, bins=bins);
    ax1.set_xlabel(x_label);
    ax1.set_ylabel('Count');
    ax1.set_xlim(xlim);

    ax2=ax1.twinx();
    cdf_x, cdf_y = cdf(data);
    ax2.plot(cdf_x, cdf_y, c='b');
    ax2.set_ylabel('Cumulative');
    ax2.set_ylim(0,);
    ax1ylims = ax1.get_ybound()
    ax2ylims = ax2.get_ybound()
    minresax1=3
    minresax2=.2
    ax1factor = minresax1 * 6
    ax2factor = minresax2 * 6
    ax1.set_yticks(np.linspace(ax1ylims[0],
                               ax1ylims[1]+(ax1factor -
                               (ax1ylims[1]-ax1ylims[0]) % ax1factor) %
                               ax1factor,
                               7))
    ax2.set_yticks(np.linspace(ax2ylims[0],
                               ax2ylims[1]+(ax2factor -
                               (ax2ylims[1]-ax2ylims[0]) % ax2factor) %
                               ax2factor,
                               7))
    plt.title(title);
    plt.show()

def col_summary(dt, col):
    out = dict([('name',[col]),('label',['']),('type', [dt[col].dtype]),('perc_nulls',[dt[col].isnull().sum()/len(dt[col])*100]),
     ('num_uniques', [dt[col].nunique(dropna=False)]),('examples', [str(dt[col].unique()[0:5])])])
    return out

def tab_summary(dt):
    summary = pd.DataFrame()
    for col in dt.columns:
        summary = pd.concat([summary, pd.DataFrame(col_summary(dt, col))])
    summary.reset_index(inplace=True, drop=True)
    return summary

def get_most_common(data_table, column, count):
    df = data_table.groupby(column)['HAS.transactionRevenue'].agg([('count','count'),('num_transactions','sum')], axis=1).sort_values('count',ascending=False).reset_index()
    return df[column][:count].str.lower().values

# https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields
def load_df(csv_path='input/train.csv', JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource'], nrows=None, name=''):
    log.info('Loading {}...'.format(csv_path))
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'},nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    
    df.name = name
    log.info('{} size = {}'.format(df.name, len(df)))
    return df

def table2features(table_in, cols_to_exclude):
    out = pd.DataFrame()
    features=[]
    for col in table_in.columns:
        if col not in cols_to_exclude:
            out[col] = table_in[col]
            features.append(col)
            if table_in[col].dtypes == bool:
                out[col] = table_in[col].astype(int)
    out = out.fillna(0)
    return features, out.values

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def precision_recall_report(y_target, y_pred):
    precision, recall, threshold = precision_recall_curve(y_true = y_target, probas_pred =y_pred)
    dim = min(len(precision),len(recall),len(threshold))
    plt.plot(threshold[:dim], precision[:dim])
    plt.plot(threshold[:dim], recall[:dim])
    plt.xlim(0,1)
    plt.legend(['Precision','Recall'])

try:
    if log:
        log.info('Logger is already running')
except:
    log = get_logger()
    log.disabled = False
# Loading train/test
nrows=None
train = load_df('../input/train.csv', nrows=nrows, name='train_dataset')
test = load_df('../input/test.csv', nrows=nrows, name='test_dataset')
log.info('Data loaded.')
log.info('Dropping columns unique to train data...')
# dropping columns that do not exist in test table
excl = 'totals.transactionRevenue'
for col in train.columns:
    if col != excl and col not in test.columns:
        train.drop(col, axis=1, inplace=True)
        log.info('\tDropped {}.'.format(col))
        
log.info('Dropping columns without variability...')
# dropping columns with only 1 distinct obervations
for col in train.columns:
    if train[col].nunique(dropna=False)==1:
        for tab in [train, test]:
            tab.drop(col, axis=1, inplace=True)
        log.info('\tDropped {}.'.format(col))
    
log.info('Adding date and time columns...')
#adding date/time
for tab in [train, test]:
    tab['date'] = pd.to_datetime(tab['visitStartTime'], unit='s')
    tab['hour'] = tab['date'].dt.hour
    tab['dayofweek'] = tab['date'].dt.weekday
    tab['dayofmonth'] = tab['date'].dt.day
    tab['month'] = tab['date'].dt.month
    tab['trafficSource.adwordsClickInfo.gclId']=tab['trafficSource.adwordsClickInfo.gclId'].isna()
    
log.info('Converting totals to numeric columns...')
# numeric columns
to_numeric =['totals.bounces', 'totals.hits', 'totals.newVisits', 'totals.pageviews', 
             'trafficSource.adwordsClickInfo.page']
for tab in [train, test]:
    for col in to_numeric:
        tab[col] = tab[col].fillna(0).astype(np.int64)

log.info('Computing log of transaction revenues...')
# Computing log of Transaction Revenue
train['log.totals.transactionRevenue'] =np.log1p(train['totals.transactionRevenue'].fillna(0).astype(np.int64))
train['HAS.transactionRevenue']=(train['log.totals.transactionRevenue']>0).astype(np.int32)

log.info('Adding time to the next/previous session...')
# Adding time to next/previus sessions
# adopted from https://www.kaggle.com/ashishpatel26/future-is-here
for i in range(1,3):
    for tab in [test, train]:
        tab.sort_values(['fullVisitorId', 'date'], ascending=True, inplace=True)
        tab['time_to_prev_session_{}'.format(i)] = \
            ((tab['date'] - tab[['fullVisitorId', 'date']].groupby('fullVisitorId')['date'].shift(i))*\
            (tab['fullVisitorId']==tab['fullVisitorId'].shift(i))).fillna(0).astype(np.int64)/1e9/3600
        tab['time_to_next_session_{}'.format(i)] = \
            -((tab['date'] - tab[['fullVisitorId', 'date']].groupby('fullVisitorId')['date'].shift(-i))*\
            (tab['fullVisitorId']==tab['fullVisitorId'].shift(-i))).fillna(0).astype(np.int64)/1e9/3600
for tab in [train, test]:
    tab['nb_pageviews'] = tab['date'].map(
        tab[['date', 'totals.pageviews']].groupby('date')['totals.pageviews'].sum())
    tab['ratio_pageviews'] = tab['totals.pageviews'] / tab['nb_pageviews']
summary = tab_summary(train)
summary.sort_values('num_uniques')
_ = train[train['HAS.transactionRevenue']==1]['HAS.transactionRevenue'].values.sum()
__ = len(train)-_
labels = 'No Revenue', 'With Revenue'
sizes = [__, _]
colors = ['yellowgreen','gold']
explode = (0.1, 0)  # explode 1st slice
plt.figure(figsize=(5,5))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=30)
plt.title('Fraction of sessions with revenue')
plt.show()
_ = train[train['HAS.transactionRevenue']==1]['log.totals.transactionRevenue'].values
plot_hist_cdf(_,bins=50, title='Distribution of transaction revenue',x_label='log.totals.transactionRevenue')
for col in train.columns:
    if train[col].nunique(dropna=False)<700 and col!='HAS.transactionRevenue':
        plt.figure(figsize=(15,5))
        df_HAS = train.groupby(col)['HAS.transactionRevenue'].agg([('has_transactions','sum')], axis=1)
        df_ALL = train.groupby(col)['HAS.transactionRevenue'].agg([('all_visits','count')], axis=1)
        df = pd.concat([df_HAS, df_ALL], axis=1)
        df['has_transactions'] = df['has_transactions']/df['has_transactions'].sum()*100
        df['all_visits'] = df['all_visits']/df['all_visits'].sum()*100
        df.reset_index(inplace=True)
        df2 = pd.melt(df, id_vars=col, value_vars=['has_transactions', 'all_visits'], var_name='perc_visits')
        sns.barplot(x=col, y='value', hue='perc_visits', data=df2)
        plt.ylabel('% of all visits')
        plt.xticks(rotation=90)
        plt.show()
# adapted from https://www.kaggle.com/prashantkikani/teach-lightgbm-to-sum-predictions-fe
browsers = get_most_common(train, 'device.browser',5)
os = get_most_common(train, 'device.operatingSystem',6)
countries = get_most_common(train, 'geoNetwork.country',100)
cities = get_most_common(train, 'geoNetwork.city',100)
regions = get_most_common(train, 'geoNetwork.region',20)
sources = get_most_common(train, 'trafficSource.source',10)

def map_category(x, categories):
    if x in categories:
        return x.lower()
    else:
        return 'others'

log.info('Feature mapping and defining interaction features...')
for tab in[train, test]:
    tab['device.browser'] = tab['device.browser'].map(lambda x:map_category(str(x).lower(), browsers)).astype('str')
    tab['device.operatingSystem'] = tab['device.operatingSystem'].map(lambda x:map_category(str(x).lower(), os)).astype('str')
    tab['geoNetwork.country'] = tab['geoNetwork.country'].map(lambda x:map_category(str(x).lower(), countries)).astype('str')
    tab['geoNetwork.city'] = tab['geoNetwork.city'].map(lambda x:map_category(str(x).lower(), cities)).astype('str')
    tab['geoNetwork.region'] = tab['geoNetwork.region'].map(lambda x:map_category(str(x).lower(), regions)).astype('str')
    tab['trafficSource.source'] = tab['trafficSource.source'].map(lambda x:map_category(str(x).lower(), sources)).astype('str')
# Aggregating features at visitor level
cols_to_agg=['visitNumber','totals.bounces', 'totals.hits',
            'totals.newVisits','totals.pageviews', 'hour', 'nb_pageviews', 
             'ratio_pageviews', 'time_to_prev_session_1', 'time_to_next_session_1']

aggs = {'sum_':'sum', 'mean_':'mean'}

log.info('Adding aggregated features to train set...')
_ = train.groupby('fullVisitorId')[cols_to_agg].agg(aggs)
_.columns = _.columns.map(''.join)
_.fillna(0, inplace=True)
train = train.join(_,on='fullVisitorId')
log.info('train data shape: {}'.format(train.shape))

log.info('Adding aggregated features to test set...')
_ = test.groupby('fullVisitorId')[cols_to_agg].agg(aggs)
_.columns = _.columns.map(''.join)
_.fillna(0, inplace=True)
test = test.join(_,on='fullVisitorId')
log.info('test data shape: {}'.format(test.shape))
# Features to exclude
exclude_features = ['date', 'fullVisitorId', 'sessionId', 'visitId', 'visitStartTime', 
                   'totals.transactionRevenue', 'log.totals.transactionRevenue', 
                   'HAS.transactionRevenue',
                    'trafficSource.adwordsClickInfo.adNetworkType', 'trafficSource.adwordsClickInfo.isVideoAd', 
                    'trafficSource.adwordsClickInfo.slot', 'trafficSource.campaign', 'trafficSource.isTrueDirect', 
                    'geoNetwork.continent', 'geoNetwork.networkDomain', 'trafficSource.referralPath']

# One-hot-coding
dummies_max = 10
log.info('One-hot-coding of features with less than {} categories:'.format(dummies_max))
to_dummies =[]
for col in train.columns:
    nuniq = train[col].nunique(dropna=False)
    if nuniq>2 and nuniq<dummies_max and col not in exclude_features:
        to_dummies.append(col)
        log.info('\tone-hot-coding: {}'.format(col))
        
log.info('\tInitial size of train data set: {}'.format(train.shape))
_ = pd.concat([train, test], sort=False)
_ = pd.get_dummies(_, dummy_na=False, columns=to_dummies, drop_first=True)
train = _[:len(train)]
test = _[len(train):]
test = test.drop(columns=['totals.transactionRevenue', 'log.totals.transactionRevenue', 'HAS.transactionRevenue'],axis=1)
log.info('\tFinal size of train data set: {}'.format(train.shape))

# Factorizing
cat_features = [col for col in train.columns 
                if (col not in exclude_features) & (train[col].dtypes == 'object')]
log.info('Factorizing categorical features...')
for col in cat_features:
    train[col], indexer = pd.factorize(train[col])
    test[col] = indexer.get_indexer(test[col])

# Get test/train arrays
log.info('Getting test/train arrays...')
y_clf = train['HAS.transactionRevenue'].values
features_train, X_train = table2features(train, exclude_features)
features_test, X_test = table2features(test, exclude_features)
# LGBM model
def get_lgbm_clf(num_leaves=100, lr=0.02):
    model_clf = lgb.LGBMClassifier(num_leaves=num_leaves, learning_rate=lr, n_estimators=1000,
                                    subsample=.9, colsample_bytree=.9, random_state=42)
    return model_clf

# StratifiedKFold Training
n_splits=5
models_clf1 = []
models_clf2 = []
fold_ = 1
feature_importance = pd.DataFrame()
class_weights = class_weight.compute_class_weight('balanced',[0,1],y_clf.flatten())
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=32)

for train_index, val_index in skf.split(X_train, y_clf):
    log.info('Trainign with fold = {}'.format(fold_))
    trn_x, trn_y = X_train[train_index], y_clf[train_index]
    val_x, val_y = X_train[val_index], y_clf[val_index]
    model_clf1 = get_lgbm_clf()
    # LGBM classification
    model_clf1.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=100, verbose=100)
    # LGBM: recording feature importance
    _ = pd.DataFrame()
    _['feature'] = features_train
    _['gain'] = model_clf1.booster_.feature_importance(importance_type='gain')
    _['fold'] = fold_
    fold_ += 1
    feature_importance = pd.concat([feature_importance, _], axis=0, sort=False)
    
    #recoding models
    models_clf1.append(model_clf1)
log.info('Finished training.')

# Making predictions
log.info('Predictions from LGBM model...')
y_preds1_train = np.zeros((n_splits, len(X_train)))
y_preds1_test = np.zeros((n_splits, len(X_test)))
for i in range(n_splits):
    y_preds1_train[i] = models_clf1[i].predict_proba(X_train, num_iteration=models_clf1[i].best_iteration_)[:,1]
    y_preds1_test[i] = models_clf1[i].predict_proba(X_test, num_iteration=models_clf1[i].best_iteration_)[:,1]
train['predicted_prob_clf1'] = y_preds1_train.mean(axis=0)
test['predicted_prob_clf1'] = y_preds1_test.mean(axis=0)

# Feature importance plot
feature_importance['gain_log'] = np.log1p(feature_importance['gain'])
mean_gain = feature_importance[['gain','feature']].groupby('feature').mean()
feature_importance['mean_gain'] = feature_importance['feature'].map(mean_gain['gain'])

plt.figure(figsize=(5,30))
sns.barplot(x='gain_log', y='feature', data=feature_importance.sort_values('gain_log', ascending=False))
plt.show()
# Creating dataset arrays
log.info('Creating datasets for LGBM regression...')
y_reg = train['log.totals.transactionRevenue'].values
y_clf = train['HAS.transactionRevenue'].values
features_train, X_train = table2features(train, exclude_features)
features_test, X_test = table2features(test, exclude_features)

def get_lgbm_reg(num_leaves=100, lr=0.02):
    model_reg =  lgb.LGBMRegressor(num_leaves=num_leaves, learning_rate=lr,
        n_estimators=1000, subsample=.9, colsample_bytree=.9, random_state=42)
    return model_reg

# Training
log.info('Training LGBM regressor...')
n_splits=5
models_reg = []
fold_ = 1
mean_rmse = 0
feature_importance = pd.DataFrame()
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
for train_index, val_index in skf.split(X_train, y_clf):
    log.info('Training fold {}'.format(fold_))
    trn_x, trn_y = X_train[train_index], y_reg[train_index]
    val_x, val_y = X_train[val_index], y_reg[val_index]
    
    # regression model
    model_reg = get_lgbm_reg(lr=0.03)
    model_reg.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=100,
        verbose=100, eval_metric='rmse')
    
    #recording feature importance
    _ = pd.DataFrame()
    _['feature'] = features_train
    _['gain'] = model_reg.booster_.feature_importance(importance_type='gain')
    _['fold'] = fold_
    fold_ += 1
    feature_importance = pd.concat([feature_importance, _], axis=0, sort=False)
    
    #recoding models
    models_reg.append(model_reg)
    mean_rmse += model_reg.best_score_['valid_0']['rmse']/n_splits
log.info('Mean RMSE: {:.3f}'.format(mean_rmse))
# creating prediction arrays
log.info('Predicting log of revenues...')
cutoff = 0
y_preds_train = np.zeros((n_splits, len(X_train)))
y_preds_test = np.zeros((n_splits, len(X_test)))
for i in range(n_splits):
    y_preds_train[i] = models_reg[i].predict(X_train, num_iteration=models_reg[i].best_iteration_)
    y_preds_test[i] = models_reg[i].predict(X_test, num_iteration=models_reg[i].best_iteration_)
y_pred_train = y_preds_train.mean(axis=0)
y_pred_test = y_preds_test.mean(axis=0)
y_pred_train[y_pred_train<cutoff]=0
y_pred_test[y_pred_test<cutoff]=0
train['predicted_log.revenue1'] = y_pred_train
test['predicted_log.revenue1'] = y_pred_test
# Prediction vs target plots
cutoff=10
_, __ = cdf(trn_y[trn_y>0])
plt.plot(_,__)

_,__ = cdf(y_pred_train[y_pred_train>cutoff])
plt.plot(_,__)

for i in range(n_splits):
    _, __ = cdf(y_preds_train[i][y_preds_train[i]>cutoff])
    plt.plot(_, __)

plt.legend(['target','mean prediction', 'fold 1', 'fold 2','fold 3','fold 4','fold 5'])
plt.show()
# Feature importance plot
feature_importance['gain_log'] = np.log1p(feature_importance['gain'])
mean_gain = feature_importance[['gain','feature']].groupby('feature').mean()
feature_importance['mean_gain'] = feature_importance['feature'].map(mean_gain['gain'])

plt.figure(figsize=(5,30))
sns.barplot(x='gain_log', y='feature', data=feature_importance.sort_values('gain_log', ascending=False))
plt.show()
test['PredictedLogRevenue'] = np.expm1(test['predicted_log.revenue1'])
out = test[['fullVisitorId','PredictedLogRevenue']].groupby('fullVisitorId', axis=0).sum()
out['PredictedLogRevenue'] = np.log1p(out['PredictedLogRevenue'])
out.to_csv('submission.csv')
out.shape

