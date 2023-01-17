%matplotlib inline

import os
import sys
import multiprocessing as mp
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from functools import partial
from scipy.stats import skew, kurtosis, iqr
from sklearn.externals import joblib
from tqdm import tqdm_notebook as tqdm

warnings.filterwarnings('ignore')
def add_features(feature_name, aggs, features, feature_names, groupby):
    feature_names.extend(['{}_{}'.format(feature_name, agg) for agg in aggs])
    for agg in aggs:
        if agg == 'kurt':
            agg_func = kurtosis
        elif agg == 'iqr':
            agg_func = iqr
        else:
            agg_func = agg
        g = groupby[feature_name].agg(agg_func).reset_index().rename(index=str, columns={feature_name: '{}_{}'.format(feature_name, agg)})
        features = features.merge(g, on='SK_ID_CURR', how='left')
    return features, feature_names


def add_features_in_group(features, gr_, feature_name, aggs, prefix):
    for agg in aggs:
        if agg == 'sum':
            features['{}{}_sum'.format(prefix, feature_name)] = gr_[feature_name].sum()
        elif agg == 'mean':
            features['{}{}_mean'.format(prefix, feature_name)] = gr_[feature_name].mean()
        elif agg == 'max':
            features['{}{}_max'.format(prefix, feature_name)] = gr_[feature_name].max()
        elif agg == 'min':
            features['{}{}_min'.format(prefix, feature_name)] = gr_[feature_name].min()
        elif agg == 'std':
            features['{}{}_std'.format(prefix, feature_name)] = gr_[feature_name].std()
        elif agg == 'count':
            features['{}{}_count'.format(prefix, feature_name)] = gr_[feature_name].count()
        elif agg == 'skew':
            features['{}{}_skew'.format(prefix, feature_name)] = skew(gr_[feature_name])
        elif agg == 'kurt':
            features['{}{}_kurt'.format(prefix, feature_name)] = kurtosis(gr_[feature_name])
        elif agg == 'iqr':
            features['{}{}_iqr'.format(prefix, feature_name)] = iqr(gr_[feature_name])
        elif agg == 'median':
            features['{}{}_median'.format(prefix, feature_name)] = gr_[feature_name].median()
        return features

def parallel_apply(groups, func, index_name='Index', num_workers=1, chunk_size=100000):
    n_chunks = np.ceil(1.0 * groups.ngroups / chunk_size)
    indeces, features  = [],[]
    for index_chunk, groups_chunk in tqdm(chunk_groups(groups, chunk_size), total=n_chunks):
        with mp.pool.Pool(num_workers) as executor:
            features_chunk = executor.map(func, groups_chunk)
        features.extend(features_chunk)
        indeces.extend(index_chunk)
    features = pd.DataFrame(features)
    features.index = indeces
    features.index.name = index_name
    return features

def chunk_groups(groupby_object, chunk_size):
    n_groups = groupby_object.ngroups
    group_chunk, index_chunk = [],[]
    for i, (index, df) in enumerate(groupby_object):
        group_chunk.append(df)
        index_chunk.append(index)

        if (i + 1) % chunk_size == 0 or i + 1 == n_groups:
            group_chunk_, index_chunk_ = group_chunk.copy(), index_chunk.copy()
            group_chunk, index_chunk = [],[]
            yield index_chunk_, group_chunk_
application  = pd.read_csv('../input/application_train.csv')
installments = pd.read_csv('../input/installments_payments.csv')
installments.head()
INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['AMT_INSTALMENT',
                   'AMT_PAYMENT',
                   'DAYS_ENTRY_PAYMENT',
                   'DAYS_INSTALMENT',
                   'NUM_INSTALMENT_NUMBER',
                   'NUM_INSTALMENT_VERSION'
                   ]:
        INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES.append((select, agg))
INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES)]
groupby_aggregate_names = []
for groupby_cols, specs in tqdm(INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES):
    group_object = installments.groupby(groupby_cols)
    for select, agg in tqdm(specs):
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        application = application.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
        groupby_aggregate_names.append(groupby_aggregate_name)
application.head()
application_agg = application[groupby_aggregate_names + ['TARGET']]
application_agg_corr = abs(application_agg.corr())
application_agg_corr.sort_values('TARGET', ascending=False)['TARGET']
positive_ID = application[application['TARGET']==1]['SK_ID_CURR'].tolist()
positive_ID[:4]
value_counts = installments[installments['SK_ID_CURR'].isin(positive_ID)]['SK_ID_CURR'].value_counts()
value_counts.head()
sns.distplot(value_counts)
installments_one = installments[installments['SK_ID_CURR']==328162]
installments_one.sort_values(['DAYS_INSTALMENT'],ascending=False).head(10)
# installments_ = installments[installments['SK_ID_CURR'].isin(positive_ID[:100])]
installments_ = installments.sample(10000)
installments_['instalment_paid_late_in_days'] = installments_['DAYS_ENTRY_PAYMENT'] - installments_['DAYS_INSTALMENT'] 
installments_['instalment_paid_late'] = (installments_['instalment_paid_late_in_days'] > 0).astype(int)
installments_['instalment_paid_over_amount'] = installments_['AMT_PAYMENT'] - installments_['AMT_INSTALMENT']
installments_['instalment_paid_over'] = (installments_['instalment_paid_over_amount'] > 0).astype(int)
features = pd.DataFrame({'SK_ID_CURR':installments_['SK_ID_CURR'].unique()})
groupby = installments_.groupby(['SK_ID_CURR'])
installments_.head()
feature_names = []

features, feature_names = add_features('NUM_INSTALMENT_VERSION', 
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                     features, feature_names, groupby)

features, feature_names = add_features('instalment_paid_late_in_days', 
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                     features, feature_names, groupby)

features, feature_names = add_features('instalment_paid_late', ['sum','mean'],
                                     features, feature_names, groupby)

features, feature_names = add_features('instalment_paid_over_amount', 
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                     features, feature_names, groupby)

features, feature_names = add_features('instalment_paid_over', ['sum','mean'],
                                     features, feature_names, groupby)
    
display(features.head())
def last_k_instalment_features(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['DAYS_INSTALMENT'],ascending=False, inplace=True)
    
    features = {}

    for period in periods:
        gr_period = gr_.iloc[:period]

        features = add_features_in_group(features,gr_period, 'NUM_INSTALMENT_VERSION', 
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                         'last_{}_'.format(period))
        
        features = add_features_in_group(features,gr_period, 'instalment_paid_late_in_days', 
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                         'last_{}_'.format(period))
        features = add_features_in_group(features,gr_period ,'instalment_paid_late', 
                                     ['count','mean'],
                                         'last_{}_'.format(period))
        features = add_features_in_group(features,gr_period ,'instalment_paid_over_amount', 
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                         'last_{}_'.format(period))
        features = add_features_in_group(features,gr_period,'instalment_paid_over', 
                                     ['count','mean'],
                                         'last_{}_'.format(period))
    
    return features
func = partial(last_k_instalment_features, periods=[1,5,10,20,50,100])

g = parallel_apply(groupby, func, index_name='SK_ID_CURR',
                   num_workers=16, chunk_size=10000).reset_index()
features = features.merge(g, on='SK_ID_CURR', how='left')

display(features.head())
from sklearn.linear_model import LinearRegression
def trend_in_last_k_instalment_features(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['DAYS_INSTALMENT'],ascending=False, inplace=True)
    
    features = {}

    for period in periods:
        gr_period = gr_.iloc[:period]


        features = _add_trend_feature(features,gr_period,
                                      'instalment_paid_late_in_days','{}_period_trend_'.format(period)
                                     )
        features = _add_trend_feature(features,gr_period,
                                      'instalment_paid_over_amount','{}_period_trend_'.format(period)
                                     )
    return features

def _add_trend_feature(features,gr,feature_name, prefix):
    y = gr[feature_name].values
    try:
        x = np.arange(0,len(y)).reshape(-1,1)
        lr = LinearRegression()
        lr.fit(x,y)
        trend = lr.coef_[0]
    except:
        trend=np.nan
    features['{}{}'.format(prefix,feature_name)] = trend
    return features
func = partial(trend_in_last_k_instalment_features, periods=[10,50,100,500])

g = parallel_apply(groupby, func, index_name='SK_ID_CURR',
                   num_workers=16, chunk_size=10000).reset_index()
features = features.merge(g, on='SK_ID_CURR', how='left')

display(features.head())
