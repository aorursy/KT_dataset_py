# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from kaggle.competitions import twosigmanews
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from sys import getsizeof
print(os.listdir("../input"))
env = twosigmanews.make_env()
print('Done!')

isTestCode = True

print('Is a test?\t', isTestCode)
# Any results you write to the current directory are saved as output.
sample_frac = 0.1
market_data, news_data = env.get_training_data()
market_data = market_data.sample(frac=sample_frac, random_state=2018)
news_data = news_data.sample(frac=sample_frac, random_state=2018)
gc.collect()
print('shape of market_data:\t', market_data.shape)
print('shape of news_data:\t', news_data.shape)
print('memory(market_data):\t', getsizeof(market_data))
print('memory(news_data):\t', getsizeof(news_data))
if isTestCode:
    market_data = market_data.tail(10000)
    news_data = news_data.tail(50000)
news_cols_agg = {
    'urgency': ['min', 'count'],
    'takeSequence': ['max'],
    'bodySize': ['min', 'max', 'mean', 'std'],
    'wordCount': ['min', 'max', 'mean', 'std'],
    'sentenceCount': ['min', 'max', 'mean', 'std'],
    'companyCount': ['min', 'max', 'mean', 'std'],
    'marketCommentary': ['mean', 'std'],
    'relevance': ['min', 'max', 'mean', 'std'],
    'sentimentNegative': ['min', 'max', 'mean', 'std'],
    'sentimentNeutral': ['min', 'max', 'mean', 'std'],
    'sentimentPositive': ['min', 'max', 'mean', 'std'],
    'sentimentWordCount': ['min', 'max', 'mean', 'std'],
    'noveltyCount12H': ['min', 'max', 'mean', 'std'],
    'noveltyCount24H': ['min', 'max', 'mean', 'std'],
    'noveltyCount3D': ['min', 'max', 'mean', 'std'],
    'noveltyCount5D': ['min', 'max', 'mean', 'std'],
    'noveltyCount7D': ['min', 'max', 'mean', 'std'],
    'volumeCounts12H': ['min', 'max', 'mean', 'std'],
    'volumeCounts24H': ['min', 'max', 'mean', 'std'],
    'volumeCounts3D': ['min', 'max', 'mean', 'std'],
    'volumeCounts5D': ['min', 'max', 'mean', 'std'],
    'volumeCounts7D': ['min', 'max', 'mean', 'std']
}

def market_news_join(market_data, news_data, news_cols_agg=news_cols_agg, agg_day=7):
    news_cols = ['time', 'assetName'] + sorted(news_cols_agg.keys())
    
    # delete the unnecessary data
    news_data = news_data[news_data.time<=market_data.time.max()]
    news_data = news_data[news_data.time>=(market_data.time.min()-np.timedelta64(agg_day,'D'))]
    
#     print('shape(news in joinFun):\t', news_data.shape)
#     print('memory(news in joinFun):\t', getsizeof(news_data))
    
#     print('max_time:\t', news_data.time.max())
#     print('min_time:\t', news_data.time.min())
    
    merge_data = pd.merge(market_data, news_data[news_cols], how='inner', on='assetName', suffixes=(['_market','_news']))
    
    # window limit
    merge_data = merge_data[np.array((merge_data.time_market-merge_data.time_news).dt.days<agg_day)&np.array((merge_data.time_market-merge_data.time_news).dt.days>=0)]
    group_data = merge_data.groupby(by=['assetName']).agg(news_cols_agg)
    
    group_data.columns = ['_'.join(col).strip() for col in group_data.columns.values]
    
    result = market_data.join(group_data, on=['assetName'])
    
    # release memory
    del merge_data
    del group_data
#     print(gc.collect())
    
    return result
# sorted by time
news_data = news_data[['time', 'assetName']+sorted(news_cols_agg.keys())]
news_data = news_data.sort_values(by=['time'])
market_data = market_data.sort_values(by=['time'])
print(gc.collect())
slice_list = []
slice_size = 100000
for i in range(int(market_data.shape[0]/slice_size)+1):
    slice_data = market_news_join(market_data.iloc[(i*slice_size):((i+1)*slice_size)], news_data)
    slice_list.append(slice_data)
    
del market_data
del news_data
print('release memory:\t', gc.collect())
dataTrain = pd.concat(slice_list)
feature_used_to_train = dataTrain.columns.tolist()
feature_used_to_train.remove('time')
feature_used_to_train.remove('assetName')
feature_used_to_train.remove('assetCode')
feature_used_to_train.remove('returnsOpenNextMktres10')
feature_used_to_train.remove('universe')
def metricFun(test_pre, test_data):
    data_score = pd.DataFrame({'time':test_data.time, 'val':test_data.returnsOpenNextMktres10 * test_pre})
    day_sum = data_score.groupby(by=['time']).sum()
    score = day_sum.val.mean()/day_sum.val.std()
    return score
kf = KFold(n_splits=5, shuffle=True, random_state=2018)
scorelist = []
modellist = []
for train_index, test_index in kf.split(dataTrain):
    train_x = dataTrain.iloc[train_index][feature_used_to_train]
    train_y = dataTrain.iloc[train_index].returnsOpenNextMktres10
    
    test_x = dataTrain.iloc[test_index][feature_used_to_train]
    test_y = dataTrain.iloc[test_index]
    
    # the rows whose universe==1 are selected
    test_x = test_x[test_y.universe==1.0]
    test_y = test_y[test_y.universe==1.0]
    
    model = lgb.LGBMClassifier(num_leaves=127, random_state=2018)
    train_y = 2*(train_y>0).astype(int)-1
    model.fit(train_x, train_y)
    
    test_pre = model.predict_proba(test_x)[:, 1]
    test_pre = 2*test_pre-1
    
    score = metricFun(test_pre, test_y)
    
    print('-'*50)
    print('accuracy:\t', (np.sum((test_pre>0) == (test_y.returnsOpenNextMktres10>0)))/test_y.shape[0] )
    print('score:\t', score)
    scorelist.append(score)
    modellist.append(model)
    
print('mean of scores:\t', np.array(scorelist).mean())
print('std of scores:\t', np.array(scorelist).std())
import matplotlib.pyplot as plt
feature_importances = np.sum([model.feature_importances_ for model in modellist], axis=0)
feature_importances = pd.Series(feature_importances)
feature_importances.index = feature_used_to_train
plt.figure(figsize=(20, 5))
feature_importances.sort_values(ascending=False).plot(kind='bar')
plt.show()
del train_x
del train_y

del test_x
del test_y

del dataTrain

print('release:\t', gc.collect())
days = env.get_prediction_days()
count=0
for (market_data_pre, news_data_pre, template) in days:
    prelist = []
    
    data_pre = market_news_join(market_data_pre, news_data_pre)[feature_used_to_train]
    
    for model in modellist:
        out_pre = model.predict_proba(data_pre.apply(np.float64))[:, 1]
        out_pre = 2*out_pre-1
        
#         out_pre = model.predict(test_x)
        
        
        prelist.append(out_pre)
    
    template.confidenceValue=np.clip(np.mean(np.array(prelist), axis=0), -1, 1)
    env.predict(template)
    count += 1
    print(count, end=' ')
print('Done')
env.write_submission_file()