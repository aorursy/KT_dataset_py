# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.stats as st
from sklearn.preprocessing import Imputer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
tr = pd.read_csv('../input/train.csv', index_col='item_id', parse_dates=['activation_date'])
tr.info()
## to see the uniquen number of each feature
for c in tr.columns:
    print("%20s"%c + "\t" + str(len(tr[c].unique())))
tr['weekday'] = tr.activation_date.dt.weekday
tr.activation_date.value_counts().sort_index()
date_distribution = tr.activation_date.value_counts().sort_index()
date_distribution.index = [x.strftime("%m-%d") for x in date_distribution.index]
date_distribution.plot.bar(figsize=(10,2))
date_target_stats = tr.groupby('activation_date')['deal_probability']\
                      .agg(['count','mean','std','var']).sort_index()
display(date_target_stats)
date_target_stats.index = [x.strftime("%m-%d") for x in date_target_stats.index]
date_target_stats['mean'].plot.bar(figsize=(10,2))
plt.show()
date_target_stats['std'].plot.bar(figsize=(10,2))
plt.show()
date_target_stats['std'].plot.hist(figsize=(10,3),bins=100)
normal_tr = tr.loc[tr.activation_date < pd.to_datetime('2017-03-29'), :] # ignore these special date data.
# define some temp variable for store stats information. e.g: 
stat_map = {} # stats result
entr_map = {} # each feature's entropy
gini_map = {} # each feature's gini
info_map = {} # each feature's information gain with target value
corr_map = {} # each feature's corr with target value
from skfeature.utility.mutual_information import information_gain as info_gain
from skfeature.utility.entropy_estimators import entropy
def plt_bar(stats, y):
    stats.plot.bar(y='count',figsize=(10,3))
    plt.legend('')
    plt.xlabel('')
    plt.title(y)
    plt.show()


def caculate_gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))


def stats_view(df, cate, target, p=75, show_common=True):
    df = df[~df[target].isnull()]
    df.loc[df[cate].isnull(),cate] = 'NAN'
    df[cate] = df[cate].astype('str')
    
    print('-------------------------------------------------------------------------')
    y = df.deal_probability
    ncat = len(df[cate].unique())
    print(20*"="+"%20s"%cate+"\t"+"%20s "%target+"="*20)
    
    if show_common:
        print('ncat:%.2f'%ncat)
        
        catv = pd.factorize(df[cate].fillna('NAN'))[0].astype(float)
        entr = st.entropy(catv)
        gini = caculate_gini(catv)
        info = info_gain(catv, y)
        corr = st.pearsonr(catv, y)[0]
        print('fact entr:%.6f'%entr)
        print('fact gini:%.6f'%gini)
        print('fact info:%.6f'%info)
        print('fact corr:%.6f'%corr)
        
        entr_map[cate+" fact"] = entr
        gini_map[cate+" fact"] = gini
        info_map[cate+" fact"] = info
        corr_map[cate+" fact"] = corr
        
        freq = df[cate].value_counts()/df.shape[0]
        catf = df[cate].map(freq).values
        entr = st.entropy(catf)
        gini = caculate_gini(catf)
        info = info_gain(catf, y)
        corr = st.pearsonr(catf, y)[0]
        print('freq entr:%.6f'%entr)
        print('freq gini:%.6f'%gini)
        print('freq info:%.6f'%info)
        print('freq corr:%.6f'%corr)
        entr_map[cate+" freq"] = entr
        gini_map[cate+" freq"] = gini
        info_map[cate+" freq"] = info
        corr_map[cate+" freq"] = corr
        
    
    target_gp = df.groupby(cate)[target]
    stats = target_gp.agg(['count', 'mean', 'std', 'var']).sort_index()
    stats['target_p%d'%p] = target_gp.agg(lambda x:np.percentile(x,p))
    stats['uv'] = df.groupby(cate)['user_id'].agg(lambda x: len(np.unique(x)))
    stats['puv'] = stats['uv'] / stats['count']
    stats = stats.fillna(0)
    display(stats.head(50))
    
    
    base_ent = st.entropy(y)
    for c in stats.columns:
        if not show_common and c in ['count', 'uv', 'puv']:
            continue
        
        dmap = stats[c].to_dict() 
        x = df[cate].apply(lambda x:dmap.get(x,-1))
        n_miss = x.isnull().sum()
        if  n_miss > len(df)*0.8:
            print("Stats {} miss value too much, ignored.".format(c), n_miss)
            continue
        
        entr = st.entropy(stats[c])
        gini = caculate_gini(x.astype(np.float).values)
        info = info_gain(x.astype(np.float).values, y.values)
        corr = st.pearsonr(x, y)[0]
        
        entr_map[cate+" "+ c] = entr
        gini_map[cate+" "+ c] = gini
        info_map[cate+" "+ c] = info
        corr_map[cate+" "+ c] = corr
        
        print("{} {} {} entropy:{}, gini:{}, information_gain:{}  corrcoef:{}"
              .format(cate, target, c, round(entr,6), round(gini,6),round(info, 6), round(corr,6)))
        if ncat>20:
            plt_bar(stats[c].sort_values(ascending=False).head(20), c)
        else:
            plt_bar(stats[c].sort_values(ascending=False), c)
    stat_map[cate+" "+ target] = stats
# 原始统计特征分析
for c in ['image_top_1','region', 'city', 'parent_category_name', 'category_name', 'param_1',
          'param_2', 'param_3', 'item_seq_number', 'user_type', 'weekday']:
    stats_view(normal_tr, c, 'deal_probability') # analysis the target value and categorical feature.
    print(100 * "=")
    if c != 'image_top_1':
        stats_view(normal_tr, c, 'image_top_1', show_common=False) # analysis image_top_1 value and categorical feature.
        print(100 * "=")
    stats_view(normal_tr, c, 'price', show_common=False) # analysis price value and categorical feature.
    print(100 * "=")
base_feature_stats = pd.DataFrame([
    pd.Series(entr_map, name='entripy'),
    pd.Series(gini_map, name='gini'),
    pd.Series(info_map, name='information gain'),
    pd.Series(corr_map, name='corr')
]).T
base_feature_stats['abs_corr'] = np.abs(base_feature_stats['corr'])
pd.set_option('display.max_rows',1000)
pd.options.display.float_format = '{:,.4f}'.format
display(base_feature_stats)
