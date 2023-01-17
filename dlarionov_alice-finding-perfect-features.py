import pickle
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
times = ['time%s' % i for i in range(1, 11)]
sites = ['site%s' % i for i in range(1, 11)]

train = pd.read_csv('../input/train_sessions.csv', parse_dates = times, index_col='session_id')
test = pd.read_csv('../input/test_sessions.csv', parse_dates = times, index_col='session_id')

train.sort_values(by='time1', inplace=True)

idx = train.shape[0]
data = pd.concat([train, test], sort=False) # leave train.target for eda

train.shape, test.shape, data.shape
data[sites] = data[sites].fillna(0).astype(np.uint16) # float->int (55.0 -> 55)

# for each row combine site_ids into one string separated by space
data['words'] = data[sites].astype(np.str).apply(' '.join, axis=1)

#words = CountVectorizer(max_features=50000, ngram_range=(1, 3)).fit_transform(data['words'])
words = TfidfVectorizer(max_features=50000, ngram_range=(1, 3)).fit_transform(data['words'])

data.drop(['words'], inplace=True, axis=1)
words
model = LogisticRegression(random_state=17, solver='liblinear')
time_split = TimeSeriesSplit(n_splits=10)
train.time1.min(), train.time1.max(), test.time1.min(), test.time1.max()
X_train = words[:idx]
y_train = train.target

cv_scores = cross_val_score(model, X_train, y_train, cv=time_split, scoring='roc_auc')
cv_scores, cv_scores.mean()

# 0.8670500571969433 CountVectorizer
# 0.8664051910501502 TfidfVectorizer
data['min'] = data[times].min(axis=1)
data['max'] = data[times].max(axis=1)
data['seconds'] = ((data['max'] - data['min']) / np.timedelta64(1, 's'))
data['minutes'] = ((data['max'] - data['min']) / np.timedelta64(1, 'm')).round(2)
data.drop(['min','max'], inplace=True, axis=1)

data['month'] = data['time1'].apply(lambda ts: ts.month+(12*(ts.year-2013))).astype(np.int8)
data['yyyymm'] = data['time1'].apply(lambda ts: 100 * ts.year + ts.month).astype(np.int32) # wtf! why this works?
data['mm'] = data['time1'].apply(lambda ts: ts.month).astype(np.int8)
data['yyyy'] = data['time1'].apply(lambda ts: ts.year).astype(np.int8)

data['dayofweek'] = data['time1'].apply(lambda ts: ts.dayofweek).astype(np.int8)
data['weekend'] = data['time1'].apply(lambda ts: ts.dayofweek > 5).astype(np.int8)

data['hour'] = data['time1'].apply(lambda ts: ts.hour).astype(np.int8)
hosts = pd.read_pickle('../input/site_dic.pkl')
hosts = pd.DataFrame(data=list(hosts.keys()), index=list(hosts.values()), columns=['name']) # switch key and value

hosts['split'] = hosts['name'].str.split('.')
hosts['len'] = hosts['split'].map(lambda x: len(x)).astype(np.int8)
hosts['domain'] = hosts['split'].map(lambda x: x[-1])

hosts.drop(['name','split'], inplace=True, axis=1)
hosts.index.rename('site1', inplace=True) # rename index for the future merge
data = pd.merge(data, hosts, how='left', on='site1')
data.columns
_, axes = plt.subplots(nrows=1, ncols=3, figsize=(22,6))
sns.boxplot(x='target', y='minutes', data=data[:idx], ax=axes[0])
sns.violinplot(x='target', y='minutes', data=data[:idx], ax=axes[1])
sns.boxplot(x='target', y='seconds', data=data[:idx], ax=axes[2])
axes[0].set_ylim(-1,5), axes[1].set_ylim(-1,5), axes[2].set_ylim(-30,300);
data['short'] = data['minutes'].map(lambda x: x < 0.8).astype(np.int8)
data['long'] = data['minutes'].map(lambda x: x >= 0.8).astype(np.int8)
_, axes = plt.subplots(nrows=1, ncols=3, figsize=(22,4))
sns.countplot(x="dayofweek", data=data[data.target==1][:idx], ax=axes[0]) # Alice
sns.countplot(x="dayofweek", data=data[data.target==0][:idx], ax=axes[1]) # Not Alice
sns.countplot(x="dayofweek", data=data[idx:], ax=axes[2]); # Test
data["online_day"] = data['time1'].apply(lambda ts: ts.dayofweek in [0,1,3,4]).astype(np.int8)
data["mon"] = data['time1'].apply(lambda ts: ts.dayofweek in [0]).astype(np.int8) # monday
data["wen"] = data['time1'].apply(lambda ts: ts.dayofweek in [2]).astype(np.int8) # wensday
data["sun"] = data['time1'].apply(lambda ts: ts.dayofweek in [6]).astype(np.int8) # sunday
_, axes = plt.subplots(nrows=1, ncols=3, figsize=(22,4))
sns.countplot(x="month", data=data[data.target==1][:idx], ax=axes[0]) # Alice
sns.countplot(x="month", data=data[data.target==0][:idx], ax=axes[1]) # Not Alice
sns.countplot(x="month", data=data[idx:], ax=axes[2]); # Test
agg = data[data.target==1].groupby(['mm']).seconds.agg({ 'mean', 'sum', 'count'})
agg

# TODO exploit aggregates
# data = pd.merge(data, agg, how='left', on='mm')
_, axes = plt.subplots(nrows=1, ncols=3, figsize=(22,4))
sns.countplot(x="hour", data=data[data.target==1][:idx], ax=axes[0]) # Alice
sns.countplot(x="hour", data=data[data.target==0][:idx], ax=axes[1]) # Not Alice
sns.countplot(x="hour", data=data[idx:], ax=axes[2]); # Test
''' wtf?
data['morning'] = data['time1'].apply(lambda ts: (ts.hour >= 8) & (ts.hour < 12)).astype(np.int8)
data['day'] = data['time1'].apply(lambda ts: (ts.hour >= 12) & (ts.hour < 15)).astype(np.int8)
data['evening'] = data['time1'].apply(lambda ts: (ts.hour >= 15) & (ts.hour < 19)).astype(np.int8)
data['night'] = data['time1'].apply(lambda ts: (ts.hour >= 19) | (ts.hour < 8)).astype(np.int8) # or!
'''

data['morning'] = data['time1'].apply(lambda ts: (ts.hour >= 7) & (ts.hour < 12)).astype(np.int8)
data['day'] = data['time1'].apply(lambda ts: (ts.hour >= 12) & (ts.hour < 18)).astype(np.int8)
data['evening'] = data['time1'].apply(lambda ts: (ts.hour >= 18) & (ts.hour < 23)).astype(np.int8)
data['night'] = data['time1'].apply(lambda ts: (ts.hour >= 23) | (ts.hour < 7)).astype(np.int8) # or!
_, axes = plt.subplots(nrows=1, ncols=3, figsize=(22,4))
sns.countplot(x="len", data=data[data.target==1][:idx], ax=axes[0]) # Alice
sns.countplot(x="len", data=data[data.target==0][:idx], ax=axes[1]) # Not Alice
sns.countplot(x="len", data=data[idx:], ax=axes[2]); # Test
data['big_site'] = data['len'].apply(lambda x: x > 5).astype(np.int8)
data['typical_site'] = data['len'].apply(lambda x: x == 3).astype(np.int8)
_, axes = plt.subplots(nrows=1, ncols=1, figsize=(22,4))
sns.countplot(x="domain", data=data[data.target==1][:idx], ax=axes); # Alice
data['typical_domain'] = data['domain'].map(lambda x: x in ('com', 'fr', 'net', 'uk', 'org', 'tv')).astype(np.int)
data.drop(times + sites + ['target'], inplace=True, axis=1)
data.to_pickle('dump.pkl')
data.columns
data = pd.read_pickle('dump.pkl')

# commented columns go to model
data.drop([
    'seconds', 
    'minutes', 
    'month', 
    #'yyyymm', 
    'mm', 
    'yyyy', 
    'dayofweek',
    'weekend', 
    'hour', 
    'len', 
    'domain', 
    'short', 
    'long',
    'online_day',
    'mon',
    'wen',
    'sun',
    #'morning', 
    #'day', 
    #'evening', 
    #'night', 
    'big_site',
    'typical_site',
    'typical_domain',
], inplace=True, axis=1)

data = pd.get_dummies(data, columns=[
    #'yyyy',
    #'mm',
    #'dayofweek',
    #'hour',
    #'len'
])

features_to_scale = [
    #'seconds',
    #'minutes',
    #'month',
    'yyyymm',
    #'dayofweek',
    #'hour',
    #'len',
]
data[features_to_scale] = StandardScaler().fit_transform(data[features_to_scale])
X_train = csr_matrix(hstack([words[:idx], data[:idx]]))
cv_scores = cross_val_score(model, X_train, y_train, cv=time_split, scoring='roc_auc')   
data.columns, cv_scores, cv_scores.mean()
X_train = csr_matrix(hstack([words[:idx], data[:idx]]))
y_train = train.target

params = {
    'C': np.logspace(-2, 2, 10),
    'penalty': ['l1','l2']
}

grid = GridSearchCV(estimator=model, param_grid=params, scoring='roc_auc', cv=time_split, verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

grid.best_estimator_, grid.best_score_, grid.best_params_
model = grid.best_estimator_
model.fit(X_train, y_train)

X_test = csr_matrix(hstack([words[idx:], data[idx:]]))
y_test = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({"session_id": test.index, "target": y_test})
submission.to_csv('submission.csv', index=False)
