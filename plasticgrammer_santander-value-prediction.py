import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print('train:', train.shape, 'test:', test.shape)
train.head()
print(train['target'].describe())

fig, ax = plt.subplots(1, 2, figsize=(12, 3))
sns.distplot(train['target'], ax=ax[0])
sns.distplot(np.log1p(train['target']), ax=ax[1])
train.isnull().any().any()
test.isnull().any().any()
PERC_TRESHOLD = 0.99
cols_to_drop = [col for col in train.columns[2:]
        if [i[1] for i in list(train[col].value_counts().items()) if i[0] == 0][0] >= train.shape[0] * PERC_TRESHOLD]
print('drop features: {}'.format(len(cols_to_drop)))
all_data = train.append(test, sort=False)
all_data.drop(cols_to_drop, axis=1, inplace=True)

print(all_data.shape)
data_agg = pd.DataFrame()

data_agg['svp_quantile'] = all_data[all_data > 0].quantile(0.5, axis=1)
data_agg['svp_max'] = all_data.max(axis=1)
data_agg['svp_sum_nonzeroval'] = (all_data != 0).sum(axis=1)
data_agg['svp_skew'] = all_data.skew(axis=1)
data_agg['svp_kurtosis'] = all_data.kurtosis(axis=1)
data_agg['svp_sum_all'] = all_data.sum(axis=1)
data_agg['svp_variance'] = all_data.var(axis=1)

data_agg.head()
all_data = pd.concat([all_data, data_agg], axis=1)
cols = [x for x in all_data.columns if not x in ['ID','target']]

for i, t in all_data.loc[:, cols].dtypes.iteritems():
    if t == object:
        all_data[i].fillna(all_data[i].mode()[0], inplace=True)
        all_data[i] = LabelEncoder().fit_transform(all_data[i].astype(str))
    else:
        all_data[i].fillna(all_data[i].median(), inplace=True)
_train = all_data[all_data['target'].notnull()]
_test = all_data[all_data['target'].isnull()].drop('target', axis=1)

X_train = _train.drop(['target','ID'], axis=1)
Y_train = _train['target']
X_test  = _test.drop(['ID'], axis=1)

print(X_train.shape, Y_train.shape, X_test.shape)
_ = '''
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression

feat = SelectKBest(f_regression, k=10)
X_train = pd.DataFrame(feat.fit_transform(X_train, Y_train))
X_test = pd.DataFrame(feat.transform(X_test))
_train.drop(['target','ID'], axis=1).loc[:,feat.get_support()].columns
'''
_ = '''
SelectKBest(f_regression, k=10)
['26fc93eb7', 'ac30af84a', '5bc7ab64f', '58e2e02e6', '9fd594eec','cbbc9c431', 'f190486d6', 'f74e8f13d', '555f18bd3', '6b119d8ce']
       
SelectKBest(f_regression, k=50)
['64534cc93', 'ad207f7bb', '0e1f6696a', '6d2ece683', '26fc93eb7', 'ac30af84a', 'fb49e4212', '6eef030c1', 'fc99f9426', '93ca30057',
       'e4159c59e', 'f3cf9341c', '38e6f8d32', '174edf08a', '51707c671', '341daa7d1', '53a550111', '7af000ac2', '9c42bff81', '134ac90df',
       '88d29cfaf', 'ea4887e6b', '41bc25fef', 'd4c1de0e2', '5bc7ab64f', '58e2e02e6', '9fd594eec', '3a62b36bd', 'f296082ec', '1fd0a1f2a',
       'e1d0e11b5', '0d5215715', 'ba4ceabc5', 'cbbc9c431', '698d05d29', 'aeff360c7', '96b6bd42b', 'e8d9394a0', '70feb1494', 'd79736965',
       'f190486d6', '429687d5a', 'f74e8f13d', '9d5c7cb94', '1702b5bf0', 'f1851d155', 'a8dd5cea5', 'f14b57b8f', '555f18bd3', '6b119d8ce']

SelectKBest(mutual_info_regression, k=10)
['6619d81fc', '024c577b9', '2ec5b290f', '241f0f867', '9fd594eec', 'fb0f5dbfe', 'f190486d6', '190db8488', 'c47340d97', '23310aa6f']
       
SelectKBest(mutual_info_regression, k=50)
['20aa07010', '963a49cdc', 'b30e932ba', '935ca66a9', '26fc93eb7', '0572565c2', '66ace2992', '0c9462c08', 'fb49e4212', '6619d81fc',
       '6eef030c1', 'fc99f9426', '1db387535', 'b6c0969a2', 'b43a7cfd5', '024c577b9', '2ec5b290f', '9306da53f', '0ff32eb98', '58e056e12',
       '241f0f867', '1931ccfdd', '83635fb67', 'f02ecb19c', '58e2e02e6', '9fd594eec', 'fb0f5dbfe', '91f701ba2', 'ca2b906e8', '703885424',
       'eeb9cd3aa', '324921c7b', '58232a6fb', '491b9ee45', 'c8d582dd2', 'd6bb78916', '70feb1494', 'adb64ff71', '11e12dbe8', '62e59a501',
       '15ace8c9f', '5c6487af1', 'f190486d6', 'f74e8f13d', 'c5a231d81', 'e176a204a', '1702b5bf0', '190db8488', 'c47340d97', '23310aa6f']

f3 = f1 + f2
res = [e for e in set(f3) if f3.count(e) > 1]
['58e2e02e6', '1702b5bf0', '9fd594eec', 'f190486d6', 'f74e8f13d', 'fb49e4212', 'fc99f9426', '6eef030c1', '70feb1494', '26fc93eb7']
'''
best_f = ['9fd594eec', 'f190486d6', '58e2e02e6', 'f74e8f13d', '26fc93eb7',
    '1702b5bf0', 'fb49e4212', 'fc99f9426', '6eef030c1', '70feb1494',
    'ac30af84a', '5bc7ab64f', 'cbbc9c431', '555f18bd3', '6b119d8ce',
    '6619d81fc', '024c577b9', '2ec5b290f', '241f0f867', 'fb0f5dbfe', '190db8488', 'c47340d97', '23310aa6f',
    'svp_quantile','svp_max','svp_sum_nonzeroval','svp_skew','svp_kurtosis','svp_sum_all','svp_variance']

X_train = X_train[best_f]
X_test = X_test[best_f]

print('train:', X_train.shape, 'test:', X_test.shape)
from sklearn.cluster import KMeans

for ncl in (2, 3, 5, 7, 11, 13, 17, 19, 23, 27):
    km = KMeans(n_clusters=ncl)
    X_train['kmeans_cluster_' + str(ncl)] = km.fit_predict(X_train).astype(float)
    X_test['kmeans_cluster_' + str(ncl)] = km.predict(X_test).astype(float)
from sklearn import ensemble, metrics
from sklearn import linear_model, preprocessing
from sklearn.model_selection import cross_val_score
scaler = preprocessing.StandardScaler();
scaler.fit(X_train)

reg = linear_model.ElasticNet(alpha=1, l1_ratio=0.01, max_iter=10000, positive=True, random_state=42)
def rmse_cv(model, x, y):
    rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=5))
    return rmse

score = rmse_cv(reg, scaler.transform(X_train), np.maximum(0, np.log1p(Y_train)))
print(round(score.mean(), 5)) #1.65355
reg.fit(scaler.transform(X_train), np.log1p(Y_train))
pred = reg.predict(scaler.transform(X_test))

result = np.expm1(pred)
submission = pd.DataFrame({
    "ID": test["ID"],
    "target": result
})
submission.to_csv("submission.csv", index=False)
assert len(submission[submission['target'] <= 0]) == 0
submission.head(10)
