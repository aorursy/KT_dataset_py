import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_predict
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import log_loss, make_scorer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge, SGDRegressor, SGDClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, GroupShuffleSplit, train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
#!pip install lightgbm
import lightgbm as lgb

from scipy import stats
from sklearn.neighbors import NearestNeighbors
train = pd.read_csv('../input/train.csv.gz')
test = pd.read_csv('../input/test.csv.gz')
train_checks = pd.read_csv('../input/train_checks.csv.gz')
test_checks = pd.read_csv('../input/test_checks.csv.gz')
train = train.merge(train_checks, on = 'check_id', how = 'left')
test = test.merge(test_checks, on = 'check_id', how = 'left')
train.fillna('', inplace=True)
test.fillna('', inplace=True)
catalog = pd.read_csv('../input/catalog2.csv.gz')
catalog1 = pd.read_csv('../input/catalog1.csv.gz')
#Dropping the category with only one value
catalog1 = catalog1.loc[catalog1.category != 'Аксессуары']
catalog3 = pd.read_csv('../input/catalog3.csv.gz')
catalog3 = catalog3.loc[catalog3.category.isnull() == False]
train.name.str.lower()[25:35]
catalog1['description'] = catalog1['description'].str.lower()
catalog1['description'] = catalog1['description'].str.replace('a', 'а')
catalog1['description'] = catalog1['description'].str.replace('h', 'н')
catalog1['description'] = catalog1['description'].str.replace('k', 'к')
catalog1['description'] = catalog1['description'].str.replace('b', 'в')
catalog1['description'] = catalog1['description'].str.replace('c', 'с')
catalog1['description'] = catalog1['description'].str.replace('o', 'о')
catalog1['description'] = catalog1['description'].str.replace('p', 'р')
catalog1['description'] = catalog1['description'].str.replace('t', 'т')
catalog1['description'] = catalog1['description'].str.replace('x', 'х')
catalog1['description'] = catalog1['description'].str.replace('y', 'у')
catalog1['description'] = catalog1['description'].str.replace('e', 'е')
catalog1['description'] = catalog1['description'].str.replace('m', 'м')

catalog3['description'] = catalog3['description'].str.lower()
catalog3['description'] = catalog3['description'].str.replace('a', 'а')
catalog3['description'] = catalog3['description'].str.replace('h', 'н')
catalog3['description'] = catalog3['description'].str.replace('k', 'к')
catalog3['description'] = catalog3['description'].str.replace('b', 'в')
catalog3['description'] = catalog3['description'].str.replace('c', 'с')
catalog3['description'] = catalog3['description'].str.replace('o', 'о')
catalog3['description'] = catalog3['description'].str.replace('p', 'р')
catalog3['description'] = catalog3['description'].str.replace('t', 'т')
catalog3['description'] = catalog3['description'].str.replace('x', 'х')
catalog3['description'] = catalog3['description'].str.replace('y', 'у')
catalog3['description'] = catalog3['description'].str.replace('e', 'е')
catalog3['description'] = catalog3['description'].str.replace('m', 'м')
train['name'] = train['name'].str.lower()
train['name'] = train['name'].str.replace('a', 'а')
train['name'] = train['name'].str.replace('h', 'н')
train['name'] = train['name'].str.replace('k', 'к')
train['name'] = train['name'].str.replace('b', 'в')
train['name'] = train['name'].str.replace('c', 'с')
train['name'] = train['name'].str.replace('o', 'о')
train['name'] = train['name'].str.replace('p', 'р')
train['name'] = train['name'].str.replace('t', 'т')
train['name'] = train['name'].str.replace('x', 'х')
train['name'] = train['name'].str.replace('y', 'у')
train['name'] = train['name'].str.replace('e', 'е')
train['name'] = train['name'].str.replace('m', 'м')

catalog['description'] = catalog['description'].str.lower()
catalog['description'] = catalog['description'].str.replace('a', 'а')
catalog['description'] = catalog['description'].str.replace('h', 'н')
catalog['description'] = catalog['description'].str.replace('k', 'к')
catalog['description'] = catalog['description'].str.replace('b', 'в')
catalog['description'] = catalog['description'].str.replace('c', 'с')
catalog['description'] = catalog['description'].str.replace('o', 'о')
catalog['description'] = catalog['description'].str.replace('p', 'р')
catalog['description'] = catalog['description'].str.replace('t', 'т')
catalog['description'] = catalog['description'].str.replace('x', 'х')
catalog['description'] = catalog['description'].str.replace('y', 'у')
catalog['description'] = catalog['description'].str.replace('e', 'е')
catalog['description'] = catalog['description'].str.replace('m', 'м')

test['name'] = test['name'].str.lower()
test['name'] = test['name'].str.replace('a', 'а')
test['name'] = test['name'].str.replace('h', 'н')
test['name'] = test['name'].str.replace('k', 'к')
test['name'] = test['name'].str.replace('b', 'в')
test['name'] = test['name'].str.replace('c', 'с')
test['name'] = test['name'].str.replace('o', 'о')
test['name'] = test['name'].str.replace('p', 'р')
test['name'] = test['name'].str.replace('t', 'т')
test['name'] = test['name'].str.replace('x', 'х')
test['name'] = test['name'].str.replace('y', 'у')
test['name'] = test['name'].str.replace('e', 'е')
test['name'] = test['name'].str.replace('m', 'м')
%%time
import scipy as sp
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 5))
tfidf_words = TfidfVectorizer(ngram_range=(1, 6))
X = sp.sparse.hstack((vectorizer.fit_transform(train.name), tfidf_words.fit_transform(train.name)))

X_test = sp.sparse.hstack((vectorizer.transform(test.name), tfidf_words.transform(test.name)))
%%time
X_catalog = sp.sparse.hstack((vectorizer.transform(catalog.description.fillna('')), tfidf_words.transform(catalog.description.fillna(''))))
catalog_labeler = LabelEncoder()
y_catalog = catalog_labeler.fit_transform(catalog.category)

labeler = LabelEncoder()
y = labeler.fit_transform(train.category)
clipping = 0.001

clipped_log_loss = make_scorer(log_loss, eps = clipping, greater_is_better = False, needs_proba = True)
%%time
model_catalog = LogisticRegression(C = 100)
model_catalog.fit(X_catalog, y_catalog)

X_meta_catalog = model_catalog.predict_proba(X)
X_catalog1 = sp.sparse.hstack((vectorizer.transform(catalog1.description.fillna("")), tfidf_words.transform(catalog1.description.fillna(""))))
y_catalog1 = catalog_labeler.fit_transform(catalog1.category)
model_catalog1 = LogisticRegression(C = 100)
model_catalog1.fit(X_catalog1, y_catalog1)

X_meta_catalog1 = model_catalog1.predict_proba(X)
X_catalog3 = sp.sparse.hstack((vectorizer.transform(catalog3.description.fillna("")), tfidf_words.transform(catalog3.description.fillna(""))))
y_catalog3 = catalog_labeler.fit_transform(catalog3.category)
model_catalog3 = LogisticRegression(C = 100)
model_catalog3.fit(X_catalog3, y_catalog3)

X_meta_catalog3 = model_catalog3.predict_proba(X)
X_meta = np.zeros((X.shape[0], 25))
X_test_meta = []
gkf = list(GroupKFold(n_splits=4).split(X, y, train.check_id.values))
for fold_i, (train_i, test_i) in enumerate(gkf):
    print(fold_i)
    model = LogisticRegression(C = 100)
    model.fit(X.tocsr()[train_i], y[train_i])
    X_meta[test_i, :] = model.predict_proba(X.tocsr()[test_i])
    X_test_meta.append(model.predict_proba(X_test))
X_test_meta = np.stack(X_test_meta)
X_test_meta_mean = np.mean(X_test_meta, axis = 0)
train_df = train.copy()
train_df['datetime'] = pd.to_datetime(train_df['datetime'])
train_df['date'] = train_df['datetime'].dt.date
train_df['check_price_mean'] = train_df.groupby('check_id')['price'].transform('mean')
train_df['check_price_median'] = train_df.groupby('check_id')['price'].transform('median')
train_df['check_price_min'] = train_df.groupby('check_id')['price'].transform('min')
train_df['check_price_max'] = train_df.groupby('check_id')['price'].transform('max')
train_df['check_count_max'] = train_df.groupby('check_id')['count'].transform('max')
train_df['check_count_sum'] = train_df.groupby('check_id')['count'].transform('sum')
train_df['check_count_min'] = train_df.groupby('check_id')['count'].transform('min')
train_df['check_count_mean'] = train_df.groupby('check_id')['count'].transform('mean')
train_df['check_count_median'] = train_df.groupby('check_id')['count'].transform('median')
train_df['check_count_count'] = train_df.groupby('check_id')['count'].transform('count')
train_df['shop_count_count'] = train_df.groupby('shop_name')['count'].transform('count')
train_df['shop_count_max'] = train_df.groupby('shop_name')['count'].transform('max')
train_df['shop_count_sum'] = train_df.groupby('shop_name')['count'].transform('sum')
train_df['shop_count_min'] = train_df.groupby('shop_name')['count'].transform('min')
train_df['shop_count_mean'] = train_df.groupby('shop_name')['count'].transform('mean')
train_df['shop_count_median'] = train_df.groupby('shop_name')['count'].transform('median')
train_df['shop_price_mean'] = train_df.groupby('shop_name')['price'].transform('mean')
train_df['shop_price_median'] = train_df.groupby('shop_name')['price'].transform('median')
train_df['shop_price_min'] = train_df.groupby('shop_name')['price'].transform('min')
train_df['shop_price_max'] = train_df.groupby('shop_name')['price'].transform('max')
train_df['shop_sum_mean'] = train_df.groupby('shop_name')['sum'].transform('mean')
train_df['shop_sum_median'] = train_df.groupby('shop_name')['sum'].transform('median')
train_df['shop_sum_min'] = train_df.groupby('shop_name')['sum'].transform('min')
train_df['shop_sum_max'] = train_df.groupby('shop_name')['sum'].transform('max')
train_df['mean_count_day'] = train_df.groupby(['shop_name'])['count'].transform('sum') / train_df.groupby(['shop_name'])['date'].transform('count')
train_df['mean_sum_day'] = train_df.groupby(['shop_name'])['sum'].transform('sum') / train_df.groupby(['shop_name'])['date'].transform('count')
train_df.fillna(train_df.mean(), inplace=True)
train_df.drop(['category', 'shop_name', 'datetime', 'date'], axis=1, inplace=True)
train_df.drop(['name'], axis=1, inplace=True)
train_df.drop(['check_id'], axis=1, inplace=True)
test_df = test.copy()
test_df['datetime'] = pd.to_datetime(test_df['datetime'])
test_df['date'] = test_df['datetime'].dt.date
test_df['check_price_mean'] = test_df.groupby('check_id')['price'].transform('mean')
test_df['check_price_median'] = test_df.groupby('check_id')['price'].transform('median')
test_df['check_price_min'] = test_df.groupby('check_id')['price'].transform('min')
test_df['check_price_max'] = test_df.groupby('check_id')['price'].transform('max')
test_df['check_count_max'] = test_df.groupby('check_id')['count'].transform('max')
test_df['check_count_sum'] = test_df.groupby('check_id')['count'].transform('sum')
test_df['check_count_min'] = test_df.groupby('check_id')['count'].transform('min')
test_df['check_count_mean'] = test_df.groupby('check_id')['count'].transform('mean')
test_df['check_count_median'] = test_df.groupby('check_id')['count'].transform('median')
test_df['check_count_count'] = test_df.groupby('check_id')['count'].transform('count')
test_df['shop_count_count'] = test_df.groupby('shop_name')['count'].transform('count')
test_df['shop_count_max'] = test_df.groupby('shop_name')['count'].transform('max')
test_df['shop_count_sum'] = test_df.groupby('shop_name')['count'].transform('sum')
test_df['shop_count_min'] = test_df.groupby('shop_name')['count'].transform('min')
test_df['shop_count_mean'] = test_df.groupby('shop_name')['count'].transform('mean')
test_df['shop_count_median'] = test_df.groupby('shop_name')['count'].transform('median')
test_df['shop_price_mean'] = test_df.groupby('shop_name')['price'].transform('mean')
test_df['shop_price_median'] = test_df.groupby('shop_name')['price'].transform('median')
test_df['shop_price_min'] = test_df.groupby('shop_name')['price'].transform('min')
test_df['shop_price_max'] = test_df.groupby('shop_name')['price'].transform('max')
test_df['shop_sum_mean'] = test_df.groupby('shop_name')['sum'].transform('mean')
test_df['shop_sum_median'] = test_df.groupby('shop_name')['sum'].transform('median')
test_df['shop_sum_min'] = test_df.groupby('shop_name')['sum'].transform('min')
test_df['shop_sum_max'] = test_df.groupby('shop_name')['sum'].transform('max')
test_df['mean_count_day'] = test_df.groupby(['shop_name'])['count'].transform('sum') / test_df.groupby(['shop_name'])['date'].transform('count')
test_df['mean_sum_day'] = test_df.groupby(['shop_name'])['sum'].transform('sum') / test_df.groupby(['shop_name'])['date'].transform('count')
#test_df['unique_categories_shop'] = test_df.groupby(['shop_name'])['category'].transform('nunique')
test_df.fillna(test_df.mean(), inplace=True)
test_shop_name = test_df.shop_name
test_df.drop(['shop_name', 'datetime', 'date'], axis=1, inplace=True)
test_text = test_df.name
test_df.drop(['name'], axis=1, inplace=True)
test_check_id = test_df['check_id']
test_df.drop(['check_id', 'id'], axis=1, inplace=True)
X_meta_ = np.hstack([X_meta, X_meta_catalog, train_df, X_meta_catalog1, X_meta_catalog3])
X_test_meta_catalog = model_catalog.predict_proba(X_test)
X_test_meta_catalog1 = model_catalog1.predict_proba(X_test)
X_test_meta_catalog3 = model_catalog3.predict_proba(X_test)
X_test_meta = np.hstack([X_test_meta_mean, X_test_meta_catalog, test_df, X_test_meta_catalog1, X_test_meta_catalog3])
X_meta_.shape, X_test_meta.shape
xgb_model = xgb.XGBClassifier(learning_rate=0.05,
n_estimators=210,
max_depth=5,
min_child_weight=2.0,
gamma=0.85,
reg_alpha=0.35,
subsample=0.75,
colsample_bytree=0.55,
objective= 'multi:softprob',
nthread=6,
scale_pos_weight=3,
seed=27)
#validation
sc = []
for train_i, test_i in gkf:
    print('Fold')
    xgb_model.fit(X_meta_[train_i], y[train_i])
    predictions = xgb_model.predict_proba(X_meta_[test_i]).reshape((-1, 25))
    score = log_loss(y[test_i], predictions, eps=0.0001)
    print(score)
    sc.append(score)
print(np.mean(sc), np.std(sc))

prediction = np.zeros((3000, 25))
sc = []
for train_i, test_i in gkf:
    print('Fold')
    xgb_model.fit(X_meta_[train_i], y[train_i])
    predictions = xgb_model.predict_proba(X_meta_[test_i]).reshape((-1, 25))
    score = log_loss(y[test_i], predictions, eps=clipping)
    print(score)
    sc.append(score)
    pred = xgb_model.predict_proba(X_test_meta)
    prediction += pred
print(np.mean(sc), np.std(sc))
prediction = prediction / 4
def form_predictions(p):
    return ['%.6f' % x for x in p]
test_submission = test[['id']]
# a very important thing was lowering the clipping value
clipping=0.0001
for i, c in enumerate(labeler.classes_):
    p = prediction[:, i]
    p[p < clipping] = clipping
    p[p > (1.0 - clipping)] = (1.0 - clipping)
    test_submission[c] = form_predictions(p)
test_submission.to_csv('xgb9.csv.gz', compression='gzip', index = False, encoding='utf-8')