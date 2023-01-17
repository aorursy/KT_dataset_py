import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from contextlib import contextmanager
from operator import itemgetter
import time
from typing import List, Dict

import keras as ks
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.sparse import vstack

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

with timer('reading data'):
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')

with timer('imputation'):
    train['param_1'].fillna('missing', inplace=True)
    test['param_1'].fillna('missing', inplace=True)
    train['param_2'].fillna('missing', inplace=True)
    test['param_2'].fillna('missing', inplace=True)
    train['param_3'].fillna('missing', inplace=True)
    test['param_3'].fillna('missing', inplace=True)
    train['image_top_1'].fillna(0, inplace=True)
    test['image_top_1'].fillna(0, inplace=True)
    train['price'].fillna(0, inplace=True)
    test['price'].fillna(0, inplace=True)
    train['price'] = np.log1p(train['price'])
    test['price'] = np.log1p(test['price'])
    price_mean = train['price'].mean()
    price_std = train['price'].std()
    train['price'] = (train['price'] - price_mean) / price_std
    test['price'] = (test['price'] - price_mean) / price_std
    train['description'].fillna('', inplace=True)
    test['description'].fillna('', inplace=True)
    # City names are duplicated across region, HT: Branden Murray https://www.kaggle.com/c/avito-demand-prediction/discussion/55630#321751
    train['city'] = train['city'] + '_' + train['region']
    test['city'] = test['city'] + '_' + test['region']

with timer('add new features'):
    cat_cols = ['region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'user_type']
    num_cols = ['price', 'deal_probability']
    for c in cat_cols:
        for c2 in num_cols:
            enc = train.groupby(c)[c2].agg(['mean']).astype(np.float32).reset_index()
            enc.columns = ['_'.join([str(c), str(c2), str(c3)]) if c3 != c else c for c3 in enc.columns]
            train = pd.merge(train, enc, how='left', on=c)
            test = pd.merge(test, enc, how='left', on=c)
    del(enc)

train.head()
from sklearn.model_selection import train_test_split

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    ex_col = ['item_id', 'user_id', 'deal_probability', 'title', 'param_1', 'param_2', 'param_3', 'activation_date']
    df['description_len'] = df['description'].map(lambda x: len(str(x))).astype(np.float16) #Lenth
    df['description_wc'] = df['description'].map(lambda x: len(str(x).split(' '))).astype(np.float16) #Word Count
    df['description'] = (df['parent_category_name'] + ' ' + df['category_name'] + ' ' + df['param_1'] + ' ' + df['param_2'] + ' ' + df['param_3'] + ' ' +
                        df['title'] + ' ' + df['description'].fillna(''))
    df['description'] = df['description'].str.lower().replace(r"[^[:alpha:]]", " ")
    df['description'] = df['description'].str.replace(r"\\s+", " ")
    df['title_len'] = df['title'].map(lambda x: len(str(x))).astype(np.float16) #Lenth
    df['title_wc'] = df['title'].map(lambda x: len(str(x).split(' '))).astype(np.float16) #Word Count
    df['image'] = df['image'].map(lambda x: 1 if len(str(x))>0 else 0)
    df['price'] = np.log1p(df['price'].fillna(0))
    df['wday'] = pd.to_datetime(df['activation_date']).dt.dayofweek
    col = [c for c in df.columns if c not in ex_col]
    return df[col]

with timer('process train'):
    train, valid = train_test_split(train, test_size=0.05, shuffle=True, random_state=37)
    y_train = train['deal_probability'].values
    X_train = preprocess(train)
    print(f'X_train: {X_train.shape}')

with timer('process valid'):
    X_valid = preprocess(valid)
    print(f'X_valid: {X_valid.shape}')

with timer('process test'):
    X_test = preprocess(test)
    print(f'X_test: {X_test.shape}')

X_train.head()
# Do some normalization
desc_len_mean = X_train['description_len'].mean()
desc_len_std = X_train['description_len'].std()
X_train['description_len'] = (X_train['description_len'] - desc_len_mean) / desc_len_std
X_valid['description_len'] = (X_valid['description_len'] - desc_len_mean) / desc_len_std
X_test['description_len'] = (X_test['description_len'] - desc_len_mean) / desc_len_std

desc_wc_mean = X_train['description_wc'].mean()
desc_wc_std = X_train['description_wc'].std()
X_train['description_wc'] = (X_train['description_wc'] - desc_wc_mean) / desc_wc_std
X_valid['description_wc'] = (X_valid['description_wc'] - desc_wc_mean) / desc_wc_std
X_test['description_wc'] = (X_test['description_wc'] - desc_wc_mean) / desc_wc_std

title_len_mean = X_train['title_len'].mean()
title_len_std = X_train['title_len'].std()
X_train['title_len'] = (X_train['title_len'] - title_len_mean) / title_len_std
X_valid['title_len'] = (X_valid['title_len'] - title_len_mean) / title_len_std
X_test['title_len'] = (X_test['title_len'] - title_len_mean) / title_len_std

title_wc_mean = X_train['title_wc'].mean()
title_wc_std = X_train['title_wc'].std()
X_train['title_wc'] = (X_train['title_wc'] - title_wc_mean) / title_wc_std
X_valid['title_wc'] = (X_valid['title_wc'] - title_wc_mean) / title_wc_std
X_test['title_wc'] = (X_test['title_wc'] - title_wc_mean) / title_wc_std

image_top_1_mean = X_train['image_top_1'].mean()
image_top_1_std = X_train['image_top_1'].std()
X_train['image_top_1'] = (X_train['image_top_1'] - image_top_1_mean) / image_top_1_std
X_valid['image_top_1'] = (X_valid['image_top_1'] - image_top_1_mean) / image_top_1_std
X_test['image_top_1'] = (X_test['image_top_1'] - image_top_1_mean) / image_top_1_std
# I don't know why I need to fill NA a second time, but alas here we are...
X_train.fillna(0, inplace=True)
X_valid.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)
X_train.columns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

with timer('TFIDF'):
    tfidf = TfidfVectorizer(ngram_range=(1, 2),
                            max_features=100000,
                             token_pattern='\w+',
                            encoding='KOI8-R')
    tfidf_train = tfidf.fit_transform(X_train['description'])
    tfidf_valid = tfidf.transform(X_valid['description'])
    tfidf_test = tfidf.transform(X_test['description'])
with timer('Dummy'):
    dummy_cols = ['parent_category_name', 'category_name', 'user_type', 'image_top_1', 'wday', 'region', 'city']
    for col in dummy_cols:
        le = LabelEncoder()
        le.fit(X_train[col] + X_valid[col] + X_test[col])
        le.fit(list(X_train[col].values.astype('str')) + list(X_valid[col].values.astype('str')) + list(X_test[col].values.astype('str')))
        X_train[col] = le.transform(list(X_train[col].values.astype('str')))
        X_valid[col] = le.transform(list(X_valid[col].values.astype('str')))
        X_test[col] = le.transform(list(X_test[col].values.astype('str')))

with timer('Dropping'):
    X_train.drop('description', axis=1, inplace=True)
    X_valid.drop('description', axis=1, inplace=True)
    X_test.drop('description', axis=1, inplace=True)

with timer('OHE'):
    ohe = OneHotEncoder(categorical_features=[X_train.columns.get_loc(c) for c in dummy_cols])
    X_train = ohe.fit_transform(X_train)
    print(f'X_train: {X_train.shape}')
    X_valid = ohe.transform(X_valid)
    print(f'X_valid: {X_valid.shape}')
    X_test = ohe.transform(X_test)
    print(f'X_test: {X_test.shape}')
def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond  = tf.keras.backend.abs(error) < clip_delta
    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
    return tf.where(cond, squared_loss, linear_loss)

def fit_predict(xs, y_train, loss_fn='mean_squared_error') -> np.ndarray:
    X_train, X_test = xs
    config = tf.ConfigProto(
        intra_op_parallelism_threads=4, use_per_session_threads=4, inter_op_parallelism_threads=4)
    with tf.Session(graph=tf.Graph(), config=config) as sess, timer('fit_predict'):
        ks.backend.set_session(sess)
        model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
        out = ks.layers.Dense(192, activation='relu')(model_in)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(1)(out)
        model = ks.Model(model_in, out)
        model.compile(loss=loss_fn, optimizer=ks.optimizers.Adam(lr=2e-3))
        for i in range(3):
            with timer(f'epoch {i + 1}'):
                model.fit(x=X_train, y=y_train, batch_size=2**(8 + i), epochs=1, verbose=0)
        return model.predict(X_test, batch_size=2**(8 + i))[:, 0]

X_train = X_train.tocsr()
X_valid = X_valid.tocsr()
X_test = X_test.tocsr()
X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, test_size = 0.5, shuffle = False)    

preds_oofs = []
preds_valids = []
preds_tests = []
for r in range(8):
    with timer('Round {}'.format(r)):
        if r % 2 == 0:
            loss_name = 'huber_loss'
            loss = huber_loss
        else:
            loss_name = 'mean_squared_error'
            loss = 'mean_squared_error'
        if r >= 4:
            print('Running loss = {}, binary = True'.format(loss_name))
            xs = [x.astype(np.bool).astype(np.float32) for x in [X_train_1, X_train_2]]
            y_pred1 = fit_predict(xs, y_train=y_train_1, loss_fn=loss)
            xs = [x.astype(np.bool).astype(np.float32) for x in [X_train_2, X_train_1]]
            y_pred2 = fit_predict(xs, y_train=y_train_2, loss_fn=loss)
            xs = [x.astype(np.bool).astype(np.float32) for x in [X_train_1, X_valid]]
            y_predf = fit_predict(xs, y_train=y_train_1, loss_fn=loss)
            xs = [x.astype(np.bool).astype(np.float32) for x in [X_train_1, X_test]]
            y_predt = fit_predict(xs, y_train=y_train_1, loss_fn=loss)
        else:
            print('Running loss = {}, binary = False'.format(loss_name))
            xs = [X_train_1, X_train_2]
            y_pred1 = fit_predict(xs, y_train=y_train_1, loss_fn=loss)
            xs = [X_train_2, X_train_1]
            y_pred2 = fit_predict(xs, y_train=y_train_2, loss_fn=loss)
            xs = [X_train_1, X_valid]
            y_predf = fit_predict(xs, y_train=y_train_1, loss_fn=loss)
            xs = [X_train_1, X_test]
            y_predt = fit_predict(xs, y_train=y_train_1, loss_fn=loss)
        preds_oof = np.concatenate((y_pred2, y_pred1), axis=0)
        preds_valid = y_predf
        preds_test = y_predt
        print('Round {} OOF RMSE: {:.4f}'.format(r, np.sqrt(mean_squared_error(train['deal_probability'], preds_oof))))
        print('Round {} Valid RMSE: {:.4f}'.format(r, np.sqrt(mean_squared_error(valid['deal_probability'], preds_valid))))
        preds_oofs.append(preds_oof)
        preds_valids.append(preds_valid)
        preds_tests.append(preds_test)
preds_oof = np.mean(preds_oofs, axis=0)
print('Overall OOF RMSE: {:.4f}'.format(np.sqrt(mean_squared_error(train['deal_probability'], preds_oof))))
preds_valid = np.mean(preds_valids, axis=0)
print('Overall Valid RMSE: {:.4f}'.format(np.sqrt(mean_squared_error(valid['deal_probability'], preds_valid))))
# As we can see, the individual submodels have very low correlation with each other!
import numpy as np
np.mean(np.corrcoef(preds_oofs), axis=0)
with timer('reading data'):
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    target = train['deal_probability']

with timer('imputation'):
    train['param_1'].fillna('missing', inplace=True)
    test['param_1'].fillna('missing', inplace=True)
    train['param_2'].fillna('missing', inplace=True)
    test['param_2'].fillna('missing', inplace=True)
    train['param_3'].fillna('missing', inplace=True)
    test['param_3'].fillna('missing', inplace=True)
    train['price'].fillna(0, inplace=True)
    test['price'].fillna(0, inplace=True)
    # City names are duplicated across region, HT: Branden Murray https://www.kaggle.com/c/avito-demand-prediction/discussion/55630#321751
    train['city'] = train['city'] + '_' + train['region']
    test['city'] = test['city'] + '_' + test['region']

with timer('FE'):
    trainp = preprocess(train)
    testp = preprocess(test)
    for col in ['param_1', 'param_2', 'param_3']:
        trainp[col] = train[col]
        testp[col] = test[col]

print(train.shape)
train.head()
with timer('drop'):
    trainp.drop(['description', 'image'], axis=1, inplace=True)
    testp.drop(['description', 'image'], axis=1, inplace=True)
print(trainp.shape)
print(testp.shape)
with timer('To cat'):
    trainp['image_top_1'] = trainp['image_top_1'].astype('str').fillna('missing')
    testp['image_top_1'] = testp['image_top_1'].astype('str').fillna('missing') # My pet theory is that image_top_1 is categorical. Fight me.
    cat_cols = ['region', 'city', 'parent_category_name', 'category_name',
                'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1', 'wday']
    for col in trainp.columns:
        print(col)
        if col in cat_cols:
            trainp[col] = trainp[col].astype('category')
            testp[col] = testp[col].astype('category')
        else:
            trainp[col] = trainp[col].astype(np.float64)
            testp[col] = testp[col].astype(np.float64)
print(trainp.shape)
print(trainp.columns)
trainp.head()
trainp.dtypes
with timer('Split'):
    train, valid, y_train, y_valid = train_test_split(trainp, target, test_size=0.05, shuffle=True, random_state=37)
    test = testp
    print(train.shape)
    print(valid.shape)
    print(test.shape)
with timer('Submodels'):
    train_models = pd.DataFrame(np.array(preds_oofs).transpose())
    valid_models = pd.DataFrame(np.array(preds_valids).transpose())
    test_models = pd.DataFrame(np.array(preds_tests).transpose())
    train_models.columns = ['nn_' + str(i + 1) for i in range(train_models.shape[1])]
    valid_models.columns = ['nn_' + str(i + 1) for i in range(train_models.shape[1])]
    test_models.columns = ['nn_' + str(i + 1) for i in range(train_models.shape[1])]
    print(train_models.shape)
    print(valid_models.shape)
    print(test_models.shape)
with timer('Concat'):
    print(train.shape)
    X_train = pd.concat([train.reset_index(), train_models.reset_index()], axis=1)
    print(X_train.shape)
    print('-')
    print(valid.shape)
    X_valid = pd.concat([valid.reset_index(), valid_models.reset_index()], axis=1)
    print(X_valid.shape)
    print('-')
    print(test.shape)
    X_test = pd.concat([test.reset_index(), test_models.reset_index()], axis=1)
    print(X_test.shape)

X_train.head()
X_train.drop('index', axis=1, inplace=True)
X_valid.drop('index', axis=1, inplace=True)
X_test.drop('index', axis=1, inplace=True)
print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)
#del X_train_1
#del X_train_2
del trainp
del testp
gc.collect()
from pprint import pprint
import lightgbm as lgb

d_train = lgb.Dataset(X_train, label=y_train)
d_valid = lgb.Dataset(X_valid, label=y_valid)
watchlist = [d_train, d_valid]
params = {'application': 'regression',
          'metric': 'rmse',
          'nthread': 3,
          'verbosity': -1,
          'data_random_seed': 3,
          'learning_rate': 0.05,
          'num_leaves': 31,
          'bagging_fraction': 0.8,
          'feature_fraction': 0.2,
          'lambda_l1': 3,
          'lambda_l2': 3,
          'min_data_in_leaf': 40}
model = lgb.train(params,
                  train_set=d_train,
                  num_boost_round=1500,
                  valid_sets=watchlist,
                  verbose_eval=100)
pprint(sorted(list(zip(model.feature_importance(), X_train.columns)), reverse=True))
print('Done')
valid_preds = model.predict(X_valid).clip(0, 1)
print('Overall Valid RMSE: {:.4f}'.format(np.sqrt(mean_squared_error(y_valid, valid_preds))))
test_preds = model.predict(X_test).clip(0, 1)
submission = pd.read_csv('../input/test.csv', usecols=["item_id"])
submission["deal_probability"] = test_preds
submission.to_csv("submit_boosting_mlp.csv", index=False, float_format="%.2g")
submission.head()