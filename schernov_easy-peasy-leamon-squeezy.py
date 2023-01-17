import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
import os
from contextlib import contextmanager
from operator import itemgetter
import time
import gc

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.feature_extraction.text import CountVectorizer as CntVec
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('[{}] done in {:.2f} seconds'.format(name, time.time() - t0))

def preprocess(df):
    df['name'] = df['name'].fillna('') + ' ' + df['brand_name'].fillna('')
    df['text'] = (df['item_description'].fillna('') + ' ' + df['name'] + ' ' + df['category_name'].fillna(''))
    df['category_name'].fillna('unk', inplace=True)
    return df[['name', 'text', 'shipping', 'item_condition_id', 'category_name']]

def on_field(f, *vec):
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

def to_records(df):
    return df.to_dict(orient='records')

def fit_predict(xs, ys, X_final=None):
    X_train, X_test = xs
    y_train, y_test = ys
    regressor = Ridge(alpha=0.01)
    regressor.fit(X_train, y_train)
    y_hat = np.expm1(regressor.predict(X_test).reshape(-1, 1))[:,0]
    #y_hat_final = regressor.predict(X_final)
    print('Valid RMSLE: {:.4f}'.format(np.sqrt(mean_squared_log_error(y_test, y_hat))))
    #return y_hat_final
# test = pd.read_csv('../input/test_items.csv.zip', compression='zip')
# test.shape
train = pd.read_csv('../input/train_items.csv.zip', compression='zip')
train = train[train['price'] > 0].reset_index(drop=True)
train.shape
vectorizer = make_union(
        
        on_field('name', Tfidf(max_features=50000, 
                               token_pattern='\w+', 
                               ngram_range=(1, 1),
                               min_df=1)),
        
        on_field('text', Tfidf(max_features=50000, 
                               token_pattern='\w+', 
                               min_df=1, 
                               ngram_range=(1, 1))),
    
        on_field(['shipping', 'item_condition_id', 'category_name'], 
                 FunctionTransformer(to_records, validate=False), DictVectorizer()),
    
        n_jobs=1)

y_scaler = StandardScaler()

cv = KFold(n_splits=10, shuffle=True, random_state=42)
train_ids, valid_ids = next(cv.split(train))
train, valid = train.iloc[train_ids], train.iloc[valid_ids]

y_train = np.log1p(train['price'].values.reshape(-1, 1))
y_valid = valid['price'].values

X_train = vectorizer.fit_transform(preprocess(train)).astype(np.float32)
X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)
#X_final = vectorizer.transform(preprocess(test)).astype(np.float32)

print('X_train: {0} of {1}'.format(X_train.shape, X_train.dtype))
print('y_train: {0} of {1}'.format(y_train.shape, y_train.dtype))

print('X_valid: {0} of {1}'.format(X_valid.shape, X_valid.dtype))
print('y_valid: {0} of {1}'.format(y_valid.shape, y_valid.dtype))

#print('X_test: {0} of {1}'.format(X_final.shape, X_final.dtype))

del train
del valid

gc.collect()
#y_hat = 
fit_predict([X_train, X_valid], [y_train, y_valid])#, X_final)
#y_hat.shape
# subm = pd.read_csv('../input/sample_submission.csv.zip', compression='zip')
# subm['price'] = np.expm1(y_hat)
# subm.head()
