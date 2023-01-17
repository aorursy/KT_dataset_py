import gc
import numpy as np 
import pandas_profiling
import pandas as pd 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import ShuffleSplit
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline, make_union
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf

import os
print(os.listdir("../input"))
def load_data(path):
    """
    Loads data and splits category_name column into 3 seperate columns.
    """
    def split_col_by_sep(x, na="NAN", sep='/'):
        if x is None:
            x = str()
        else:
            x = str(x)

        x_arr = x.split(sep)
        x_arr_len = len(x_arr)

        if x_arr_len == 3:
            return x_arr
        elif x_arr_len == 2:
            return [x_arr[0], x_arr[1], na]
        elif x_arr_len == 1:
            return [x_arr[0], na, na]
        else:
            return [na, na, na]
    
    df = pd.read_csv(path, sep='\t')
    df = df.replace(np.nan, 'NAN', regex=True)
    oj = []
    for row in df['category_name'].map(lambda x: split_col_by_sep(x, sep='/')):
        oj.append(row)
    oj = np.array(oj)
    
    df['category_name_1'] = pd.Series(oj[:,0])
    df['category_name_2'] = pd.Series(oj[:,1])
    df['category_name_3'] = pd.Series(oj[:,2])
    
    return df
%%time
train = load_data('../input/train.tsv')
train.drop(["train_id"], axis=1, inplace=True)
pandas_profiling.ProfileReport(train)
class LeaveOnlyMostPopularValuesTransformer(BaseEstimator, TransformerMixin):
    """
    """
    def __init__(self, no_of_most_popular=100, other_cat = "OTHER"):
        self.no_of_most_popular = no_of_most_popular
        self.other_cat = other_cat

    def fit(self, x, y = None):
        self.popular_cats = dict()
        for col in x.columns:
            self.popular_cats[col] = set(x[col].value_counts().head(self.no_of_most_popular).index.values)
        return self

    def transform(self, x):
        for col in x.columns:
            x.loc[:, col] = x.loc[:, col].map(lambda x: x if x in self.popular_cats[col] else self.other_cat)
        return x

    
class ContainsPopularWordTransformer(BaseEstimator, TransformerMixin):
    """
    """
    def __init__(self, no_of_most_popular_words=100):
        self.no_of_most_popular_words = no_of_most_popular_words

    def fit(self, x, y = None):
        self.most_popular_words = dict()
        for col in x:
            c = Counter()
            for line in x[col].str.lower().str.split():
                c.update(line)
            
            self.most_popular_words[col] = [w[0] for w in c.most_common(self.no_of_most_popular_words)]
        return self

    def transform(self, x):
        ret = pd.DataFrame()
        for col in x:
            for word in self.most_popular_words[col]:
                ret.loc[:, col + '_' + word] = x.loc[:, col].map(lambda a: 1 if word in a.split() else 0)
        return ret
    
    
class SelectDFColumn(BaseEstimator, TransformerMixin):
    """
    """
    def __init__(self, colnames):
        self.colnames = colnames

    def fit(self, x, y = None):
        return self

    def transform(self, x):
        if type(self.colnames) is str:
            return x.loc[:, self.colnames].values.reshape(-1, 1)
        else:
            return x.loc[:, self.colnames]
preprocessing_pipeline = make_pipeline(
    make_union(
        make_pipeline(
            SelectDFColumn(['item_condition_id', 'shipping', 'category_name_1', 'category_name_2', 'category_name_3','brand_name']),
            OneHotEncoder(handle_unknown='ignore')
        ),
        make_pipeline(
            SelectDFColumn(['name', 'item_description']),
            LeaveOnlyMostPopularValuesTransformer(1000),
            OneHotEncoder(handle_unknown='ignore')
        ),
        make_pipeline(
            SelectDFColumn(['name', 'item_description']),
            ContainsPopularWordTransformer(20)
        ),
    )
)
%%time
X_train = preprocessing_pipeline.fit_transform(train)
print(X_train.shape)
Y_train = train[['price']].values.reshape([-1,])
cv = ShuffleSplit(n_splits=3, test_size=0.1, random_state=42)
def rmsle(y_true, y_pred):
    return np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))

def tf_rmsle(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.pow(tf.math.log1p(y_pred) - tf.math.log1p(y_true), 2)))

def baseline_model():
    model = Sequential()
    model.add(InputLayer(input_shape=[X_train.shape[1],], sparse=True))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(loss=tf_rmsle, optimizer='adam', metrics=[tf_rmsle])
    return model
%%time
estimator = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=2056, verbose=1)
score = cross_val_score(estimator, X_train, Y_train, scoring=make_scorer(rmsle), cv=cv)
print(np.mean(score))
%%time
model = baseline_model()
ch = model.fit(X_train, Y_train, epochs=15, batch_size=2056)
%%time
test = load_data('../input/test_stg2.tsv')
test.drop('test_id', axis=1, inplace=True)
X_test = preprocessing_pipeline.transform(test)
predicts = model.predict(X_test)
submission = pd.read_csv('../input/sample_submission_stg2.csv')
submission.price = pd.Series(predicts.reshape([-1,]))
submission.to_csv("submission.csv", index=False)
print('DONE!')
