import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/train.csv', index_col='id')
targets = train.loc[:, 'target'].values

texts = [text for doc_id, text in train.loc[:, 'comment_text'].iteritems()]
del train
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer



count_vectorizer = CountVectorizer(ngram_range=(1,2), min_df=50, max_df=0.2)

count_vectorizer.fit(texts)

vectorized_texts = 1.0 * count_vectorizer.transform(texts)  # 1.0 since LGBM wants floats
from sklearn.model_selection import train_test_split



X_train, X_valid, Y_train, Y_valid = train_test_split(

    vectorized_texts, targets, test_size=0.2, random_state=80745, shuffle=True)
import lightgbm as lgb

train_data = lgb.Dataset(X_train, Y_train)

valid_data = lgb.Dataset(X_valid, Y_valid, reference=train_data)



param = {

    'num_leaves':31,

    'num_trees':150,

    'objective':'cross_entropy',

    'metric': ['auc']

}
bdt = lgb.train(param, train_data, 100, valid_sets=[valid_data])
test_data = pd.read_csv('../input/test.csv', index_col=0)
test_texts = [text for doc_id, text in test_data.loc[:, 'comment_text'].iteritems()]

test_vectorized_texts = 1.0 * count_vectorizer.transform(test_texts)

predictions = bdt.predict(test_vectorized_texts)

test_data['prediction'] = predictions

final_result = test_data[['prediction']].to_csv('lightgbm_primer_submission.csv')