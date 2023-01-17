import re

import numpy as np

import pandas as pd

from sklearn.feature_extraction import DictVectorizer
TRAIN_FILE = '../input/train.csv'

TEST_FILE = '../input/test.csv'
train_df = pd.read_csv(TRAIN_FILE)

test_df = pd.read_csv(TEST_FILE)

all_df = train_df.append(test_df)
# 1h-vectorize categories

vec = DictVectorizer()

g = all_df.copy()



for x in g:

    if re.match(r"cat\d+", x) is None:

         del g[x]



cats_1h = vec.fit_transform(g.to_dict('records')).toarray()

cats_1h = np.array(cats_1h, dtype=np.float32)

del g
# create vectors of cont- values

g = all_df.copy()



for x in g:

    if re.match(r"cont\d+", x) is None:

         del g[x]



conts = g.to_dict('split')

conts = np.array(conts['data'], dtype=np.float32)



del g
X = np.hstack((cats_1h, conts))
X_train = X[:len(train_df)]

X_test = X[len(train_df):]

y_train = np.array(train_df['loss'].values, dtype=np.float32)

ids_test = np.array(test_df['id'], dtype=np.int64)
print ('X_train', X_train.shape)

print ('y_train', y_train.shape)

print ('X_test', X_test.shape)

print ('ids_test', ids_test.shape)