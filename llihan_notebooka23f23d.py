import numpy as np

import gc

import pandas as pd

import tensorflow as tf

from sklearn.feature_extraction import stop_words

from collections import Counter

from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

import logging

from scipy.sparse import csr_matrix, hstack
NUM_BRANDS = 4000

NUM_CATEGORIES = 1000

NAME_MIN_DF = 10

MAX_FEATURES_ITEM_DESCRIPTION = 50000
train = pd.read_csv('../input/train.tsv', sep='\t')

test = pd.read_csv('../input/test.tsv', sep='\t')



nrow_train = train.shape[0]

merged = pd.concat([train, test])
def print_null_values(train):

    # null brand_name

    print('{0}% of brand_name is null'.format(len(train[train['brand_name'].isnull()]) / len(train) * 100))

    # null category_name

    print('{0}% of category_name is null'.format(len(train[train['category_name'].isnull()]) / len(train) * 100))

    # null item_condition_id

    print('{0}% of item_condition_id is null'.format(len(train[train['item_condition_id'].isnull()]) / len(train) * 100))

    # null item_description

    print('{0}% of item_description is null'.format(len(train[train['item_description'].isnull()]) / len(train) * 100))

    # null name

    print('{0}% of name is null'.format(len(train[train['name'].isnull()]) / len(train) * 100))

    # null shipping

    print('{0}% of shipping is null'.format(len(train[train['shipping'].isnull()]) / len(train) * 100))

    # null price

    print('{0}% of price is null'.format(len(train[train['price'].isnull()]) / len(train) * 100))

    

#print_null_values(train)
# Treat null values

merged[merged['brand_name'].isnull()] = 'missing'

merged[merged['item_description'].isnull()] = 'missing'

merged.dropna(subset=['category_name'], inplace=True)
# Split category



def _safe_split(x):

    vals = x.split('/')

    while len(vals) < 3:

        vals.append('missing')

    return vals[0], vals[1], vals[2]

        

merged['cat_1'], merged['cat_2'], merged['cat_3'] = zip(*merged['category_name'].apply(_safe_split))
# Deal with description

tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,

                         ngram_range=(1, 3),

                         stop_words='english')

X_description = tv.fit_transform(merged['item_description'])
lb = LabelBinarizer(sparse_output=True)

X_brand = lb.fit_transform(merged['brand_name'])
cv = CountVectorizer(min_df=NAME_MIN_DF)

X_name = cv.fit_transform(merged['name'])
cv = CountVectorizer()

X_category1 = cv.fit_transform(merged['cat_1'])

X_category2 = cv.fit_transform(merged['cat_2'])

X_category3 = cv.fit_transform(merged['cat_3'])
X_dummies = csr_matrix(pd.get_dummies(merged[['item_condition_id', 'shipping']],

                                          sparse=True).values)
sparse_merge = hstack((

    X_dummies, 

    X_description, 

    X_brand, 

    X_category1, 

    X_category2, 

    X_category3, 

    X_name)).tocsr()
#gc.collect()
X = sparse_merge[:nrow_train]

X_test = sparse_merge[nrow_train:]

y = np.log1p(train["price"])
# Network Parameters

n_hidden_1 = 30

n_hidden_2 = 25

num_steps = 1

num_input = sparse_merge.shape[1]

learning_rate = 0.08

batch_size = 32
inputs = tf.placeholder(tf.float32, shape=[None, num_input])

labels = tf.placeholder(tf.float32, shape=[None, 1])

initializer = tf.contrib.layers.xavier_initializer()

fc = tf.layers.dense(inputs, n_hidden_1, activation=tf.nn.relu, kernel_initializer=initializer)

fc2 = tf.layers.dense(fc, n_hidden_2, activation=tf.nn.relu)

fc3 = tf.layers.dropout(fc2, rate=0.25)

normal_initializer = tf.initializers.random_normal()

fc_out = tf.layers.dense(fc3, 1, kernel_initializer=normal_initializer)

cost = tf.losses.mean_squared_error(labels, fc_out)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()



def batch_generator(X_data, y_data, counter=0, batch_size=32):

    index = np.arange(np.shape(y_data)[0])

    index_batch = index[batch_size*counter:batch_size*(counter+1)]

    X_batch = X_data[index_batch,:].todense()

    y_batch = y_data[index_batch]

    return np.array(X_batch), y_batch.reshape(batch_size, 1)

def batch_generator_test(X_data, counter=0, batch_size=32):

    index = np.arange(np.shape(X_data)[0])

    index_batch = index[batch_size*counter:batch_size*(counter+1)]

    X_batch = X_data[index_batch,:].todense()

    return np.array(X_batch)





sess = tf.Session()

sess.run(init)   

for step in range(num_steps):

    number_of_batches = int(X.shape[0] / batch_size)

    for i in range(number_of_batches):

        _inputs, _labels = batch_generator(X, y, counter=i, batch_size=batch_size)

        _, lost = sess.run([

            optimizer,

            cost

        ], feed_dict={

            inputs: _inputs,

            labels: _labels

        })

        if i % 200 == 0:

            print("step: {}, lost: {}".format(step, lost))
predictions = []

number_of_batches_submission = int(X_test.shape[0] / batch_size)

for i in range(number_of_batches_submission):

    _inputs_test = batch_generator_test(X_test, counter=i, batch_size=batch_size)

    _predictions = sess.run([fc_out], feed_dict={

        inputs: _inputs_test

    })

    predictions.extend(_predictions[0])



predictions = np.array(predictions).reshape(-1)

test_ids = test['test_id'].values.astype(np.int32)

submission = pd.DataFrame({

    'price': predictions

})

#submission['price'] = np.expm1(submission['price'])

submission.head()

    