# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_excel('../input/toy_sales.xlsx')
df.head()

df.set_index('customer', inplace=True)
T0 = datetime.date(2017, 1, 1)
df['days'] = (T0 - df['date']).dt.days
future = df[df['days']<0]
past = df[df['days']>=0]
past_group = past.groupby(by=['customer'])
x = past_group['days'].min()
x.name = 'recency'
x = pd.DataFrame(x)
x['frequency'] = past_group['days'].count()

future_group = future.groupby(by=['customer'])
y = future_group['days'].count()
y.name = 'y_frequency'
y = pd.DataFrame(y)
y['y_recency'] = future_group['days'].max()*(-1)

data=x.join(y)
data['y_recency'].fillna(value=366, inplace=True)
data['y_frequency'].fillna(value=0, inplace=True)
data['y_frequency'][data['y_frequency']>0] = 1
data.head()
import matplotlib.pyplot as plt
x0, y0 = data[data['y_frequency']==0]['recency'],data[data['y_frequency']==0]['frequency']
x1, y1 = data[data['y_frequency']>0]['recency'],data[data['y_frequency']>0]['frequency']
plt.scatter(x0, y0, label='No-buy', color='r')
plt.scatter(x1, y1, label='Buyer', color='b')

plt.xlabel('recency', fontsize=16)
plt.ylabel('frequency', fontsize=16)
plt.title('scatter plot - recency vs frequency',fontsize=20)
plt.legend(loc='upper right')
plt.show()
from sklearn.model_selection import train_test_split

x = data[['recency', 'frequency']]
y_class = data['y_frequency']
y_regress = data['y_recency']

x_train, x_test, yc_train, yc_test = train_test_split(x, y_class, test_size=0.25)
x_train, x_test, yr_train, yr_test = train_test_split(x, y_regress, test_size=0.25)
from sklearn import tree
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.naive_bayes import GaussianNB


predictions, scores = {}, {}
predictions['true_values'] = yc_test.values

classifiers = [
    {'name': 'tree', 'clf': tree.DecisionTreeClassifier()},
    {'name': 'forest', 'clf': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)},
    {'name': 'logistic', 'clf': linear_model.LogisticRegression()},
    {'name': 'kneighbors', 'clf': KNeighborsClassifier()},
    {'name': 'svc', 'clf':  SVC(gamma=2, C=1)},
    {'name': 'gaussian', 'clf': GaussianProcessClassifier(1.0 * RBF(1.0))},
    {'name': 'neural', 'clf':  MLPClassifier(alpha=1)},
    {'name': 'adaboost', 'clf':  AdaBoostClassifier()},
     ]

for c in classifiers:
    clf = c['clf']
    clf.fit(x_train, yc_train)
    predictions[c['name']] = clf.predict(x_test)
    scores[c['name']] = [clf.score(x_test, yc_test)]

pd.DataFrame(predictions)
pd.DataFrame(scores).T
predictions, scores = {}, {}
predictions['true_values'] = yr_test.values

regressors = [
    {'name': 'tree', 'clf': tree.DecisionTreeRegressor()},
    {'name': 'gradientboost', 'clf': GradientBoostingRegressor(n_estimators=100, learning_rate=0.5, max_depth=1, random_state=0, loss='ls')},
    {'name': 'linear', 'clf': linear_model.LinearRegression()},
    {'name': 'kneighbors', 'clf': KNeighborsRegressor()},
    {'name': 'svr', 'clf':  SVR()},
    {'name': 'gaussian', 'clf': GaussianProcessRegressor(1.0 * RBF(1.0))},
    {'name': 'neural', 'clf':  MLPRegressor(alpha=1, max_iter=1000)},
    {'name': 'adaboost', 'clf':  AdaBoostRegressor()},
     ]

for r in regressors:
    clf = r['clf']
    clf.fit(x_train, yr_train)
    predictions[r['name']] = clf.predict(x_test)
    scores[r['name']] = [clf.score(x_test, yr_test)]
pd.DataFrame(predictions).T
pd.DataFrame(scores).T
import tensorflow as tf
def build_model_columns(cols):
    """Builds a set of wide and deep feature columns."""
    assert isinstance(cols, pd.Index)
    feats = list()
    for c in cols:
        feats.append(tf.feature_column.numeric_column(c))

    # return wide_columns, deep_columns, weight_column
    return feats, feats
def input_fn(features, labels, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""

    # Convert the inputs to a Dataset.
    ds = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    if shuffle:
        ds = ds.shuffle(1000)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    ds = ds.repeat(num_epochs)
    ds = ds.batch(batch_size)
    return ds
def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset
def build_estimator(cols):
    """Build an estimator appropriate for the given model type."""
    wide_columns, deep_columns = build_model_columns(cols)

    #hidden_units = [100, 75, 50, 25]
    a = len(wide_columns) #365
    hidden_units = [a*5, int(a/2), int(a/2/2), int(a/2/2/2)]

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto())

    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir='../input/',
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config,
        dnn_optimizer=tf.train.AdamOptimizer(),
        dnn_dropout=0.8)
clf = build_estimator(x_train.columns)


clf.train(input_fn=lambda: input_fn(x_train.reset_index().drop('customer', axis=1), yc_train, num_epochs=100, shuffle=True, batch_size=40))
# clf.get_variable_names(self)
# __dict__
# __weakref__
# config
# model_fn
# clf.train(input_fn=lambda: ds)
# (dict(x_train), yc_train)
x_train.reset_index().drop('customer', axis=1)
