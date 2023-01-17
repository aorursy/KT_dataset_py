# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import xgboost as xgb
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
# Read data
df_train = pd.read_csv("../input/train.csv", index_col='ID')
feature_cols = list(df_train.columns)
feature_cols.remove("TARGET")
df_test = pd.read_csv("../input/test.csv", index_col='ID')

# Split up the data
X_all = df_train[feature_cols]
y_all = df_train["TARGET"]
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.3, random_state=5, stratify=y_all)
# Get top features from xgb model
model = xgb.XGBRegressor(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=9,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=5
)

# Train cv
xgb_param = model.get_xgb_params()
dtrain = xgb.DMatrix(X_train.values, label=y_train.values, missing=np.nan)
cv_result = xgb.cv(
    xgb_param,
    dtrain,
    num_boost_round=model.get_params()['n_estimators'],
    nfold=5,
    metrics=['auc'],
    early_stopping_rounds=50)
best_n_estimators = cv_result.shape[0]
model.set_params(n_estimators=best_n_estimators)

# Train model
model.fit(X_train, y_train, eval_metric='auc')

# Predict training data
y_hat_train = model.predict(X_train)

# Predict test data
y_hat_test = model.predict(X_test)

# Print model report:
print("\nModel Report")
print("best n_estimators: {}".format(best_n_estimators))
print("AUC Score (Train): %f" % roc_auc_score(y_train, y_hat_train))
print("AUC Score (Test) : %f" % roc_auc_score(y_test,  y_hat_test))
# Get important features
feat_imp = list(pd.Series(model.booster().get_fscore()).sort_values(ascending=False).index)
# Even out the targets
df_train_1 = df_train[df_train["TARGET"] == 1]
df_train_0 = df_train[df_train["TARGET"] == 0].head(df_train_1.shape[0])
df_train = df_train_1.append(df_train_0)

# Scale data
X_all = df_train[feat_imp].copy(deep=True)
y_all = df_train["TARGET"]
X_all[feat_imp] = sklearn.preprocessing.scale(X_all, axis=0, with_mean=True, with_std=True, copy=True)
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.3, random_state=5, stratify=y_all)

# Create second complementary column at position 0
y_train_2cols = np.array(list(zip((1 - y_train).values, y_train.values)))
# Tensorflow Model
import tensorflow as tf

# Hyperparameters
n_steps = 3001
batch_size = 200
learning_rate0 = 0.05
decay_steps = 500
decay_rate = 0.8

# Network parameters
n_h1 = 20
n_h2 = 20
n_features = X_train.shape[1]
n_labels = 2

# L2 regularization
beta = 1e-5

# Dropout
keep_prob = 0.5

graph = tf.Graph()
with graph.as_default():
    # Allocate variables
    tf_X_train = tf.placeholder(tf.float32, shape=(batch_size, n_features))
    tf_y_train = tf.placeholder(tf.float32, shape=(batch_size, n_labels))
    tf_X_test = tf.constant(X_test.values, dtype=tf.float32)
    
    tf_keep_prob = tf.placeholder(tf.float32)
    
    # Hidden layer
    with tf.name_scope('h1') as scope:
        weights_h1 = tf.Variable(
            tf.truncated_normal(
                [n_features, n_h1],
                stddev=1.0 / np.sqrt(n_features)
        ), name='weights_h1')
        biases_h1 = tf.Variable(tf.zeros([n_h1]), name='biases_h1')        
        h1 = tf.nn.relu(tf.matmul(tf_X_train, weights_h1) + biases_h1)

        # Dropout
        h1 = tf.nn.dropout(h1, tf_keep_prob)
        
    # Hidden layer 2
    with tf.name_scope('h2') as scope:
        weights_h2 = tf.Variable(
            tf.truncated_normal(
                [n_h1, n_h2],
                stddev=1.0 / np.sqrt(n_h1)
        ), name='weights_h2')
        biases_h2 = tf.Variable(tf.zeros([n_h2]), name='biases_h2')
        h2 = tf.nn.relu(tf.matmul(h1, weights_h2) + biases_h2)

        # Dropout
        h2 = tf.nn.dropout(h2, tf_keep_prob)
        
    # Output layer
    with tf.name_scope('softmax_linear'):
        weights_out = tf.Variable(
            tf.truncated_normal(
                [n_h2, n_labels],
                stddev=1.0 / np.sqrt(n_h2)
            ), name='weights_out')
        biases_out = tf.Variable(tf.zeros([n_labels]), name='biases_out')
        logits = tf.matmul(h2, weights_out) + biases_out

    # Training computation.
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_y_train))
    
    # L2 Regularization
    #loss += beta * (tf.nn.l2_loss(weights_h1) + tf.nn.l2_loss(weights_h2) + tf.nn.l2_loss(weights_out))

    # Optimizer
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        learning_rate0, global_step, decay_steps, decay_rate, staircase=False)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    # Predictions for the training and test datasets.
    yhat_train = tf.nn.softmax(logits)
    yhat_test = tf.nn.relu(tf.matmul(tf_X_test, weights_h1) + biases_h1)
    yhat_test = tf.nn.relu(tf.matmul(yhat_test, weights_h2) + biases_h2)
    yhat_test = tf.nn.softmax(tf.matmul(yhat_test, weights_out) + biases_out)
    
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    for step in range(n_steps):    
        _batch_idx = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
        batch_X = X_train.values[_batch_idx, :]
        batch_y = y_train_2cols[_batch_idx, :]

        # Prepare a dictionary telling the session where to feed the minibatch.
        feed_dict = {
            tf_X_train: batch_X,
            tf_y_train: batch_y,
            tf_keep_prob: keep_prob
        }
        _, l, pred = session.run([optimizer, loss, yhat_train], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Batch loss at step {0:d}: {1:.6f}".format(step, l))
            print("Batch score: {0:.6f}".format(roc_auc_score(batch_y[:, 1], pred[:, 1])))
            print("Test score: {0:.6f}".format(roc_auc_score(y_test.values, yhat_test.eval()[:, 1])))
    print("Final Test score: {0:.6f}".format(roc_auc_score(y_test.values, yhat_test.eval()[:, 1])))
