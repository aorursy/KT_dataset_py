# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import LabelEncoder

x_train = pd.read_csv('../input/X_train.csv')

y_train = pd.read_csv('../input/y_train.csv')

LabelEncoder_x = LabelEncoder()

label = LabelEncoder_x.fit_transform(y_train.surface)
import scipy

def stft(x, fftsize=24, overlap_pct=.5):   

    hop = int(fftsize * (1 - overlap_pct))

    w = scipy.hanning(fftsize + 1)[:-1]    

    raw1 = np.array([20 *np.log(np.abs(np.fft.rfft(w * x[i:i + fftsize]))+1) for i in range(0, len(x) - fftsize, hop)])

    raw2 = np.array([np.angle(np.fft.rfft(w * x[i:i + fftsize])) for i in range(0, len(x) - fftsize, hop)])

    return [raw1[:, :(fftsize // 2)], raw2[:, :(fftsize // 2)]]
import cv2 as cv2

def features_extraction(df, columns):

    ids = df.series_id.unique()

    features1 = []

    features2 = []

    for ide in ids:

        id_features1 = []

        id_features2 = []

        for column in columns:

            img1, img2 = stft(df[df['series_id']==ide][column])

            id_features1.append(cv2.resize(img1,(12,12)))

            id_features2.append(cv2.resize(img2,(12,12)))

        features1.append(np.array(id_features1))

        features2.append(np.array(id_features2))

    return np.transpose(np.array(features1), (0, 3, 2, 1)), np.transpose(np.array(features2), (0, 3, 2, 1))
columns = ['orientation_X', 'orientation_Y','orientation_Z','orientation_W', 'angular_velocity_X','angular_velocity_Y','angular_velocity_Z', 'linear_acceleration_X', 'linear_acceleration_Y', 'linear_acceleration_Z']

features1, features2 = features_extraction(x_train, columns)
import tensorflow as tf

def cnn_model_fn(features, labels, mode):

    """Model function for CNN."""

    # Input Layer

    input_layer1 = tf.reshape(features["x1"], [-1, 12, 12, 10])

    input_layer2 = tf.reshape(features["x2"], [-1, 12, 12, 10])



    # Convolutional Layer #1

    conv11 = tf.layers.conv2d(

      inputs=input_layer1,

      filters=32,

      kernel_size=[3, 3],

      padding="same",

      activation=tf.nn.relu)

    

    conv12 = tf.layers.conv2d(

      inputs=input_layer2,

      filters=32,

      kernel_size=[3, 3],

      padding="same",

      activation=tf.nn.relu)



    # Pooling Layer #1

    pool11 = tf.layers.max_pooling2d(inputs=conv11, pool_size=[2, 2], strides=2)

    pool12 = tf.layers.max_pooling2d(inputs=conv12, pool_size=[2, 2], strides=2)



    # Convolutional Layer #2

    conv21 = tf.layers.conv2d(

      inputs=pool11,

      filters=64,

      kernel_size=[3, 3],

      padding="same",

      activation=tf.nn.relu)

    

    conv22 = tf.layers.conv2d(

      inputs=pool12,

      filters=64,

      kernel_size=[3, 3],

      padding="same",

      activation=tf.nn.relu)

    

    # Normalization

    batch_mean, batch_var = tf.nn.moments(conv21, list(range(len(conv21.get_shape()) - 1)))

    conv21 = tf.nn.batch_normalization(conv21, batch_mean, batch_var,

                                       offset=None, scale=None,

                                       variance_epsilon=1e-3)

    

    batch_mean, batch_var = tf.nn.moments(conv22, list(range(len(conv22.get_shape()) - 1)))

    conv22 = tf.nn.batch_normalization(conv22, batch_mean, batch_var,

                                       offset=None, scale=None,

                                       variance_epsilon=1e-3)



    conv2 = tf.concat([conv21, conv22], 3)

    

    # Convolutional Layer #3

    conv3 = tf.layers.conv2d(

      inputs=conv2,

      filters=64,

      kernel_size=[3, 3],

      padding="same",

      activation=tf.nn.relu)

    

    # Pooling Layer #2

    pool2 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)



    # Dense Layer

    pool2_flat = tf.reshape(pool2, [-1, 3 * 3 * 64 ])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(

      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)



    # Logits Layer

    logits = tf.layers.dense(inputs=dropout, units=9)



    # Compute predictions.

    predictions = {

    # Generate predictions (for PREDICT and EVAL mode)

    "features": dense

    }



    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)



    # Calculate Loss (for both TRAIN and EVAL modes): penalized for imbalanced data

    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits))



    # Configure the Training Op (for TRAIN mode)

    if mode == tf.estimator.ModeKeys.TRAIN:

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        train_op = optimizer.minimize(

        loss=loss,

        global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)

    eval_metric_ops = {

      "accuracy": tf.metrics.accuracy(

          labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(

      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
classifier = tf.estimator.Estimator(model_fn=cnn_model_fn)

train_input_fn = tf.estimator.inputs.numpy_input_fn(

    x={"x1": features1.astype("float32"), "x2": features2.astype("float32")},

    y=label,

    batch_size=1600,

    num_epochs=None,

    shuffle=True)

classifier.train(

    input_fn=train_input_fn,

    steps=2000)
pre_input_fn = tf.estimator.inputs.numpy_input_fn(

        x={"x1": features1.astype("float32"),"x2": features2.astype("float32")},

        num_epochs=1,

        shuffle=False)

predictions = classifier.predict(input_fn=pre_input_fn);

res = []

for pred_dict in predictions:

    pretrained = pred_dict['features']

    res.append(pretrained)
from sklearn import svm

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedShuffleSplit

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
clf_svc = svm.SVC(gamma='scale', decision_function_shape='ovo',class_weight="balanced")

clf_svc.fit(np.array(res), label)

scores = cross_val_score(clf_svc, np.array(res), label, cv=cv)

scores
from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(n_estimators=120, random_state=0,class_weight="balanced")

clf_rf.fit(np.array(res), label)

scores = cross_val_score(clf_rf, np.array(res), label, cv=cv)

scores
import lightgbm as lgb  

import pickle  

from sklearn.metrics import roc_auc_score  

from sklearn.model_selection import train_test_split  

  

X, val_X, y, val_y = train_test_split(np.array(res),label,test_size=0.05,random_state=1,stratify=label)  



lgb_train = lgb.Dataset(X, y)  

lgb_eval = lgb.Dataset(val_X, val_y, reference=lgb_train)  



params = {  

    'boosting_type': 'gbdt',  

    'objective': 'multiclass',  

    'num_class': 9,  

    'metric': 'multi_error',  

    'num_leaves': 60,  

    'min_data_in_leaf': 50,  

    'learning_rate': 0.01,  

    'feature_fraction': 0.8,  

    'bagging_fraction': 0.8,  

    'bagging_freq': 5,  

    'lambda_l1': 0.4,  

    'lambda_l2': 0.5,  

    'min_gain_to_split': 0.2,  

    'verbose': 5,  

    'is_unbalance': True  

}  

  

# train  

gbm = lgb.train(params,  

                lgb_train,  

                num_boost_round=3000,  

                valid_sets=lgb_eval,  

                early_stopping_rounds=500)  
x_test = pd.read_csv('../input/X_test.csv')

test_features1, test_features2 = features_extraction(x_test, columns)
pre_input_fn = tf.estimator.inputs.numpy_input_fn(

        x={"x1": test_features1.astype("float32"),"x2": test_features2.astype("float32")},

        num_epochs=1,

        shuffle=False)

predictions = classifier.predict(input_fn=pre_input_fn);

res = []

for pred_dict in predictions:

    pretrained = pred_dict['features']

    res.append(pretrained)

# SVM Prediction

res_svc = clf_svc.predict(np.array(res))

# RF Prediction

res_rf = clf_rf.predict(np.array(res))

# light gbm Prediction

preds = gbm.predict(np.array(res), num_iteration=gbm.best_iteration)

res_lgb = []

for pred in preds:  

    res_lgb.append(int(np.argmax(pred)))
res_svc = LabelEncoder_x.inverse_transform(res_svc)

res_rf = LabelEncoder_x.inverse_transform(res_rf)

res_lgb = LabelEncoder_x.inverse_transform(res_lgb)
import time

from datetime import datetime

ver = 'CNN_svm'

filename = 'subm_{}_{}_'.format(ver, datetime.now().strftime('%Y-%m-%d'))

pd.DataFrame({

    'series_id': x_test.series_id.unique(),

    'surface': res_svc

}).to_csv(filename+'1'+'.csv', index=False)

ver = 'CNN_rf'

filename = 'subm_{}_{}_'.format(ver, datetime.now().strftime('%Y-%m-%d'))

pd.DataFrame({

    'series_id': x_test.series_id.unique(),

    'surface': res_rf

}).to_csv(filename+'1'+'.csv', index=False)

ver = 'CNN_lgb'

filename = 'subm_{}_{}_'.format(ver, datetime.now().strftime('%Y-%m-%d'))

pd.DataFrame({

    'series_id': x_test.series_id.unique(),

    'surface': res_lgb

}).to_csv(filename+'1'+'.csv', index=False)