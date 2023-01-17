# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.feature_column as fc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head(5)
train.info()
#no nan value
#explorary data analysis
#define two function used to draw bar and hist 
def drawSimpleBar(key):
    '''
     expolore data with simple bar with proportion
    :param key: feature name in csv
    :return: None
    '''
    print('=========' + key + '=============')
    print('99% percentile value is {:.4f}'.format(np.percentile(train[key], 99)))
    print('mean value is {:.4f}'.format(np.mean(train[key])))
    counts = train[key].value_counts(normalize=True)
    plt.bar(counts.index, counts)
    plt.xticks(counts.index)
    plt.title('Distribution proportion of ' + key)
    plt.show()

def drawSimpleHist(key):
    '''
    expolore data with simple histogram
    :param key: feature name in csv
    :return: None
    '''
    print('=========' + key + '=============')
    print('Top5 commonest value for ' + key + ' is \n{}'
          .format(train[key].value_counts(normalize=True).head(5)))
    plt.hist(train[key], bins=50, rwidth=0.8, align='left', color='green')
    plt.title('Distribution histogram of ' + key)
    plt.show()
    

#we ignore matchid,groupid as no meaning

#assists
drawSimpleBar('assists')
#boosts
drawSimpleBar('boosts')
#damageDealt
drawSimpleHist('damageDealt')
#DBNOs
#Number of enemy players knocked
drawSimpleBar('DBNOs')
#headshotKills
drawSimpleBar('headshotKills')
#heals
drawSimpleBar('heals')
#killPlace
#Ranking in match of number of enemy players killed
drawSimpleHist('killPlace')
#killPoints
drawSimpleHist('killPoints')
#kills
drawSimpleBar('kills')
#killStreaks
drawSimpleBar('killStreaks')
#longestKill

#Longest distance between player and player killed at time of death.
#This may be misleading, as downing a player and driving away may lead to a large longestKill stat

drawSimpleHist('longestKill')
#revives
#Number of times this player revived teammates.
drawSimpleBar('revives')
#rideDistance
drawSimpleHist('rideDistance')
#roadKills
#Number of kills while in a vehicle
drawSimpleBar('roadKills')
#swimDistance TODO
drawSimpleHist('swimDistance')
#teamKills
#Number of times this player killed a teammate.
drawSimpleBar('teamKills')
#vehicleDestroys
drawSimpleBar('vehicleDestroys')
#walkDistance
drawSimpleHist('walkDistance')
# weaponsAcquired
drawSimpleBar('weaponsAcquired')
#winPoints
drawSimpleHist('winPoints')
#winPlacePerc
drawSimpleHist('winPlacePerc')
#draw correlation heatmap
corr = train.corr()
labels = train.keys()
fig, ax = plt.subplots(figsize=(10,10))#type: plt.Figure, plt.Axes
im = ax.imshow(corr)
ax.set_xticks(range(corr.shape[0]))
ax.set_yticks(range(corr.shape[1]))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
plt.setp(ax.get_xticklabels(), rotation=90, ha='right')
cbar = fig.colorbar(im,ax=ax)
plt.savefig('correlation heatmap')
plt.show()
train = train[train.keys()[3:]]
#randomnize
train = train.sample(frac=1.0)

#min-max normalize
train = (train - train.min()) / (train.max() - train.min())

#split train and validation set
split_index = int(len(train)*0.8)
train, val = train[:split_index], train[split_index:]

#get feature and label
train_x, train_y = train, train.pop('winPlacePerc')
val_x, val_y = val, val.pop('winPlacePerc') #type: pd.DataFrame,pd.Series
# test = test[test.keys()[3:]]

#construct input function
train_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=train_x, y=train_y, num_epochs=None, shuffle=True,
    batch_size=20000
)
train_eval_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=train_x, y=train_y, shuffle=False
)
val_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=val_x, y=val_y, shuffle=False
)
test = test[test.keys()[3:]]
test_input_fn= tf.estimator.inputs.pandas_input_fn(
    x=test, shuffle=False
)
feature_columns = [
    fc.numeric_column('assists'),
    fc.numeric_column('boosts'),
    fc.numeric_column('damageDealt'),
    fc.numeric_column('DBNOs'),
    fc.numeric_column('headshotKills'),
    fc.numeric_column('heals'),
    fc.numeric_column('killPlace'),
    fc.numeric_column('killPoints'),
    fc.numeric_column('kills'),
    fc.numeric_column('killStreaks'),
    fc.numeric_column('longestKill'),
    fc.numeric_column('maxPlace'),
    fc.numeric_column('numGroups'),
    fc.numeric_column('revives'),
    fc.numeric_column('rideDistance'),
    fc.numeric_column('roadKills'),
    fc.numeric_column('swimDistance'),
    fc.numeric_column('teamKills'),
    fc.numeric_column('vehicleDestroys'),
    fc.numeric_column('walkDistance'),
    fc.numeric_column('weaponsAcquired'),
    fc.numeric_column('winPoints'),
]
def putg_mode_fn(features, labels, mode):
    if labels is not None and labels.dtype == tf.float64:
        labels = tf.cast(labels, tf.float32)
    inputs = fc.input_layer(features, feature_columns)
    hidden1 = tf.layers.dense(
        inputs=inputs,
        units=128,
        activation=tf.nn.relu
    )
    drop1 = tf.layers.dropout(inputs=hidden1, rate=0.1)
    hidden2 = tf.layers.dense(
        inputs=drop1,
        units=64,
        activation=tf.nn.relu
    )
    drop2 = tf.layers.dropout(inputs=hidden2, rate=0.1)
    hidden3 = tf.layers.dense(
        inputs=drop2,
        units=32,
        activation=tf.nn.relu
    )
    drop3 = tf.layers.dropout(inputs=hidden3, rate=0.1)
    out = tf.layers.dense(
        inputs=drop3,
        units=1,
    )
    out = tf.reshape(out, (-1,))

    norm_out = (out - tf.reduce_min(out))/(tf.reduce_max(out) - tf.reduce_min(out))
    # norm_out = tf.reshape(tf.nn.sigmoid(out1), (-1,))
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={'predictions': norm_out})

    mse_loss = tf.losses.absolute_difference(labels, norm_out)
    mae_loss = tf.losses.absolute_difference(labels, norm_out)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.002)
        train_op = optimizer.minimize(mse_loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=mae_loss,
            train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=mae_loss,
            eval_metric_ops={'mae':tf.metrics.mean_absolute_error(labels, norm_out)})

pubg_estimator = tf.estimator.Estimator(
    model_fn=putg_mode_fn,
    model_dir='model/dnn'
)

periods = 1
steps_per_period = 1
for i in range(periods):
    pubg_estimator.train(
        input_fn=train_input_fn,
        steps=steps_per_period
    )
    train_val_results = pubg_estimator.evaluate(input_fn=train_eval_input_fn)
    val_results = pubg_estimator.evaluate(input_fn=val_input_fn)
    print('epoch {}/{} : final mae loss for train data : {}'.format(i+1, periods, np.mean(train_val_results['mae'])))
    print('epoch {}/{} : final mae loss for val   data : {}'.format(i+1, periods, np.mean(val_results['mae'])))

predict_result = pubg_estimator.predict(test_input_fn, predict_keys=['predictions'])

submission = pd.read_csv('../input/sample_submission.csv')
submission['winPlacePerc'] = pd.Series([result['predictions'] for result in predict_result])
submission.to_csv('submission.csv', index=False)