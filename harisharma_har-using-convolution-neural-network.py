# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/A_DeviceMotion_data/A_DeviceMotion_data"))

# Any results you write to the current directory are saved as output.
folders = glob('../input/A_DeviceMotion_data/A_DeviceMotion_data/*_*')
folders = [s for s in folders if "csv" not in s]
df_all_list = []
activity_codes = {'dws':0,'jog':1,'sit':2,'std':3,'ups':4,'wlk':5}
activity_types = list(activity_codes.keys())

for j in folders:
    #print('j',j)
    csv = glob(j + '/*')
    for i in csv:
        df = pd.read_csv(i)
        df['activity'] = activity_codes[j[49:52]]
        df['sub_num'] = i[len(j)+5:-4]
        expnum = np.zeros(df.shape[0])
        df_all_list.append(df)
df_all = pd.concat(df_all_list,axis=0)
df_all = df_all.drop('Unnamed: 0',axis=1)
print(df_all.shape)
print(df_all.columns)
for act in activity_types:
    plt.subplot('61'+str(activity_codes[act]))
    plt.subplots_adjust(hspace=1.0)
    df = df_all[(df_all['sub_num']=='1') & (df_all['activity']==activity_codes[act])]
    plt.title(act)
    plt.plot(df['userAcceleration.z'][:400])
    plt.xticks([]) # turn off x labels
    plt.yticks([])  # turn off y labels
segment_size = 400
data_all_x_list = []
data_all_y_list = []
for j in folders:
    #print('j',j)
    csv = glob(j + '/*')
    for i in csv:
        df = pd.read_csv(i)
        df = df.drop('Unnamed: 0',axis=1)
        win_count = int(df.shape[0]/segment_size)
        data_x = np.zeros((win_count,segment_size,df.shape[1]))
        data_y = np.zeros(win_count)
        for c in range(win_count):
            start_idx = c*segment_size
            end_idx = start_idx + segment_size
            data_x[c,:,:] = df[start_idx:end_idx].values
            data_y[:] = activity_codes[j[49:52]]
        data_all_x_list.append(data_x)
        data_all_y_list.append(data_y)
data_all_x = np.concatenate(data_all_x_list,axis=0)
data_all_y = np.concatenate(data_all_y_list,axis=0)
data_all_y = data_all_y.astype(int)
print(data_all_x.shape)
print(data_all_y.shape)
def cnn_model_fn(features,labels,mode):
    conv1 = tf.layers.conv1d(inputs=features,
                             filters=32,
                             kernel_size=5,
                             padding='same',
                             data_format='channels_last',
                             activation=tf.nn.relu)
    print('conv1.shape',conv1.shape)
    pool1 = tf.layers.max_pooling1d(inputs=conv1,pool_size=2,strides=2)
    print('pool1.shape',pool1.shape)
    
    conv2 = tf.layers.conv1d(inputs=pool1,
                             filters=64,
                             kernel_size=5,
                             padding='same',
                             data_format='channels_last',
                             activation=tf.nn.relu)
    print('conv2.shape',conv2.shape)
    pool2 = tf.layers.max_pooling1d(inputs=conv2,pool_size=2,strides=2)
    print('pool2.shape',pool2.shape)
    
    pool2_flat = tf.reshape(pool2,[-1,100*64])  
    dense1 = tf.layers.dense(inputs=pool2_flat,units=500,activation=tf.nn.relu)
    
    dropput = tf.layers.dropout(inputs=dense1,rate=0.1,training=(mode==tf.estimator.ModeKeys.TRAIN))
    
    dense2 =  tf.layers.dense(inputs=dropput,units=100,activation=tf.nn.relu)
    
    logits = tf.layers.dense(inputs=dense2,units=6)
    
    predictions = { 'classes':tf.arg_max(logits,dimension=1),
                 'probabilites': tf.nn.softmax(logits,name = 'softmax_tensor')}
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
    
    eval_metric_op = {'accuracy': tf.metrics.accuracy(labels=labels,predictions=predictions['classes'])}
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_op)
train_x,test_x,train_y,test_y = train_test_split(data_all_x,data_all_y,test_size=0.1)
har_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir='/tmp/har_classifer_model')
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = train_x,
    y = train_y,
    batch_size=10,
    num_epochs=None,
    shuffle=True)


har_classifier.train(
    input_fn=train_input_fn,
    steps=20000)
    #hooks=[logging_hook])
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = test_x,
    y = test_y,
    num_epochs=1,
    shuffle=False)

eval_results = har_classifier.evaluate(input_fn=test_input_fn)
print(eval_results)
