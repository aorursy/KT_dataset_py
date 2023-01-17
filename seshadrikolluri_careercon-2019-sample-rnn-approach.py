# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

import seaborn as sns

import random 

import tensorflow as tf

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



%matplotlib inline

# Any results you write to the current directory are saved as output.
df_X_train = pd.read_csv('../input/X_train.csv')

df_y_train = pd.read_csv('../input/y_train.csv')

df_X_test = pd.read_csv('../input/X_test.csv')
print(df_X_train.shape,df_X_test.shape)
# https://stackoverflow.com/questions/53033620/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr?rq=1

def quaternion_to_euler(x, y, z, w):

    import math

    t0 = +2.0 * (w * x + y * z)

    t1 = +1.0 - 2.0 * (x * x + y * y)

    X = math.atan2(t0, t1)



    t2 = +2.0 * (w * y - z * x)

    t2 = +1.0 if t2 > +1.0 else t2

    t2 = -1.0 if t2 < -1.0 else t2

    Y = math.asin(t2)



    t3 = +2.0 * (w * z + x * y)

    t4 = +1.0 - 2.0 * (y * y + z * z)

    Z = math.atan2(t3, t4)



    return X, Y, Z



def fe_step0 (actual):

    

    # https://www.mathworks.com/help/aeroblks/quaternionnorm.html

    # https://www.mathworks.com/help/aeroblks/quaternionmodulus.html

    # https://www.mathworks.com/help/aeroblks/quaternionnormalize.html

        

    actual['norm_quat'] = (actual['orientation_X']**2 + actual['orientation_Y']**2 + actual['orientation_Z']**2 + actual['orientation_W']**2)

    actual['mod_quat'] = (actual['norm_quat'])**0.5

    actual['norm_X'] = actual['orientation_X'] / actual['mod_quat']

    actual['norm_Y'] = actual['orientation_Y'] / actual['mod_quat']

    actual['norm_Z'] = actual['orientation_Z'] / actual['mod_quat']

    actual['norm_W'] = actual['orientation_W'] / actual['mod_quat']

    

    return actual



def fe_step1 (actual):

    """Quaternions to Euler Angles"""

    

    x, y, z, w = actual['norm_X'].tolist(), actual['norm_Y'].tolist(), actual['norm_Z'].tolist(), actual['norm_W'].tolist()

    nx, ny, nz = [], [], []

    for i in range(len(x)):

        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])

        nx.append(xx)

        ny.append(yy)

        nz.append(zz)

    

    actual['euler_x'] = nx

    actual['euler_y'] = ny

    actual['euler_z'] = nz

    return actual





print('Before Step 0:', df_X_train.shape,df_X_test.shape)



df_X_train = fe_step0(df_X_train)

df_X_test = fe_step0(df_X_test)



print('Before Step 1:',df_X_train.shape,df_X_test.shape)



df_X_train = fe_step1(df_X_train)

df_X_test = fe_step1(df_X_test)



print('Before Step 2:',df_X_train.shape,df_X_test.shape)

df_X_train.shape[1]
train_dataset_all = np.array(df_X_train.drop(columns=['row_id', 'series_id','measurement_number'])).reshape(-1,128,df_X_train.shape[1]-3)

train_labels_all = pd.get_dummies(df_y_train['surface'])

label_names = [col for col in train_labels_all.columns]

train_labels_all = np.array(train_labels_all)



train_dataset, val_dataset, train_labels, val_labels = train_test_split(train_dataset_all, train_labels_all, test_size=0.1, random_state=0)

test_dataset_all = np.array(df_X_test.drop(columns=['row_id', 'series_id','measurement_number'])).reshape(-1,128,df_X_train.shape[1]-3)
train_dataset.shape,val_dataset.shape
#http://ataspinar.com/2018/07/05/building-recurrent-neural-networks-in-tensorflow/

signal_length = 128

num_components = train_dataset.shape[2]

num_labels = 9



num_hidden = 128

learning_rate = 0.001

lambda_loss = 0.010

total_steps = 1000

display_step = 100

batch_size = 1000



tf.reset_default_graph()



def lstm_rnn_model(data, num_hidden, num_labels):

    splitted_data = tf.unstack(data, axis=1)

    

    cell = tf.keras.layers.LSTMCell(num_hidden)



    outputs, current_state = tf.nn.static_rnn(cell, splitted_data, dtype=tf.float32)

    output = outputs[-1]

    

    w_softmax = tf.Variable(tf.truncated_normal([num_hidden, num_labels]))

    b_softmax = tf.Variable(tf.random_normal([num_labels]))

    logit = tf.matmul(output, w_softmax) + b_softmax

    return logit



def bidirectional_rnn_model(data, num_hidden, num_labels):

    splitted_data = tf.unstack(data, axis=1)



    lstm_cell1 = tf.keras.layers.LSTMCell(num_hidden) #, forget_bias=1.0, state_is_tuple=True

    lstm_cell2 = tf.keras.layers.LSTMCell(num_hidden) # , forget_bias=1.0, state_is_tuple=True

    outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_cell1, lstm_cell2, splitted_data, dtype=tf.float32)

    #outputs, _, _ = keras.layers.Bidirectional(keras.layers.RNN(cell, unroll=True))

    output = outputs[-1]

    

    w_softmax = tf.Variable(tf.truncated_normal([num_hidden*2, num_labels]))

    b_softmax = tf.Variable(tf.random_normal([num_labels]))

    logit = tf.matmul(output, w_softmax) + b_softmax

    return logit



def accuracy(y_predicted, y):

    return (100.0 * np.sum(np.argmax(y_predicted, 1) == np.argmax(y, 1)) / y_predicted.shape[0])

 

####



#1) First we put the input data in a tensorflow friendly form.    

tf_dataset = tf.placeholder(tf.float32, shape=(None, signal_length, num_components))

tf_labels = tf.placeholder(tf.float32, shape = (None, num_labels))



#2) Then we choose the model to calculate the logits (predicted labels)

# We can choose from several models:

#logits = rnn_model(tf_dataset, num_hidden, num_labels)

#logits = lstm_rnn_model(tf_dataset, num_hidden, num_labels)

logits = bidirectional_rnn_model(tf_dataset, num_hidden, num_labels)

#logits = multi_rnn_model(tf_dataset, num_hidden, num_labels)

#logits = gru_rnn_model(tf_dataset, num_hidden, num_labels)



#3) Then we compute the softmax cross entropy between the logits and the (actual) labels

l2 = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_labels)) + l2



#4. 

# The optimizer is used to calculate the gradients of the loss function 

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)



# Predictions for the training, validation, and test data.

prediction = tf.nn.softmax(logits)



session = tf.Session()

 

tf.global_variables_initializer().run(session=session)





# Add ops to save and restore all the variables.

saver = tf.train.Saver()





print("\nInitialized")



for step in range(total_steps):

    #Since we are using stochastic gradient descent, we are selecting  small batches from the training dataset,

    #and training the convolutional neural network each time with a batch. 

    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

    batch_data = train_dataset[offset:(offset + batch_size), :, :]

    batch_labels = train_labels[offset:(offset + batch_size), :]



    feed_dict = {tf_dataset : batch_data, tf_labels : batch_labels}

    _, l, train_predictions = session.run([optimizer, loss, prediction], feed_dict=feed_dict)

    train_accuracy = accuracy(train_predictions, batch_labels)



    if step % display_step == 0:

        feed_dict = {tf_dataset : val_dataset, tf_labels : val_labels}

        #feed_dict = {tf_dataset : train_dataset, tf_labels : train_labels}

        _, val_predictions = session.run([loss, prediction], feed_dict=feed_dict)

        test_accuracy = accuracy(val_predictions, val_labels)

        #test_accuracy = accuracy(train_predictions, train_labels)

        message = "step {:04d} : loss is {:06.2f}, accuracy on training set {} %, accuracy on test set {:02.2f} %".format(step, l, train_accuracy, test_accuracy)

        print(message)

        

save_path = saver.save(session, "/tmp/model.ckpt")

print("Model saved in path: %s" % save_path)

    

feed_dict = {tf_dataset : test_dataset_all, tf_labels : val_labels}  

test_predictions = session.run(prediction, feed_dict=feed_dict)
submission = pd.read_csv("../input/sample_submission.csv")

submission['surface'] = pd.DataFrame(np.argmax(test_predictions, axis = 1))[0].apply(lambda x : label_names[x]).values.reshape(-1)

submission.to_csv("submission_birectional.csv", index = False)

submission.head()