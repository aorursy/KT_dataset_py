# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from scipy.misc import imresize

import os

import glob

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from os.path import isfile, join, abspath, exists, isdir, expanduser

from PIL import Image

from pathlib import Path

from skimage.transform import resize

from sklearn.model_selection import train_test_split

from keras import backend as K

from PIL import Image

import tensorflow as tf

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
BASE_INPUT_FOLDER_PATH = Path("../input")

df = pd.read_csv(BASE_INPUT_FOLDER_PATH / "boneage-training-dataset.csv", index_col=0, header=0)

train_image_paths = list((BASE_INPUT_FOLDER_PATH / "boneage-training-dataset" / "boneage-training-dataset").glob("*.png"))



image_paths_df = pd.DataFrame()

image_paths_df["id"] = np.array([x.stem for x in train_image_paths]).astype(int)

image_paths_df["path"] = [str(x) for x in train_image_paths]

image_paths_df.set_index("id", inplace=True)



df = df.merge(image_paths_df, left_index=True, right_index=True)

df.head()
size = 124

X_train = []

y_train = []

i = 0 

for index, row in df.iterrows():

    if i%1000 == 0:

        print('Loading %d/12600'%i)

    path = row['path']

    orig = Image.open(path)

    img_arr = np.array(orig)

    resized = imresize(img_arr, (size, size))

    X_train.append(resized)

    if row['male'] == False:

        y_train.append(0)

    else:

        y_train.append(1)

    i+=1

print('Done loading')
plt.imshow(X_train[0])
X_train, y_train = np.stack(X_train, axis=0), np.stack(y_train, axis=0)

print(X_train.shape)

print(y_train.shape)
from sklearn import model_selection

X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=.2)
X_train = X_train[..., None]

X_val = X_val[...,None]
print('Train data shape: ', X_train.shape)

print('Train labels shape: ', y_train.shape)

print('Validation data shape: ', X_val.shape)

print('Validation labels shape: ', y_val.shape)
def run_model(session, predict, loss_val, Xd, yd,

              epochs=1, batch_size=64, print_every=100,

              training=None, plot_losses=False, print_acc=False):

    # have tensorflow compute accuracy

    correct_prediction = tf.equal(tf.argmax(predict,1), y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    

    # shuffle indicies

    train_indicies = np.arange(Xd.shape[0])

    np.random.shuffle(train_indicies)



    training_now = training is not None

    

    # setting up variables we want to compute (and optimizing)

    # if we have a training function, add that to things we compute

    variables = [mean_loss,correct_prediction,accuracy]

    if training_now:

        variables[-1] = training

    # counter 

    iter_cnt = 0

    for e in range(epochs):

        # keep track of losses and accuracy

        correct = 0

        losses = []

        # make sure we iterate over the dataset once

        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):

            # generate indicies for the batch

            start_idx = (i*batch_size)%Xd.shape[0]

            idx = train_indicies[start_idx:start_idx+batch_size]

            

            # create a feed dictionary for this batch

            feed_dict = {X: Xd[idx,:],

                         y: yd[idx],

                         is_training: training_now }

            # get batch size

            actual_batch_size = yd[idx].shape[0]

            

            # have tensorflow compute loss and correct predictions

            # and (if given) perform a training step

            loss, corr, _ = session.run(variables,feed_dict=feed_dict)

            

            # aggregate performance stats

            losses.append(loss*actual_batch_size)

            correct += np.sum(corr)

        

            iter_cnt += 1

        total_correct = correct/Xd.shape[0]

        total_loss = np.sum(losses)/Xd.shape[0]

        if print_acc:

            print('Validation accuracy: {0:.3g}. Overall loss: {1:.3g}'.format(total_correct, total_loss))



    return total_loss,total_correct
import math

L = [1e-5, 1e-3, 1e-2]

best_val_acc = -1

i = 1

for dropout_strength in [.4, .6, .8, .9]:

    for reg in L:

        for starter_learning_rate in L:

            def my_model(X,y,is_training):

                Wconv1 = tf.get_variable("Wconv1", shape=[2, 2, 1, 10])

                bconv1 = tf.get_variable("bconv1", shape=[10])

                betabatch1 = tf.get_variable("betabatch", shape = [10])

                gammabatch1 = tf.get_variable("gammabatch", shape = [10])



                Wconv2 = tf.get_variable("Wconv2", shape=[5, 5, 10, 60])

                bconv2 = tf.get_variable("bconv2", shape=[60])

                betabatch2 = tf.get_variable("betabatch2", shape = [60])

                gammabatch2 = tf.get_variable("gammabatch2", shape = [60])



                Wconv3 = tf.get_variable("Wconv3", shape=[5, 5, 60, 70])

                bconv3 = tf.get_variable("bconv3", shape=[70])



                Wconv4 = tf.get_variable("Wconv4", shape=[3, 3, 70, 80])

                bconv4 = tf.get_variable("bconv4", shape=[80])



                Wconv5 = tf.get_variable("Wconv5", shape=[3, 3, 80, 90])

                bconv5 = tf.get_variable("bconv5", shape=[90])





                W6 = tf.get_variable("W6", shape=[6*6*90, 800])

                b6 = tf.get_variable("b6", shape=[800])

                W7 = tf.get_variable("W7", shape=[800,900])

                b7 = tf.get_variable("b7", shape=[900])

                W8 = tf.get_variable("W8", shape=[900,2])

                b8 = tf.get_variable("b8", shape=[2])



                ### CONV1 ###



                conv1 = tf.nn.conv2d(X, Wconv1, strides=[1,1,1,1], padding='VALID') + bconv1

                relu1 = tf.nn.relu(conv1)

                relu1 = tf.nn.dropout(relu1, dropout_strength)

                mpool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID' )

                meanbatch1, variancebatch1 = tf.nn.moments(mpool1, axes = [0,1,2], keep_dims=True)

                norm1 = tf.nn.batch_normalization(mpool1, meanbatch1, variancebatch1, betabatch1, gammabatch1, 1e-5)



                ### CONV2 ###



                conv2 = tf.nn.conv2d(norm1, Wconv2, strides=[1,1,1,1], padding='SAME') + bconv2

                relu2 = tf.nn.relu(conv2)

                relu2 = tf.nn.dropout(relu2, dropout_strength)

                mpool2 = tf.nn.max_pool(relu2, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='VALID' )

                meanbatch2, variancebatch2 = tf.nn.moments(mpool2, axes = [0,1,2], keep_dims=True)

                norm2 = tf.nn.batch_normalization(mpool2, meanbatch2, variancebatch2, betabatch2, gammabatch2, 1e-5)



                ### CONV3 ###

                # Padding

                norm2 = tf.image.resize_image_with_crop_or_pad(

                norm2,

                29,

                29

            )

                conv3 = tf.nn.conv2d(norm2, Wconv3, strides=[1,2,2,1], padding='VALID') + bconv3

                conv4 = tf.nn.conv2d(conv3, Wconv4, strides=[1,1,1,1], padding='SAME') + bconv4

                conv5 = tf.nn.conv2d(conv4, Wconv5, strides=[1,1,1,1], padding='SAME') + bconv5

                mpool3 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID' )



                ### FC6, FC7, FC8 ###

                mpool3_flat = tf.reshape(mpool3,[-1,6*6*90])

                fc6 = tf.matmul(mpool3_flat,W6) + b6

                fc7 = tf.matmul(fc6,W7) + b7

                y_out = tf.matmul(fc7,W8) + b8

                regularizers = (tf.nn.l2_loss(Wconv1) + tf.nn.l2_loss(Wconv2) + tf.nn.l2_loss(Wconv3) +  tf.nn.l2_loss(Wconv4) +\

                            tf.nn.l2_loss(Wconv5) + tf.nn.l2_loss(W6) + tf.nn.l2_loss(W7) + tf.nn.l2_loss(W8))

                return y_out, regularizers

            # hyperparameters

            decay_steps = 1000

            decay_rate = 0.8

            

            tf.reset_default_graph()



            X = tf.placeholder(tf.float32, [None, 124, 124, 1])

            y = tf.placeholder(tf.int64, [None])

            is_training = tf.placeholder(tf.bool)



            y_out, regularizers = my_model(X,y,is_training)



            # loss function with L2 regularization

            total_loss = tf.losses.softmax_cross_entropy(tf.one_hot(y,2),logits=y_out)

            mean_loss = tf.reduce_mean(total_loss)

            mean_loss = tf.reduce_mean(mean_loss + reg * regularizers)



            # define our optimizer



            global_step = tf.Variable(0, trainable=False)

            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,

                                                       decay_steps, decay_rate, staircase=True)

            optimizer = tf.train.AdamOptimizer(learning_rate) # select optimizer and set learning rate



            # batch normalization in tensorflow requires this extra dependency

            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(extra_update_ops):

                train_step = optimizer.minimize(mean_loss)

            params = (dropout_strength, reg, starter_learning_rate, decay_steps, decay_rate)

#             print('HYPERPARAMETERS: ', (dropout_strength, reg, starter_learning_rate, decay_steps, decay_rate))

            sess = tf.Session()



            sess.run(tf.global_variables_initializer())

            print('Training and validating %d/36'%i)

            run_model(sess,y_out,mean_loss,X_train,y_train,1,64,100,train_step,False)

            total_loss,total_correct = run_model(sess,y_out,mean_loss,X_val,y_val,1,64,print_acc=True)

            if total_correct > best_val_acc:

                best_val_acc = total_correct

                best_params = params 

            i += 1

                

                
print('Best val accuracy:', best_val_acc)

print('Best params:', best_params)
# redefine run_model

def run_model(session, predict, loss_val, Xd, yd,

              epochs=1, batch_size=64, print_every=100,

              training=None, plot_losses=False):

    # have tensorflow compute accuracy

    correct_prediction = tf.equal(tf.argmax(predict,1), y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    

    # shuffle indicies

    train_indicies = np.arange(Xd.shape[0])

    np.random.shuffle(train_indicies)



    training_now = training is not None

    

    # setting up variables we want to compute (and optimizing)

    # if we have a training function, add that to things we compute

    variables = [mean_loss,correct_prediction,accuracy]

    if training_now:

        variables[-1] = training

    # counter 

    iter_cnt = 0

    for e in range(epochs):

        # keep track of losses and accuracy

        correct = 0

        losses = []

        # make sure we iterate over the dataset once

        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):

            # generate indicies for the batch

            start_idx = (i*batch_size)%Xd.shape[0]

            idx = train_indicies[start_idx:start_idx+batch_size]

            

            # create a feed dictionary for this batch

            feed_dict = {X: Xd[idx,:],

                         y: yd[idx],

                         is_training: training_now }

            # get batch size

            actual_batch_size = yd[idx].shape[0]

            

            # have tensorflow compute loss and correct predictions

            # and (if given) perform a training step

            loss, corr, _ = session.run(variables,feed_dict=feed_dict)

            

            # aggregate performance stats

            losses.append(loss*actual_batch_size)

            correct += np.sum(corr)

            

            # print every now and then

            if loss > 500: 

                return 

            if training_now and (iter_cnt % print_every) == 0:

                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\

                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))

            iter_cnt += 1

        total_correct = correct/Xd.shape[0]

        total_loss = np.sum(losses)/Xd.shape[0]

        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\

              .format(total_loss,total_correct,e+1))

        if plot_losses:

            plt.plot(losses)

            plt.grid(True)

            plt.title('Epoch {} Loss'.format(e+1))

            plt.xlabel('minibatch number')

            plt.ylabel('minibatch loss')

            plt.show()

    return total_loss,total_correct
# hyperparameters

dropout_strength, reg, starter_learning_rate, decay_steps, decay_rate = best_params

tf.reset_default_graph()



X = tf.placeholder(tf.float32, [None, 124, 124, 1])

y = tf.placeholder(tf.int64, [None])

is_training = tf.placeholder(tf.bool)



y_out, regularizers = my_model(X,y,is_training)



# loss function with L2 regularization

total_loss = tf.losses.softmax_cross_entropy(tf.one_hot(y,2),logits=y_out)

mean_loss = tf.reduce_mean(total_loss)

mean_loss = tf.reduce_mean(mean_loss + reg * regularizers)



# define our optimizer



global_step = tf.Variable(0, trainable=False)

learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,

                                           decay_steps, decay_rate, staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate) # select optimizer and set learning rate



# batch normalization in tensorflow requires this extra dependency

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(extra_update_ops):

    train_step = optimizer.minimize(mean_loss)



sess = tf.Session()



sess.run(tf.global_variables_initializer())

print('Training')

run_model(sess,y_out,mean_loss,X_train,y_train,20,64,200,train_step,True)

print('Validation')

run_model(sess,y_out,mean_loss,X_val,y_val,1,64)
BASE_INPUT_FOLDER_PATH = Path("../input")

df_test = pd.read_csv(BASE_INPUT_FOLDER_PATH / "boneage-test-dataset.csv", index_col=0, header=0)

test_image_paths = list((BASE_INPUT_FOLDER_PATH / "boneage-test-dataset" / "boneage-test-dataset").glob("*.png"))



image_paths_df = pd.DataFrame()

image_paths_df["id"] = np.array([x.stem for x in test_image_paths]).astype(int)

image_paths_df["path"] = [str(x) for x in test_image_paths]

image_paths_df.set_index("id", inplace=True)



df_test = df_test.merge(image_paths_df, left_index=True, right_index=True)
df_test.head()
size = 124

X_test = []

y_test = []

i = 0 

for index, row in df_test.iterrows():

    path = row['path']

    orig = Image.open(path)

    img_arr = np.array(orig)

    resized = imresize(img_arr, (size, size))

    X_test.append(resized)

    if row['Sex'] == 'M':

        y_test.append(1)

    else:

        y_test.append(0)

print('Done loading')
X_test, y_test = np.stack(X_test, axis=0), np.stack(y_test, axis=0)

print(X_test.shape)

print(y_test.shape)
X_test = X_test[..., None]

print('Test')

run_model(sess,y_out,mean_loss,X_test,y_test,1,64)