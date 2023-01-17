# This Python 3 environment comes with many helpful analytics libraries installed

import scipy.misc

import random

from PIL import Image

import numpy as np

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import scipy

import os

from tensorflow.core.protobuf import saver_pb2

import pandas as pd
# driving data processing

xs = []

ys = []



#points to the end of the last batch

train_batch_pointer = 0

val_batch_pointer = 0



#read data.txt

with open("/kaggle/input/sullychenselfdrivingcar/driving_dataset/data.txt") as f:

    for line in f:

        xs.append("/kaggle/input/sullychenselfdrivingcar/driving_dataset/" + line.split()[0])

        #the paper by Nvidia uses the inverse of the turning radius,

        #but steering wheel angle is proportional to the inverse of turning radius

        #so the steering wheel angle in radians is used as the output

        ys.append(float(line.split()[1]) * scipy.pi / 180)



#get number of images

num_images = len(xs)





train_xs = xs[:int(len(xs) * 0.7)]

train_ys = ys[:int(len(xs) * 0.7)]



val_xs = xs[-int(len(xs) * 0.3):]

val_ys = ys[-int(len(xs) * 0.3):]



num_train_images = len(train_xs)

num_val_images = len(val_xs)

print("Number of training images",num_train_images)

print("Number of validation images",num_val_images)



def LoadTrainBatch(batch_size):

    global train_batch_pointer

    x_out = []

    y_out = []

    for i in range(0, batch_size):

        x_out.append(np.array(Image.fromarray(np.array(Image.open(train_xs[(train_batch_pointer + i) % num_train_images]))[-150:]).resize((200, 66)))/255.0)

        #x_out.append(scipy.misc.imresize(scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images])[-150:], [66, 200]) / 255.0)

        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])

    train_batch_pointer += batch_size

    return x_out, y_out



def LoadValBatch(batch_size):

    global val_batch_pointer

    x_out = []

    y_out = []

    for i in range(0, batch_size):

        x_out.append(np.array(Image.fromarray(np.array(Image.open(val_xs[(val_batch_pointer + i) % num_val_images]))[-150:]).resize((200, 66)))/255.0)

        #x_out.append(scipy.misc.imresize(scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images])[-150:], [66, 200]) / 255.0)

        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])

    val_batch_pointer += batch_size

    return x_out, y_out
# define model

def weight_variable(shape):

    initial = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(initial)



def bias_variable(shape):

    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial)



def conv2d(x, W, stride):

    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')



x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3])

y_ = tf.placeholder(tf.float32, shape=[None, 1])



x_image = x



#first convolutional layer

W_conv1 = weight_variable([5, 5, 3, 24])

b_conv1 = bias_variable([24])



h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 2) + b_conv1)



#second convolutional layer

W_conv2 = weight_variable([5, 5, 24, 36])

b_conv2 = bias_variable([36])



h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)



#third convolutional layer

W_conv3 = weight_variable([5, 5, 36, 48])

b_conv3 = bias_variable([48])



h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 2) + b_conv3)



#fourth convolutional layer

W_conv4 = weight_variable([3, 3, 48, 64])

b_conv4 = bias_variable([64])



h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, 1) + b_conv4)



#fifth convolutional layer

W_conv5 = weight_variable([3, 3, 64, 64])

b_conv5 = bias_variable([64])



h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, 1) + b_conv5)



#FCL 1

W_fc1 = weight_variable([1152, 1164])

b_fc1 = bias_variable([1164])



h_conv5_flat = tf.reshape(h_conv5, [-1, 1152])

h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)



keep_prob = tf.placeholder(tf.float32)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



#FCL 2

W_fc2 = weight_variable([1164, 100])

b_fc2 = bias_variable([100])



h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)



#FCL 3

W_fc3 = weight_variable([100, 50])

b_fc3 = bias_variable([50])



h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)



h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)



#FCL 3

W_fc4 = weight_variable([50, 10])

b_fc4 = bias_variable([10])



h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)



h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)



#Output

W_fc5 = weight_variable([10, 1])

b_fc5 = bias_variable([1])



y= tf.multiply((tf.matmul(h_fc4_drop, W_fc5) + b_fc5), 2) #scale the atan output
# train the model

LOGDIR = "kaggle/output/logs"



sess = tf.InteractiveSession()



L2NormConst = 0.001



train_vars = tf.trainable_variables()



loss = tf.reduce_mean(tf.square(tf.subtract(y_, y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess.run(tf.initialize_all_variables())



# create a summary to monitor cost tensor

tf.summary.scalar("loss", loss)

# merge all summaries into a single op

merged_summary_op =  tf.summary.merge_all()



saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)



# op to write logs to Tensorboard

logs_path = '/output'

summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())



epochs = 30

batch_size = 100

checkpoint="model.ckpt"

# train over the dataset about 30 times

for epoch in range(epochs):

    for i in range(int(num_images/batch_size)):

        xs, ys = LoadTrainBatch(batch_size)

        train_step.run(feed_dict={x: xs, y_: ys, keep_prob: 0.5})

        if i % 10 == 0:

            xs, ys = LoadValBatch(batch_size)

            loss_value = loss.eval(feed_dict={x:xs, y_: ys, keep_prob: 1.0})

            print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))



        # write logs at every iteration

        summary = merged_summary_op.eval(feed_dict={x:xs, y_: ys, keep_prob: 1.0})

        summary_writer.add_summary(summary, epoch * num_images/batch_size + i)

        if i % batch_size == 0:

            filename = saver.save(sess,checkpoint)

        print("Model saved in file: %s" % filename)
# predict the degrees on validation dataset

degrees_predicted = []

smoothed_angle = []

for i in range(len(val_xs)):

    #full_image = scipy.misc.imread(val_xs[i], mode="RGB")

    #image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0

    image = np.array(Image.fromarray(np.array(Image.open(val_xs[i]))[-150:]).resize((200, 66)))/255.0

    degrees = y.eval(feed_dict={x: [image], keep_prob: 1.0})[0][0] * 180.0 / scipy.pi

    #degrees = sess.run(y,feed_dict={x: [image], keep_prob: 1.0})[0][0] * 180.0 / scipy.pi

    degrees_predicted.append(degrees)

    

data = pd.DataFrame({"degrees":degrees_predicted,"original":[val_ys[i]*180/scipy.pi for i in range(len(val_ys))]})

data.to_csv("PredictedSteeringAngle.csv",index=False)