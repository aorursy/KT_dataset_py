import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

from pathlib import Path



# import image tools

from PIL import Image



# import charts

import plotly

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



#import machine learning framework

import tensorflow as tf

from sklearn.model_selection import train_test_split

# Training Parameters

learning_rate = 0.001

num_steps = 5000

batch_size = 256#128

display_step = 100#0

MaxW, MaxH = 50, 50

# Network Parameters

num_classes = 2 # MNIST total classes (0-9 digits)

dropout = 1 # Dropout, probability to keep units



# tf Graph input

X = tf.placeholder(tf.float32, [None, MaxH, MaxW, 3])

Y = tf.placeholder(tf.float32, [None, num_classes])

keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)





weights = {

    'wd1': tf.Variable(tf.random_normal([12*12*10, 128])),

    'out': tf.Variable(tf.random_normal([128, num_classes]))

}



biases = {

    'bd1': tf.Variable(tf.random_normal([128])),

    'out': tf.Variable(tf.random_normal([num_classes]))

}



os.listdir("../input/cell_images/cell_images/")

# 

InfectedDir = "../input/cell_images/cell_images/Parasitized/";

UninfectedDir = "../input/cell_images/cell_images/Uninfected/";

InfectedFileNames = [InfectedDir + filename for filename in os.listdir(InfectedDir)]

UninfectedFileNames = [UninfectedDir + filename for filename in os.listdir(UninfectedDir)]
metadata = []

classes = []

if not Path('./metadata.npy').exists() or not Path('./classes.npy').exists()  :

    for InfectedFileName in InfectedFileNames:

        if(InfectedFileName.endswith(".png") != True):

             continue

        metadata.append(np.asarray(Image.open(InfectedFileName).resize([50,50])))

        classes.append(np.eye(num_classes)[np.array([0])].reshape([-1]).tolist())



    for UninfectedFileName in UninfectedFileNames:

        if(UninfectedFileName.endswith(".png") != True):

             continue

        metadata.append(np.asarray(Image.open(UninfectedFileName).resize([50,50])))

        classes.append(np.eye(num_classes)[np.array([1])].reshape([-1]).tolist())



    metadata = np.array(metadata)

    classes = np.array(classes)

    np.save('metadata', metadata)

    np.save('classes', classes)



metadata = np.load('./metadata.npy')

classes = np.load('./classes.npy')


# Create some wrappers for simplicity

def conv2d(x, W, b, strides=1):

    # Conv2D wrapper, with bias and relu activation

    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')

    x = tf.nn.bias_add(x, b)

    return tf.nn.relu(x)





def maxpool2d(x, k=2):

    # MaxPool2D wrapper

    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],

                          padding='SAME')





# Create model

def conv_net(x, weights, biases, dropout):



    x = tf.reshape(x, shape=[-1, MaxW, MaxH, 3])

    

    # Convolution Layer

    conv1 = tf.layers.conv2d(

        inputs=x,

        filters=50,

        kernel_size=[7, 7],

        padding='same',

        activation=tf.nn.relu

    )



    conv2 = tf.layers.conv2d(

        inputs=conv1,

        filters=90,

        kernel_size=[3, 3],

        padding='valid',

        activation=tf.nn.relu

    )



    conv3 = tf.layers.conv2d(

        inputs=conv2,

        filters=10,

        kernel_size=[5, 5],

        padding='same',

        activation=tf.nn.relu

    )



    # Max Pooling (down-sampling)

    pool1= maxpool2d(conv3, k=2)

    

    conv4 = tf.layers.conv2d(

        inputs=pool1,

        filters=10,

        kernel_size=[3, 3],

        padding='same',

        activation=tf.nn.relu

    )

    

    # Max Pooling (down-sampling)

    pool2 = maxpool2d(conv4, k=2)



    # Fully connected layer

    # Reshape conv2 output to fit fully connected layer input

    fc1 = tf.reshape(pool2, [-1, weights['wd1'].get_shape().as_list()[0]])

    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])

    fc1 = tf.nn.relu(fc1)

    

    # Apply Dropout

    fc1 = tf.nn.dropout(fc1, rate= 1 - dropout)



    # Output, class prediction

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out

# Construct model

logits = conv_net(X, weights, biases, keep_prob)

print(logits)

prediction = tf.nn.softmax(logits)



# Define loss and optimizer



loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss_op)





# Evaluate model

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
import random

def generate_batch(X, Y, batch_size):

    batch_x, batch_y = [], []

    for _ in range(batch_size):

        index = random.randint(0, len(X) - 1)

        pa = X[index, :, :] / 255

        class_ = Y[index]

        batch_x.append(pa)

        batch_y.append(class_.tolist())

    return batch_x, batch_y

df = pd.DataFrame(columns=['loss', 'accuracy'])
TrainX , TestX , TrainY , TextY = train_test_split(metadata , classes ,

                                            test_size = 0.3 ,

                                            random_state = 111)
init = tf.global_variables_initializer()

with tf.device('/device:GPU:0'):

    with tf.Session() as sess:



        # Run the initializer

        sess.run(init)



        for step in range(1, num_steps+1):

            batch_x, batch_y = generate_batch(TrainX, TrainY, batch_size)

            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})

            batch_x, batch_y = generate_batch(TestX, TextY, batch_size)

            pre, loss, acc = sess.run([prediction, loss_op, accuracy], feed_dict={X: batch_x,

                                                                                  Y: batch_y,

                                                                                  keep_prob: 1.0})

            df.loc[df.shape[0]] = [loss, acc]

            if step % display_step == 0 or step == 1:

                print("Step " + str(step) + ", Loss= " + \

                      "{}".format(loss) + ",  Accuracy= " + \

                      "{:.3f}".format(acc)

                      )

                df.to_csv('status.csv', index=False)



        print("Optimization Finished!")

    
loss = go.Scatter(

    x = df.index,

    y = df.loss,

    mode = 'lines',

    name = 'lines'

)



iplot([loss])
accuracy = go.Scatter(

    x = df.index,

    y = df.accuracy,

    mode = 'lines',

    name = 'lines'

)



iplot([accuracy])
print("Accuracy: ", df.accuracy[-100:].mean())

print("Loss: ", df.loss[-100:].mean())