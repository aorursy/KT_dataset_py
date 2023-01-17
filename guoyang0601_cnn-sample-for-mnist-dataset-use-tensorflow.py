import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf

from keras.utils import to_categorical





# Input data files are available in the "../input/" directory.

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.



# This is a bit of magic to make matplotlib figures appear inline in the notebook

# rather than in a new window.

%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'
from tensorflow.python.keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split



# Import Data

train = pd.read_csv("../input/train.csv")

test= pd.read_csv("../input/test.csv")

print("Train size:{}\nTest size:{}".format(train.shape, test.shape))



# Transform Train and Test into images\labels.

x_train = train.drop(['label'], axis=1).values.astype('float32') # all pixel values

y_train = train['label'].values.astype('int32') # only labels i.e targets digits

x_test = test.values.astype('float32')

#Reshape

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) / 255.0

y_train = y_train.reshape(y_train.shape[0], 1)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) / 255.0



x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=42)

y_train = to_categorical(y_train)

y_val = to_categorical(y_val)



print(x_train.shape)

print(y_train.shape)

print(x_val.shape)

print(y_val.shape)

print(x_test.shape)
INPUT_NODE = 784

OUTPUT_NODE = 10



IMAGE_SIZE = 28

NUM_CHANNELS = 1

NUM_LABELS = 10



CONV1_DEEP = 32

CONV1_SIZE = 5



CONV2_DEEP = 64

CONV2_SIZE = 5



FC_SIZE = 512



MINI_BATCH_COUNT = 100

LEARNING_RATE = 0.01

LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.0001

TRAINING_EPOCHS = 100

MOVING_AVERAGE_DECAY = 0.99
def inference(input_tensor, train, regularizer):

    with tf.variable_scope("layer", reuse=tf.AUTO_REUSE):

        conv1_weights = tf.get_variable(

            "weight1", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],

            initializer=tf.truncated_normal_initializer(stddev=0.1))

        conv2_weights = tf.get_variable(

            "weight2", [CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],

            initializer = tf.truncated_normal_initializer(stddev=0.1))

        fc1_weights = tf.get_variable(

            "weight3", [3136, FC_SIZE],

            initializer=tf.truncated_normal_initializer(stddev=0.1))

        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))

        fc2_weights = tf.get_variable(

            "weight4", [FC_SIZE, NUM_LABELS],

            initializer=tf.truncated_normal_initializer(stddev=0.1))

        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))

    

        conv1_biases = tf.get_variable(

            "bias1", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        conv2_biases = tf.get_variable(

            "bias2", [CONV2_DEEP], initializer = tf.constant_initializer(0.0))

        fc1_biases = tf.get_variable(

            "bias3", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc2_biases = tf.get_variable(

            "bias4", [NUM_LABELS], initializer=tf.constant_initializer(0.1))

    

    

    

    conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1,1,1,1], padding='SAME')

    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1], strides = [1,2,2,1], padding="SAME")

    conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1,1,1,1], padding='SAME')

    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    pool2 = tf.contrib.layers.flatten(pool2)

    fc1 = tf.nn.relu(tf.matmul(pool2, fc1_weights) + fc1_biases)

    if train: fc1 = tf.nn.dropout(fc1, 0.5)

    logit = tf.matmul(fc1, fc2_weights) + fc2_biases



    return logit

        
sess = tf.Session()
with tf.variable_scope("layer", reuse=tf.AUTO_REUSE):

    x = tf.placeholder(tf.float32, [None,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS],name='x-input')

    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    x_val_ph = tf.placeholder(tf.float32, [None,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS],name='x-val')

    y_val_ph = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-val')

    x_test_ph = tf.placeholder(tf.float32, [None,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS])





    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    y = inference(x,True,regularizer)

    global_step = tf.Variable(0, trainable=False)

    

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    

    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        

    y_val_result = inference(x_val_ph,None,None)

    correct_prediction = tf.equal(tf.argmax(y_val_result, 1), tf.argmax(y_val_ph, 1))

    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



    y_test_result = inference(x_test_ph,None,None)
init_op = tf.global_variables_initializer()

sess.run(init_op)

batch_size = x_train.shape[0]/MINI_BATCH_COUNT

print(f"batch size is {batch_size}")

for epoch in range(TRAINING_EPOCHS):

    for i in range(MINI_BATCH_COUNT):

        x_train_batch = x_train[int(i*batch_size):int((i+1)*batch_size)]

        y_train_batch = y_train[int(i*batch_size):int((i+1)*batch_size)]

        _, loss_value = sess.run([train_step, loss], feed_dict={x: x_train_batch, y_: y_train_batch})



    validation_accuracy = sess.run(evaluation_step, feed_dict={x_val_ph: x_val, y_val_ph: y_val})

    print(f"After {epoch} training epoch(s), loss on training batch is {loss_value}, accuracy on test batch is {validation_accuracy}")
y_result = sess.run([y_test_result], feed_dict={x_test_ph: x_test})

test_result = sess.run(tf.nn.softmax(y_result[0]))
sess.close()
results = np.argmax(test_result, axis=1)

print(f"result len is {len(results)}")

print(f"First 5 result is:\n{results[0:100]}")
results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("MNIST-CNN-ENSEMBLE.csv",index=False)