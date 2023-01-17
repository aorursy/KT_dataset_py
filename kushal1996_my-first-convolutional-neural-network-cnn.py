import time

import tensorflow as tf

from tensorflow import keras

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from __future__ import absolute_import, division, print_function

tf.logging.set_verbosity(tf.logging.INFO)



import os

print(os.listdir("../input"))
train_img = np.load('../input/kmnist-train-imgs.npz')['arr_0']

test_img= np.load('../input/kmnist-test-imgs.npz')['arr_0']

train_label = np.load('../input/kmnist-train-labels.npz')['arr_0']

test_label = np.load('../input/kmnist-test-labels.npz')['arr_0']
char_df = pd.read_csv(r'../input/kmnist_classmap.csv')

char_df
print('train images shape {}\ntest images shape {}'.format(train_img.shape , 

                                                            test_img.shape))
print('train label shape {}\ntest label shape {}'.format(train_label.shape , 

                                                        test_label.shape))
train_label[:5]
plt.figure(1 , figsize = (15 , 9))

n = 0

for i in range(49):

    n += 1

    plt.subplot(7 , 7 , n)

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

    plt.imshow(train_img[i] , cmap = 'gray')

    plt.xticks([]) , plt.yticks([])

    plt.xlabel('class {}'.format(train_label[i]))

    

plt.show()
plt.figure(1 , figsize = (15 , 9))

n = 0

for i in range(49):

    n += 1

    plt.subplot(7 , 7 , n)

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

    plt.imshow(test_img[i] , cmap = 'gray')

    plt.xticks([]) , plt.yticks([])

    plt.xlabel('class {}'.format(test_label[i]))

    

plt.show()
train_img = train_img.astype(np.float32)

test_img = test_img.astype(np.float32)

train_label = train_label.astype(np.int32)

test_label = test_label.astype(np.int32)
train_img = train_img/255

test_img = test_img/255
train_X = train_img.reshape([-1 , 28 , 28 , 1])

test_X = test_img.reshape([-1 , 28 , 28 , 1])

train_X      = np.pad(train_X, ((0,0),(2,2),(2,2),(0,0)), 'constant')

test_X = np.pad(test_X, ((0,0),(2,2),(2,2),(0,0)), 'constant')
tf.reset_default_graph()

def cnn_model_fn(features , labels , mode ):

    input_layer = tf.reshape(features['x'] , [ -1 , 32 , 32 , 1])

    

    conv1 = tf.layers.conv2d(

        inputs = input_layer , 

        filters = 6 , 

        kernel_size = [5 , 5],

        padding = 'valid',

        activation = tf.nn.tanh

        )

    pool1 = tf.layers.average_pooling2d(inputs = conv1 , 

                                        pool_size = [2 , 2] 

                                        , strides = 2 )



    conv2 = tf.layers.conv2d(

        inputs = pool1,

        filters = 16 , 

        kernel_size = [5 , 5] ,

        padding  = 'valid',

        activation = tf.nn.tanh

        )

    

    pool2 = tf.layers.average_pooling2d(inputs = conv2 , 

                                        pool_size = [2 , 2] 

                                        , strides = 2 )



    conv3 = tf.layers.conv2d(

        inputs = pool2 , 

        filters = 120 , 

        kernel_size = [5 , 5],

        padding = 'valid',

        activation = tf.nn.tanh

        )

    conv3_flat = tf.layers.flatten(conv3)

    dense = tf.layers.dense(

        inputs = conv3_flat , 

        units = 84 , 

        activation = tf.nn.tanh

        )

    logits = tf.layers.dense(

        inputs = dense,

        units = 10 

        ) 

    

    predictions = {'classes' : tf.argmax(input = logits , axis = 1 ),

                  'probabilities' : tf.nn.softmax(logits , name = 'softmax_tensor')}

    

    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(mode = mode , 

                                          predictions = predictions)

    

    #Calculate Loss

    loss = tf.losses.sparse_softmax_cross_entropy(labels = labels , logits = logits )

    

    if mode == tf.estimator.ModeKeys.TRAIN:

        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)

        train_op = optimizer.minimize(

            loss = loss , 

            global_step = tf.train.get_global_step()

            )

        return tf.estimator.EstimatorSpec(mode = mode , loss = loss , train_op = train_op)

    

    eval_metric_ops = {

        'accuracy' : tf.metrics.accuracy(labels = labels ,

                                         predictions =  predictions['classes'])

    }

    

    return tf.estimator.EstimatorSpec(mode = mode , loss = loss , eval_metric_ops = eval_metric_ops)
cnn_image_classifier = tf.estimator.Estimator(

    model_fn = cnn_model_fn , model_dir = '/tmp/model_checkpoints' 

    )
tensors_to_log = {'probabilities':'softmax_tensor'}

logging_hook = tf.train.LoggingTensorHook(

    tensors = tensors_to_log , every_n_iter = 50 

    )
train_input_fn = tf.estimator.inputs.numpy_input_fn(

    x = {'x':train_X},

    y = train_label , 

    batch_size = 100 ,

    num_epochs = None , 

    shuffle = True

    )



cnn_image_classifier.train(input_fn = train_input_fn,

                          steps = 1, 

                          hooks = [logging_hook])
cnn_image_classifier.train(input_fn = train_input_fn , steps = 10000)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(

    x = {'x' : test_X},

    y = test_label , 

    num_epochs = 1,

    shuffle = False

    )



eval_results = cnn_image_classifier.evaluate(input_fn = eval_input_fn)

print(eval_results)
pred_input_fn = tf.estimator.inputs.numpy_input_fn(

    x = {'x' : test_X},

    y = test_label,

    num_epochs = 1,

    shuffle = False

    )



y_pred = cnn_image_classifier.predict(input_fn = pred_input_fn)

classes = [p['classes'] for p in y_pred]
plt.figure(1 , figsize = (15  , 9 ))

n = 0 

for i in range(49):

    n += 1 

    plt.subplot(7 , 7 , n)

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

    r = np.random.randint(0 , 10000 , 1)[0]

    plt.imshow(test_img[r] , cmap = 'gray')

    plt.xticks([]) , plt.yticks([])

    plt.xlabel('True : {}\nPred : {} '.format(test_label[r] , classes[r]) )

    

plt.show()