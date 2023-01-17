# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import numpy as np

import scipy.io

import cv2

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
img_labels = scipy.io.loadmat("/kaggle/input/flower-dataset-102/imagelabels.mat")

img_labels = img_labels["labels"]

img_labels = img_labels[0]

for i in range(len(img_labels)):

  img_labels[i] = img_labels[i] - 1
import tarfile

import cv2

import matplotlib.pyplot as plt

from tqdm import tqdm



IMG_SIZE = 150



train_x = []

train_y = []

tar = tarfile.open('/kaggle/input/flower-dataset-102/102flowers.tgz', "r:gz")

i = 0

for tarinfo in tqdm(tar):

    i+=1

    tar.extract(tarinfo.name)

    

    if(tarinfo.name[-4:] == '.jpg'):

        var = tarinfo.name[11:15]

        img_num = int(var)-1

        train_y.append(img_labels[img_num])

        

        image = cv2.imread(tarinfo.name)

        resized = cv2.resize(image, (IMG_SIZE,IMG_SIZE))

        normalized_img = cv2.normalize(resized, None, alpha=0, beta=1, 

                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        train_x.append(normalized_img)



#         label_list.append(tarinfo.name.split('_')[0])

    if(tarinfo.isdir()):

        os.rmdir(tarinfo.name)

    else:

        os.remove(tarinfo.name) 



tar.close()

train_x = np.array(train_x)

train_y = np.array(train_y)
trainx, testx, trainy, testy = train_test_split(train_x, train_y, test_size=0.05, random_state=10)



trainx, valx, trainy, valy = train_test_split(trainx, trainy, test_size=0.15, random_state=10)



trainy = to_categorical(trainy)

testy = to_categorical(testy)

valy = to_categorical(valy)

np.save('testx.npy', testx)

np.save('testy.npy', testy)



print("Training data number:",len(trainx))

print("Testing data number:",len(testx))

print("Validation data number:",len(valx))



print("Training labels number:",len(trainy))

print("Testing labels number:",len(testy))

print("Validation labels number:",len(valy))
## HELPER FUNCTIONS FOR BUILDING CNN 

def create_weights(shape):

    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

 

def create_biases(size):

    return tf.Variable(tf.constant(0.05, shape=[size]))



def create_convolutional_layer(input,num_input_channels, conv_filter_size, num_filters):  

    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])

    biases = create_biases(num_filters)

 

    layer = tf.nn.conv2d(input=input, filter=weights,strides=[1, 1, 1, 1],padding='SAME')

    layer += biases

    layer = tf.nn.relu(layer)  

    layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    

    return layer



def create_flatten_layer(layer):

    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()

    layer = tf.reshape(layer, [-1, num_features])

 

    return layer

 

def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):

    weights = create_weights(shape=[num_inputs, num_outputs])

    biases = create_biases(num_outputs)

    

    if use_relu:

        layer = tf.add(tf.matmul(input, weights), biases)

        layer = tf.nn.relu(layer)

    else:

        layer = tf.add(tf.matmul(input, weights), biases, name='y_preds')



    return layer

## INITIALIZING CONSTANTS

x = tf.placeholder(tf.float32, shape=[None, 150,150,3], name='x')

y = tf.placeholder(tf.float32, shape=[None, 102], name='y')

NUM_EPOCHS = 30

BATCH_SIZE = 100

KEEP_PROB = 0.5

## BUILDING CNN

block1_conv1 = create_convolutional_layer(input=x, num_input_channels=3, conv_filter_size=3, num_filters=64)

block1_conv2 = create_convolutional_layer(input=block1_conv1, num_input_channels=64,conv_filter_size=3, num_filters=128) 

batch1 = tf.layers.batch_normalization(block1_conv2) 

drop1 = tf.nn.dropout(batch1, KEEP_PROB)



block2_conv1 = create_convolutional_layer(input=drop1, num_input_channels=128, conv_filter_size=3, num_filters=128)

block2_conv2 = create_convolutional_layer(input=block2_conv1, num_input_channels=128, conv_filter_size=3, num_filters=256)

batch2 = tf.layers.batch_normalization(block2_conv2) 

drop2 = tf.nn.dropout(batch2, KEEP_PROB)

           

layer_flat = create_flatten_layer(drop2)



layer_fc1 = create_fc_layer(input=layer_flat, num_inputs=layer_flat.get_shape()[1:4].num_elements(), num_outputs=1024, use_relu=True)

batch3 = tf.layers.batch_normalization(layer_fc1)

drop3 = tf.nn.dropout(batch3, KEEP_PROB)

y_preds = create_fc_layer(input=drop3, num_inputs=1024, num_outputs=102, use_relu=False)
## CALCULATING COST AND ACCURACY

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_preds, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

correct_pred = tf.equal(tf.argmax(y_preds, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    for epoch in tqdm(range(NUM_EPOCHS)):

      for batch in range(int(len(trainx)/BATCH_SIZE)):

        batch_x = trainx[batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,len(trainx))]

        batch_y = trainy[batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,len(trainy))]



        opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})



      for batch in range(int(len(valx)/BATCH_SIZE)):

        val_batch_x = valx[batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,len(valx))]

        val_batch_y = valy[batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,len(valy))]



        val_loss, val_acc= sess.run([cost, accuracy], feed_dict={x: val_batch_x, y: val_batch_y})

        

      print("Epoch "+str(epoch+1)+": Train Loss= "+"{:.4f}".format(loss)+"   Train Accuracy= " +  "{:.4f}".format(acc)+

              "   Valid Loss= "+"{:.4f}".format(val_loss)+"   Valid Accuracy= " + "{:.4f}".format(val_acc))



    ## SAVING THE MODEL

#     tf.saved_model.simple_save(sess, '', inputs={"x": x}, outputs={"y_preds": y_preds})

    

    builder = tf.saved_model.builder.SavedModelBuilder('')

    builder.add_meta_graph_and_variables(sess, ["tag"], signature_def_map= {"model": tf.saved_model.signature_def_utils.predict_signature_def(

                inputs={"x": x},

                outputs={"y_preds": y_preds})

            })

    builder.save()

    

    print('--- MODEL SAVED ---')
graph = tf.Graph()

with graph.as_default():

    with tf.Session(graph=graph) as sess:

        tf.saved_model.loader.load(sess, ["serve"], '')



        x = graph.get_tensor_by_name('x:0')

        y_preds = graph.get_tensor_by_name('y_preds:0')

        

        y_true = []

        preds = []

        for batch in range(int(len(testx)/BATCH_SIZE)):

          batch_x = testx[batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,len(testx))]

          batch_y = testy[batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,len(testy))]



          y_true.append(batch_y)

          preds.append(sess.run(y_preds, feed_dict={x: batch_x}))



        y_true = np.stack(np.array(y_true), axis=0)

        preds = np.stack(np.array(preds), axis=0)

# Calculte loss and accuracy

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = tf.cast(preds, tf.float32), 

                                                              labels=tf.cast(y_true, tf.float32)))

correct = tf.equal(np.argmax(preds, axis=2), np.argmax(y_true, axis=2))

accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Printing results

with tf.Session() as sess:

    print('Loss :',loss.eval())

    print('Accuracy :', accuracy.eval())
