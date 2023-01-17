# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

#tf.enable_eager_execution()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
os.getcwd()
os.listdir()
import tensorflow as tf

import pathlib

import matplotlib.pyplot as plt

import seaborn as sns

import random

from IPython.core.display import Image

from IPython.display import display

#print(tf.__version__)

%matplotlib inline
##### Read data

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

sample_sub_df = pd.read_csv('../input/train.csv')
train_images_path =   "../input/train_images"

test_images_path =  "../input/test_images"

print(train_images_path)
train_df['image_path'] = "../input/train_images/" + train_df['id_code'] + ".png"

test_df['image_path'] = "../input/test_images/" + test_df['id_code'] + ".png"
train_df.info()
train_df[train_df.id_code == '5d024177e214']
classes_dist = pd.DataFrame(train_df['diagnosis'].value_counts()/train_df.shape[0]).reset_index()



# barplot 

ax = sns.barplot(x="index", y="diagnosis", data=classes_dist)



# Imbalanced dataset with 49% - no DR, 8% proliferative - i.e most severe DR

# Model Building - Need to do oversampling for minority classes
test_df.info()
root_path = pathlib.Path(train_images_path) # Returns POSIX path

# for item in root_path.iterdir():

#     print(item)

#     break
#!pip install ipython

#!conda install -c anaconda ipython
all_paths = list(root_path.glob("*.png"))

all_paths[0:3]
#all_paths = list(root_path.glob("*.png"))

## Display few images of each class - 

all_paths = [str(path) for path in all_paths]

random.shuffle(all_paths)





# Image('../input/train_images/5d024177e214.png',width=300,height=300)

for n in range(3):

    image_path = random.choice(all_paths)

    print(image_path)

    display(Image(image_path,width=300,height=300))
# To decode datatype

def preprocess_image(image,labels):

    image = tf.image.decode_png(image, channels=3)

    image = tf.image.resize(image, [28,28])         # IMAGE RESIZING

    image = tf.cast(image, tf.float32)                 

    image /= 255.0  # normalize to [0,1] range         # IMAGE NORMALIZE

    

    

    return(image,labels)



# Read image, and process

def load_and_preprocess_image(path,labels):

    image = tf.read_file(path)

    return(preprocess_image(image,labels))

labels = tf.convert_to_tensor(np.array(train_df['diagnosis']), dtype=tf.int32)

filenames  = tf.convert_to_tensor(train_df['image_path'].tolist(), dtype=tf.string)

filenames[0:5]

labels[0:4]
filenames.shape
tf.one_hot(labels[0], 5)
def read_images(filenames,labels,batch_size):

    

    

    dataset= tf.data.Dataset.from_tensor_slices((filenames,labels))

    

    

    #labels =  tf.data.Dataset.from_tensor_slices(labels).map(lambda z: tf.one_hot(z, 5))

    

    #labels =  tf.data.Dataset.from_tensor_slices(labels)

    #dataset = dataset.shuffle(len(labels))



    # Image preprocessing

    #dataset = dataset.map(load_and_preprocess_image,num_parallel_calls=4)

    dataset = dataset.map(load_and_preprocess_image) 

    dataset.make_initializable_iterator()

    

    

    #print(dataset)

    # Get one batch

    dataset = dataset.batch(batch_size,drop_remainder=False)

    dataset = dataset.prefetch(1)

    #dataset = dataset.shape

    

#     X,Y = tf.train.batch([dataset, labels], batch_size=batch_size,

#                           capacity=batch_size * 8,

#                           num_threads=4

    #print(labels)

    return(dataset,labels)



# ## Create train & test datasets

# filenames  = tf.convert_to_tensor(train['image_path'].tolist(), dtype=tf.string)

# dx_train = tf.data.Dataset.from_tensor_slices(filenames)



# labels = tf.convert_to_tensor(np.array(train['diagnosis']), dtype=tf.int32)

# dy_train = tf.data.Dataset.from_tensor_slices(labels).map(lambda z: tf.one_hot(z, 5))



# train_dataset = tf.data.Dataset.zip((dx_train, dy_train)).shuffle(500).repeat().batch(30)
# filenames  = tf.convert_to_tensor(valid['image_path'].tolist(), dtype=tf.string)

# dx_valid = tf.data.Dataset.from_tensor_slices(filenames)



# labels = tf.convert_to_tensor(np.array(valid['diagnosis']), dtype=tf.int32)

# dy_valid = tf.data.Dataset.from_tensor_slices(labels).map(lambda z: tf.one_hot(z, 5))



# valid_dataset = tf.data.Dataset.zip((dx_test, dy_test)).shuffle(500).repeat().batch(30)
# iterator = tf.data.Iterator.from_structure(train_dataset.output_types,

#                                                train_dataset.output_shapes)

# next_element = iterator.get_next()



# training_init_op = iterator.make_initializer(train_dataset)

# validation_init_op = iterator.make_initializer(valid_dataset)
def cnn_model(in_data):

    input_layer = tf.reshape(in_data, [-1, 28, 28, 3])

    

    input_layer = in_data

    conv1 = tf.layers.conv2d(

      inputs=input_layer,

      filters=32,

      kernel_size=[5, 5],

      padding="same",

      activation=tf.nn.relu)



     # Pooling Layer #1

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)



    # Convolutional Layer #2 and Pooling Layer #2

    conv2 = tf.layers.conv2d(

      inputs=pool1,

      filters=64,

      kernel_size=[5, 5],

      padding="same",

      activation=tf.nn.relu)



    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)



      # Dense Layer

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    #dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)

#     dropout = tf.layers.dropout(

#           inputs=dense, rate=0.4, training == tf.estimator.ModeKeys.TRAIN)



      # Logits Layer

    logits = tf.layers.dense(inputs=dense, units=5)



    predictions = {

      # Generate predictions (for PREDICT and EVAL mode)

      "classes": tf.argmax(input=logits, axis=1),

      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the

      # `logging_hook`.

      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")

  }



    return predictions
labels
dataset,labels = read_images(filenames,labels,128)

iter = dataset.make_one_shot_iterator()

x, y = iter.get_next()



predictions = cnn_model(x)

predictions
predictions["classes"]
predictions["probabilities"]
#tf.disable_eager_execution()
x.shape
predictions 
EPOCHS = 10

BATCH_SIZE = 128

# using two numpy arrays

# features, labels = (np.array([np.random.sample((100,2))]), 

#                     np.array([np.random.sample((100,1))]))

# dataset = tf.data.Dataset.from_tensor_slices((filenames,labels)).repeat().batch(BATCH_SIZE)

# iter = dataset.make_one_shot_iterator()

# x, y = iter.get_next()



dataset,labels = read_images(filenames,labels,128)

iter = dataset.make_one_shot_iterator()

x, y = iter.get_next()



#tf.disable_eager_execution()

# make a simple model

# net = tf.layers.dense(x, 8, activation=tf.tanh) # pass the first value from iter.get_next() as input

# net = tf.layers.dense(net, 8, activation=tf.tanh)

predictions = cnn_model(x)

prediction = predictions["classes"]

loss = tf.losses.mean_squared_error(prediction, y) # pass the second value from iter.get_net() as label

train_op = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(EPOCHS):

        _, loss_value = sess.run([train_op, loss])

        print("Iter: {}, Loss: {:.4f}".format(i, loss_value))
## Testing



# with tf.Session() as sess:

#     data,labels = read_images(filenames,labels,128)

    

#     iterator = data.make_initializable_iterator()

#     image_batch,labels= iterator.get_next()

    

#     #image_batch,labels = next(iter(data))

#     out = tf.layers.conv2d(image_batch,filters=32,kernel_size=(3,3))

#     out = tf.nn.relu(out)

#     out = tf.layers.max_pooling2d(out, 2, 2)

#     out = tf.reshape(out, [-1, 32 * 32 * 16])

#     # Now, logits is [batch_size, 6]

#     logits = tf.layers.dense(out, 5)
# predictions = tf.argmax(logits, 1)

# predictions
# loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

# optimizer = tf.train.AdamOptimizer(0.01)



# # Create the training operation

# train_op = optimizer.minimize(loss)
# N_CLASSES = train_df.diagnosis.nunique()

# N_CLASSES
#tf.reset_default_graph() 
# Build the data input

#X, Y = read_images(filenames, labels, batch_size)