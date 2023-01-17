# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/train/"))

import keras

import cv2

from keras.applications.vgg16 import VGG16

import tensorflow as tf

import random

np.random.seed(0)

from tqdm import tqdm_notebook

# model = VGG16(weights='imagenet', include_top=False)

# Any results you write to the current directory are saved as output.
TRAIN_IMAGES_PATH = '../input/train/train'

TEST_IMAGES_PATH = '../input/test/test'
# Create training, test dataframes for image IDs and their respective labels

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/sample_submission.csv')

train_image_ids = train_df['id']

training_labels = train_df['has_cactus']

test_image_ids = test_df['id']

test_labels = test_df['has_cactus']
def get_images(folder_path, image_ids):

    """

    Function to read images from disk and normalize them

    """

    all_images = list()

    for image_name in tqdm_notebook(image_ids):

        image_path = os.path.join(folder_path, image_name)

        image = cv2.imread(image_path)

        all_images.append(image)

    input_images = np.stack(all_images)

    return input_images, input_images / 255

    
# Store train and test images in a numpy array

all_train_images, normalized_images = get_images(TRAIN_IMAGES_PATH, train_image_ids)

test_images, normalized_test_images = get_images(TEST_IMAGES_PATH, test_image_ids)
augs = [np.fliplr, np.flipud, np.rot90] # List of augmentations to be applied to the data

def augment(images, labels, augs):

    """

    Apply data augmentation to all training images

    To tackle class imbalance, apply one of the augmentations ( Randomly chosen )

    to each image having label 1, and apply all transformations to image having

    label 0.

    """

    all_images = list()

    all_labels = list()

    for i,image in tqdm_notebook(enumerate(images)):

        all_images.append(image)

        cur_label = labels[i]

        all_labels.append(cur_label)

        if cur_label == 1:

            all_images.append(augs[random.randint(0,2)](image))

            all_labels.append(cur_label)

        else:

            for aug in augs:

                all_labels.append(cur_label)

                all_images.append(aug(image))

    

    return np.stack(all_images), np.array(all_labels)

# 

normalized_train_images, final_training_labels = augment(normalized_images, np.array(training_labels), augs)

NUM_TRAIN_IMAGES = int(0.75 * normalized_train_images.shape[0])

indices = np.random.permutation(normalized_train_images.shape[0])

training_idx, val_idx = indices[:NUM_TRAIN_IMAGES], indices[NUM_TRAIN_IMAGES:]

train_data = normalized_train_images[training_idx,:]

train_labels = np.array(final_training_labels)[training_idx]



val_data = normalized_train_images[val_idx,:]

val_labels = final_training_labels[val_idx]
def show_images_horizontally(images, labels=[], lookup_label=None,

                            figsize=(15, 30)):

    """

    Utility function to show images

    """



    import matplotlib.pyplot as plt

    from matplotlib.pyplot import figure, imshow, axis

    print(labels[0])

    fig = figure(figsize=figsize)

    for i in range(images.shape[0]):

        fig.add_subplot(1, images.shape[0], i + 1)

        if lookup_label:

            plt.title(lookup_label[labels[i]])

        imshow(images[i])

        axis('off')

print(final_training_labels[:30])



show_images_horizontally(normalized_train_images[10:20], np.array(final_training_labels[10:20]), lookup_label={1:"has_cactus", 0: "no_cactus"})
def get_weights(shape):

  """

  Weights initializer

  """

  initializer = tf.contrib.layers.xavier_initializer()

  return tf.Variable(initializer(shape=shape))

  

def get_biases(length):

    """

    Initializing bias

    """

    return tf.Variable(tf.constant(0.0005, shape=[length]))

    
def conv_layer(input, in_channels, filter_size, num_filters):

    """

    Apply convolution operation to the image

    """

    shape = [filter_size, filter_size, in_channels, num_filters]

    weights = get_weights(shape)

    bias = get_biases(num_filters)

    layer = tf.nn.convolution(input,

    weights,

    strides=[1,1], dilation_rate=[1,1], 

    padding="SAME")

    layer= tf.nn.bias_add(layer, bias)

    new_layer = tf.nn.relu(layer)

    print(new_layer)

    return new_layer, weights





def flatten(input_tensor):

    """

    Flattens input tensor

    """

    layer_shape = input_tensor.get_shape()

    total_elements = layer_shape[1:4].num_elements()

    layer = tf.reshape(input_tensor, [-1, total_elements])

    return layer, total_elements





def fc_layer(input, in_features, out_features):

  """

  Create fully connected layer

  """

  weight = get_weights([in_features, out_features])

  bias = get_biases(out_features)

  layer = tf.matmul(input, weight) + bias

  return layer
input_image = tf.placeholder(name="input", shape=(None, 32, 32, 3), dtype=tf.float32)

labels = tf.placeholder(name="labels", shape=(None), dtype=tf.int64)

# input_image = tf.cast(input_image, tf.float32)





with tf.variable_scope("block1_conv1"):

  layer_conv1, weights_1 = conv_layer(input_image, 3, 3, 32)

with tf.variable_scope("block1_conv2"):

  layer_conv2, weights_2 = conv_layer(layer_conv1, 32, 3, 32)

#   layer_output_pool = tf.nn.max_pool(layer_conv2, ksize = [1,2,2,1], strides = [1,1,1,1],padding = "SAME")

with tf.variable_scope("block2_conv1"):

  layer_conv3, weights_3 = conv_layer(layer_conv2, 32, 3, 64)

with tf.variable_scope("block2_conv2"):

  layer_conv4, weights_4 = conv_layer(layer_conv3, 64, 3, 64)

  layer_output_pool = tf.nn.max_pool(layer_conv4, ksize = [1,4,4,1], strides = [1,2,2,1], padding = "VALID")



with tf.variable_scope("block3_conv1"):

  layer_conv4, weights_4 = conv_layer(layer_output_pool, 64, 3, 128)

with tf.variable_scope("block3_conv2"):

  layer_conv5, weights_5 = conv_layer(layer_conv4, 128, 3, 128)

with tf.variable_scope("block3_conv3"):

  layer_conv6, weights_6 = conv_layer(layer_conv5, 128, 3, 128)

  layer_output_pool = tf.nn.max_pool(layer_conv6, ksize = [1,4,4,1], strides = [1,2,2,1], padding = "VALID")

print(layer_output_pool, layer_conv4)

flattened_layer, in_features = flatten(layer_output_pool)

fclyr = fc_layer(flattened_layer, in_features, 128)

fc_layer2 = tf.nn.relu(fclyr)



final_layer = fc_layer(fc_layer2, 128, 2)

y_pred = tf.nn.softmax(final_layer)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_layer,

                                                        labels=tf.one_hot(labels,2))

cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

gpu_options = tf.GPUOptions(allow_growth=True)

config = tf.ConfigProto(gpu_options=gpu_options)

init=tf.global_variables_initializer()


sess = tf.Session(config = config)

NUM_ITERATIONS = 20

BATCH = 32

sess.run(init)

for i in tqdm_notebook(range(NUM_ITERATIONS)):

    num_batches = int(train_data.shape[0] / BATCH) + 1

    losses = list()

    epoch_predictions = list()

    for j in tqdm_notebook(range(num_batches)):

        batch_labels = train_labels[BATCH*j: BATCH*j + BATCH].astype(np.int64)

        batch_data = train_data[BATCH*j: BATCH*j + BATCH]

        loss, _, probabilities = sess.run([cost, optimizer, y_pred], feed_dict={input_image:batch_data, labels:batch_labels})

        predictions = np.argmax(probabilities, axis=1)

        epoch_predictions.extend(predictions)

        losses.append(loss)

    print("EPOCH " + str(i))

    epoch_predictions = np.array(epoch_predictions)

    train_loss = np.mean(np.array(losses))

    train_accuracy = (np.sum(epoch_predictions==train_labels) / train_labels.shape[0]) * 100

    num_batches = int(val_data.shape[0] / BATCH) + 1

    val_predictions = list()

    for j in range(num_batches):

        batch_data = val_data[BATCH*j: BATCH*j + BATCH]

        batch_labels = val_labels[BATCH*j: BATCH*j + BATCH].astype(np.int64)

        loss, _, probabilities = sess.run([cost, optimizer, y_pred], feed_dict={input_image:batch_data, labels:batch_labels})

        predictions = np.argmax(probabilities, axis=1)

        val_predictions.extend(predictions)

        losses.append(loss)

    val_accuracy = (np.sum(val_predictions==val_labels) / val_labels.shape[0]) * 100

    print("Loss after Epoch - %f, Train Accuracy - %f, Val Accuracy - %f" % (train_loss, train_accuracy, val_accuracy))

        

        

        
test_preds = list()

num_batches = int(normalized_test_images.shape[0] / BATCH) + 1

for j in range(num_batches):

    batch_data = normalized_test_images[BATCH*j: BATCH*j + BATCH]

    batch_labels = test_labels[BATCH*j: BATCH*j + BATCH].astype(np.int64)

    probabilities = sess.run(y_pred, feed_dict={input_image:batch_data, labels:batch_labels})

    predictions = np.argmax(probabilities, axis=1)

    test_preds.extend(predictions)

submission_df = test_df

submission_df['has_cactus'] = np.array(test_preds)

submission_df.to_csv("submissions.csv",index=False)
# test_preds


# tf.reset_default_graph()



# from keras.models import Sequential

# from keras.layers import Dense, Dropout, Flatten

# from keras.layers import Conv2D, MaxPooling2D
# model = Sequential()

# model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(32,32,3)))

# model.add(Conv2D(32, kernel_size=(3, 3),activation='relu'))

# model.add(Conv2D(64, kernel_size=(3, 3),activation='relu'))

# model.add(Conv2D(64, kernel_size=(3, 3),activation='relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(128, kernel_size=(3, 3),activation='relu'))          

# model.add(Conv2D(128, kernel_size=(3, 3),activation='relu'))

# model.add(Conv2D(128, kernel_size=(3, 3),activation='relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())

# model.add(Dense(128, activation='relu'))

# model.add(Dense(2, activation='softmax'))

# model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
# print(train_data.shape)

# print(train_labels.shape)

# history=model.fit(train_data, np.eye(2)[train_labels], epochs=20)
# # Validation data accuract

# scores = model.evaluate(val_data, np.eye(2)[np.array(val_labels, dtype=np.int64)], verbose=0)

# print("Val Accuracy:"+"%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# predictions = model.predict_classes(normalized_test_images)
# submission_df = test_df

# submission_df['has_cactus'] = predictions

# submission_df.to_csv("submissions.csv",index=False)