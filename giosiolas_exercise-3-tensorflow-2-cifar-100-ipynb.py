!sudo pip install --upgrade pip

!sudo pip install --upgrade tensorflow
from __future__ import absolute_import, division, print_function, unicode_literals # legacy compatibility



import tensorflow as tf

from tensorflow.keras import datasets, layers, models



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# helper functions



# select from from_list elements with index in index_list

def select_from_list(from_list, index_list):

  filtered_list= [from_list[i] for i in index_list]

  return(filtered_list)



# append in filtered_list the index of each element of unfilterd_list if it exists in in target_list

def get_ds_index(unfiliterd_list, target_list):

  index = 0

  filtered_list=[]

  for i_ in unfiliterd_list:

    if i_[0] in target_list:

      filtered_list.append(index)

    index += 1

  return(filtered_list)



# select a url for a unique subset of CIFAR-100 with 20, 40, 60, or 80 classes

def select_classes_number(classes_number = 20):

  cifar100_20_classes_url = "https://pastebin.com/raw/nzE1n98V"

  cifar100_40_classes_url = "https://pastebin.com/raw/zGX4mCNP"

  cifar100_60_classes_url = "https://pastebin.com/raw/nsDTd3Qn"

  cifar100_80_classes_url = "https://pastebin.com/raw/SNbXz700"

  if classes_number == 20:

    return cifar100_20_classes_url

  elif classes_number == 40:

    return cifar100_40_classes_url

  elif classes_number == 60:

    return cifar100_60_classes_url

  elif classes_number == 80:

    return cifar100_80_classes_url

  else:

    return -1
# load the entire dataset

(x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
# REPLACE WITH YOUR TEAM NUMBER

team_seed = 0
# select the number of classes

cifar100_classes_url = select_classes_number()
team_classes = pd.read_csv(cifar100_classes_url, sep=',', header=None)

CIFAR100_LABELS_LIST = pd.read_csv('https://pastebin.com/raw/qgDaNggt', sep=',', header=None).astype(str).values.tolist()[0]



our_index = team_classes.iloc[team_seed,:].values.tolist()

our_classes = select_from_list(CIFAR100_LABELS_LIST, our_index)

train_index = get_ds_index(y_train_all, our_index)

test_index = get_ds_index(y_test_all, our_index)



x_train_ds = np.asarray(select_from_list(x_train_all, train_index))

y_train_ds = np.asarray(select_from_list(y_train_all, train_index))

x_test_ds = np.asarray(select_from_list(x_test_all, test_index))

y_test_ds = np.asarray(select_from_list(y_test_all, test_index))
# print our classes

print(our_classes)
# get (train) dataset dimensions

data_size, img_rows, img_cols, img_channels = x_train_ds.shape



# set validation set percentage (wrt the training set size)

validation_percentage = 0.15

val_size = round(validation_percentage * data_size)



# Reserve val_size samples for validation and normalize all values

x_val = x_train_ds[-val_size:]/255

y_val = y_train_ds[-val_size:]

x_train = x_train_ds[:-val_size]/255

y_train = y_train_ds[:-val_size]

x_test = x_test_ds/255

y_test = y_test_ds



# summarize loaded dataset

print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))

print('Validation: X=%s, y=%s' % (x_val.shape, y_val.shape))

print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))



# get class label from class index

def class_label_from_index(fine_category):

  return(CIFAR100_LABELS_LIST[fine_category.item(0)])



# plot first few images

plt.figure(figsize=(6, 6))

for i in range(9):

	# define subplot

  plt.subplot(330 + 1 + i).set_title(class_label_from_index(y_train[i]))

	# plot raw pixel data

  plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))

  #show the figure

plt.show()
# we user prefetch https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch 

# see also AUTOTUNE

# the dataset is now "infinite"



BATCH_SIZE = 128

AUTOTUNE = tf.data.experimental.AUTOTUNE # https://www.tensorflow.org/guide/data_performance



def _input_fn(x,y, BATCH_SIZE):

  ds = tf.data.Dataset.from_tensor_slices((x,y))

  ds = ds.shuffle(buffer_size=data_size)

  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds



train_ds =_input_fn(x_train,y_train, BATCH_SIZE) #PrefetchDataset object

validation_ds =_input_fn(x_val,y_val, BATCH_SIZE) #PrefetchDataset object

test_ds =_input_fn(x_test,y_test, BATCH_SIZE) #PrefetchDataset object



# steps_per_epoch and validation_steps for training and validation: https://www.tensorflow.org/guide/keras/train_and_evaluate



def train_model(model, epochs = 10, steps_per_epoch = 2, validation_steps = 1):

  history = model.fit(train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=validation_ds, validation_steps=validation_steps)

  return(history)
# plot diagnostic learning curves

def summarize_diagnostics(history):

	plt.figure(figsize=(8, 8))

	plt.suptitle('Training Curves')

	# plot loss

	plt.subplot(211)

	plt.title('Cross Entropy Loss')

	plt.plot(history.history['loss'], color='blue', label='train')

	plt.plot(history.history['val_loss'], color='orange', label='val')

	plt.legend(loc='upper right')

	# plot accuracy

	plt.subplot(212)

	plt.title('Classification Accuracy')

	plt.plot(history.history['accuracy'], color='blue', label='train')

	plt.plot(history.history['val_accuracy'], color='orange', label='val')

	plt.legend(loc='lower right')

	return plt

 

# print test set evaluation metrics

def model_evaluation(model, evaluation_steps):

	print('\nTest set evaluation metrics')

	loss0,accuracy0 = model.evaluate(test_ds, steps = evaluation_steps)

	print("loss: {:.2f}".format(loss0))

	print("accuracy: {:.2f}".format(accuracy0))



def model_report(model, history, evaluation_steps = 10):

	plt = summarize_diagnostics(history)

	plt.show()

	model_evaluation(model, evaluation_steps)
# a simple CNN https://www.tensorflow.org/tutorials/images/cnn



def init_simple_model(summary):

  model = models.Sequential()

  model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

  model.add(layers.MaxPooling2D((2, 2)))

  model.add(layers.Conv2D(64, (3, 3), activation='relu'))

  model.add(layers.MaxPooling2D((2, 2)))

  model.add(layers.Conv2D(64, (3, 3), activation='relu'))

  model.add(layers.Flatten())

  model.add(layers.Dense(64, activation='relu'))

  model.add(layers.Dense(100, activation='softmax'))

  model.compile(optimizer=tf.optimizers.Adam(), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])

  if summary: 

    model.summary()

  return model
SIMPLE_MODEL = init_simple_model(summary = True)

SIMPLE_MODEL_history = train_model(SIMPLE_MODEL, 50, 30, 5)
model_report(SIMPLE_MODEL, SIMPLE_MODEL_history, 30)
# transfer learning: VGG16 trained on ImageNet without the top layer



def init_VGG16_model(summary):

  VGG16_MODEL=tf.keras.applications.VGG16(input_shape=(img_rows, img_cols, img_channels), include_top=False, weights='imagenet')



  # unfreeze conv layers

  VGG16_MODEL.trainable=True



  dropout_layer = tf.keras.layers.Dropout(rate = 0.5)

  global_average_layer = tf.keras.layers.GlobalAveragePooling2D()



  # add top layer for CIFAR100 classification

  prediction_layer = tf.keras.layers.Dense(len(CIFAR100_LABELS_LIST),activation='softmax')

  model = tf.keras.Sequential([VGG16_MODEL, dropout_layer, global_average_layer, prediction_layer])

  model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.00005), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])

  if summary: 

    model.summary()

  return model
VGG16_MODEL = init_VGG16_model(summary = True)

VGG16_MODEL_history = train_model(VGG16_MODEL, 25, 40, 10)
model_report(VGG16_MODEL, VGG16_MODEL_history, 30)
from IPython.display import IFrame

IFrame(src='https://www.youtube.com/embed/UX8OubxsY8w', width=640, height=480)