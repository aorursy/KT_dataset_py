from tensorflow.python.client import device_lib

device_lib.list_local_devices()
from __future__ import print_function

import pandas as pd

from sklearn.model_selection import train_test_split

import keras

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, BatchNormalization

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

from keras.callbacks import ModelCheckpoint

import os



# Helper libraries

import random

import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
# Seed value

# Apparently you may use different seed values at each stage

seed_value= 0



# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value

os.environ['PYTHONHASHSEED']=str(seed_value)



# 2. Set the `python` built-in pseudo-random generator at a fixed value

random.seed(seed_value)



# 3. Set the `numpy` pseudo-random generator at a fixed value

np.random.seed(seed_value)



# 4. Set the `tensorflow` pseudo-random generator at a fixed value

tf.set_random_seed(seed_value)



# 5. Configure a new global `tensorflow` session

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

K.set_session(sess)
num_classes = 10



# image dimensions

img_rows, img_cols = 28, 28



classes = ["Top", "Trouser", "Pullover", "Dress", "Coat",

	"Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
def load_data_from_keras():

    # get data using tf.keras.datasets. Train and test set is automatically split from datasets

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    return (x_train, y_train), (x_test, y_test)



#if you have download the data, then load it

def load_data_from_file():

    # load the data

    data_train = pd.read_csv('../input/fashion-mnist_train.csv')

    data_test = pd.read_csv('../input/fashion-mnist_test.csv')



    # split the train classes

    x_train = np.array(data_train.iloc[:, 1:])

    y_train = np.array(data_train.iloc[:, 0])



    # split the test classes

    x_test = np.array(data_test.iloc[:, 1:])

    y_test = np.array(data_test.iloc[:, 0])



    # Reshape image from 1D Array to 2D Array

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)

    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)

    

    return (x_train, y_train), (x_test, y_test)
(x_train, y_train), (x_test, y_test) = load_data_from_file()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0)
print("train feature shape = ", x_train.shape)

print("train classes shape = ", y_train.shape)

print("validation feature shape = ", x_val.shape)

print("validation classes shape = ", y_val.shape)

print("test feature shape = ", x_test.shape)

print("test classes shape = ", y_test.shape)
if K.image_data_format() == 'channels_first':

    x_train_with_channels = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)

    x_val_with_channels = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)

    x_test_with_channels = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

    input_shape = (1, img_rows, img_cols)

else:

    x_train_with_channels = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

    x_val_with_channels = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)

    x_test_with_channels = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    input_shape = (img_rows, img_cols, 1)

print("train feature shape = ", x_train_with_channels.shape)

print("validation feature shape = ", x_val_with_channels.shape)

print("test feature shape = ", x_test_with_channels.shape)
x_train_with_channels = x_train_with_channels.astype("float32") / 255.0

x_val_with_channels = x_val_with_channels.astype("float32") / 255.0

x_test_with_channels = x_test_with_channels.astype("float32") / 255.0
y_train_categorical = keras.utils.to_categorical(y_train, num_classes)

y_val_categorical = keras.utils.to_categorical(y_val, num_classes)

y_test_categorical = keras.utils.to_categorical(y_test, num_classes)
def create_model():

    learn_rate = 1



    model = Sequential()



    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding = 'same'))

    model.add(Conv2D(32, (3, 3), activation='relu', padding = 'same'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))



    model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))

    model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))



    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))



    model.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer=keras.optimizers.Adadelta(lr=learn_rate),

                  metrics=['accuracy'])

    return model
model = create_model()

model.summary()
checkpoint_path = 'cp-{epoch:04d}.ckpt'

checkpoint_dir = os.path.dirname(checkpoint_path)



cp_callback =  ModelCheckpoint(checkpoint_path,

                                 verbose=1,

                                 save_weights_only=True,

                                 period=1) #  save weights every 1 epochs
batch_size = 128

epochs = 100



model_train_history = model.fit(x_train_with_channels, y_train_categorical,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_val_with_channels, y_val_categorical),

          callbacks = [cp_callback])
# Plot training & validation accuracy values

plt.plot(model_train_history.history['acc'])

plt.plot(model_train_history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.grid(True)

plt.show()



# Plot training & validation loss values

plt.plot(model_train_history.history['loss'])

plt.plot(model_train_history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.grid(True)

plt.show()
test_loss, test_acc = model.evaluate(x_test_with_channels, y_test_categorical)

print('Test loss on last epoch:', test_loss)

print('Test accuracy on last epoch:', test_acc)
#load the model which trained until the 17th epoch

model_epoch_17 = create_model()

checkpoint_path = 'cp-0017.ckpt'

model_epoch_17.load_weights(checkpoint_path)



test_loss, test_acc = model_epoch_17.evaluate(x_test_with_channels, y_test_categorical)

print('Test loss on 17th epoch:', test_loss)

print('Test accuracy on 17th epoch:', test_acc)
#load the model which trained until the 26th epoch

model_epoch_26 = create_model()

checkpoint_path = 'cp-0026.ckpt'

model_epoch_26.load_weights(checkpoint_path)



test_loss, test_acc = model_epoch_26.evaluate(x_test_with_channels, y_test_categorical)

print('Test loss on 26th epoch:', test_loss)

print('Test accuracy on 26th epoch:', test_acc)
prediction_classes = model.predict_classes(x_test_with_channels)

print(classification_report(y_test, prediction_classes))
def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    #classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax



# Plot confusion matrix

plot_confusion_matrix(y_test, prediction_classes, classes=classes, normalize=False,

                      title='confusion matrix')
def plot_image(prediction_probability, image, actual_class, classes):

  plt.grid(False)

  plt.xticks([])

  plt.yticks([])

  

  plt.imshow(image, cmap=plt.cm.binary)

  

  predict_class = np.argmax(prediction_probability)

  if predict_class == actual_class:

    color = 'blue'

  else:

    color = 'red'

  

  plt.xlabel("{} {:2.0f}% ({})".format(classes[predict_class],

                                100*np.max(prediction_probability),

                                classes[actual_class]),

                                color=color)



def plot_value_array(prediction_probability, actual_class):

  plt.grid(True)

  plt.xticks(range(0,10,1))

  #plt.yticks(range(0,11,0.1))

  thisplot = plt.bar(range(10), prediction_probability, color="#777777")

  plt.ylim([0, 1])

  predicted_label = np.argmax(prediction_probability)

  

  thisplot[predicted_label].set_color('red')

  thisplot[actual_class].set_color('blue')
prediction_classes = model.predict_classes(x_test_with_channels)

true_predict_classes = prediction_classes==y_test

false_predict_classes = prediction_classes!=y_test



#split feature into true and false predicted feature

true_x_test = x_test[true_predict_classes]

false_x_test = x_test[false_predict_classes]



#split classes into true and false predicted classes

true_y_test = y_test[true_predict_classes]

false_y_test = y_test[false_predict_classes]



predictions_probability = model.predict(x_test_with_channels)



#split predictions probability into true and false predicted probability

true_predictions_probability = predictions_probability[true_predict_classes]

false_predictions_probability = predictions_probability[false_predict_classes]
num_rows = 10

num_cols = 4

num_images = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):

  plt.subplot(num_rows, 2*num_cols, 2*i+1)

  plot_image(false_predictions_probability[i], false_x_test[i], false_y_test[i], classes)

  plt.subplot(num_rows, 2*num_cols, 2*i+2)

  plot_value_array(false_predictions_probability[i], false_y_test[i])

plt.show()
num_rows = 10

num_cols = 4

num_images = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):

  plt.subplot(num_rows, 2*num_cols, 2*i+1)

  plot_image(true_predictions_probability[i], true_x_test[i], true_y_test[i], classes)

  plt.subplot(num_rows, 2*num_cols, 2*i+2)

  plot_value_array(true_predictions_probability[i], true_y_test[i])

plt.show()
print("image shape = ", x_test_with_channels[0,:,:,0].shape)

plt.imshow(x_test_with_channels[0,:,:,0])

plt.show()
def show_conv_output(layer_number, data_index):

  #get hidden layer output

  get_i_layer_output = K.function([model.layers[0].input],

                                    [model.layers[layer_number].output])

  layer_output = get_i_layer_output([x_test_with_channels])[0]



  print("conv output shape = ", layer_output.shape)



  (_, _, _, filters) = layer_output.shape

  n_columns = 6

  n_rows = np.math.ceil(filters / n_columns) + 1

  plt.figure(1, figsize=(20,20))

  for i in range(filters):

      plt.subplot(n_rows, n_columns, i+1)

      plt.title(' ' + str(i))

      plt.imshow(layer_output[data_index,:,:,i])
show_conv_output(layer_number=0, data_index=0)
show_conv_output(layer_number=1, data_index=0)
show_conv_output(layer_number=4, data_index=0)
show_conv_output(layer_number=5, data_index=0)