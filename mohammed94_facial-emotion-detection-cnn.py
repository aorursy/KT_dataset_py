import json

import zipfile

import os

import pandas as pd

from matplotlib import pyplot

from math import sqrt 

import numpy as np 

import scipy.misc 

from IPython.display import display 

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator 

from keras.utils import plot_model

from keras.models import Model, Sequential

from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.optimizers import Adam, SGD

from keras.regularizers import l1, l2

from keras.utils import plot_model

from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix

import itertools

from pylab import rcParams



%matplotlib inline

data= pd.read_csv('../input/challenges-in-representation-learning-facial-expression-recognition-challenge/train.csv')

data.head()
rcParams['figure.figsize'] = 15, 10
data.emotion.value_counts()
num_classes = 7

width = 48

height = 48

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

classes=np.array(emotion_labels)
depth = 1

height = int(sqrt(len(data.pixels[0].split()))) 

width = int(height)
h, w = 10, 10        

nrows, ncols = 1, 8  # array of sub-plots

figsize = [20, 30]     # figure size, inches





# create figure (fig), and array of axes (ax)

fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)



# plot simple raster image on each sub-plot

for i, axi in enumerate(ax.flat):

    # i runs from 0 to (nrows*ncols-1)

    # axi is equivalent with ax[rowid][colid]

    img = np.mat(data.pixels[i]).reshape(height, width) 

    axi.imshow(img)

    # get indices of row/column

    rowid = i // ncols

    colid = i % ncols

    axi.set_title(emotion_labels[data.emotion[i]])



plt.tight_layout(True)

plt.show()


def gray_to_rgb(im):

  '''

  converts images from single channel images to 3 channels

  '''



  w, h = im.shape

  ret = np.empty((w, h, 3), dtype=np.uint8)

  ret[:, :, 2] =  ret[:, :, 1] =  ret[:, :, 0] =  im

  return ret



def convert_to_image(pixels, mode="save", t="gray"):

  

  '''

  convert the input pixels from the single string row to  48*48 array with real pixel values

  when mode = "save" it keeps the images in flat array shape, otherwise it converts it to 48*48

  when t (for type) = "gray, it keeps the pixels single channel, otherwise it converts it to 3 channels

  '''



  if type(pixels) == str:

      pixels = np.array([int(i) for i in pixels.split()])

  if mode == "show":

    if t == "gray":

      return pixels.reshape(48,48)

    else:

      return gray_to_rgb(pixels.reshape(48,48))

  else:

      return pixels



data["pixels"] = data["pixels"].apply(lambda x : convert_to_image(x, mode="show", t="gray"))

from sklearn.model_selection import train_test_split

#split the data to train, test, and validation

X_train, X_test, y_train, y_test = train_test_split(data["pixels"],  data["emotion"], test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)



X_train = np.array(list(X_train[:]), dtype=np.float)

X_val = np.array(list(X_val[:]), dtype=np.float)

X_test = np.array(list(X_test[:]), dtype=np.float)



y_train = np.array(list(y_train[:]), dtype=np.float)

y_val = np.array(list(y_val[:]), dtype=np.float)

y_test = np.array(list(y_test[:]), dtype=np.float)



X_train = X_train.reshape(X_train.shape[0], 48, 48, 1) 

X_val = X_val.reshape(X_val.shape[0], 48, 48, 1)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
num_train = X_train.shape[0]

num_val = X_val.shape[0]

num_test = X_test.shape[0]

y_train = np_utils.to_categorical(y_train, num_classes) 

# y_val = val_set.emotion 

y_val = np_utils.to_categorical(y_val, num_classes) 

# y_test = test_set.emotion 

y_test = np_utils.to_categorical(y_test, num_classes) 
datagen = ImageDataGenerator( 

    rescale=1./255,

    rotation_range = 10,

    horizontal_flip = True,

    width_shift_range=0.1,

    height_shift_range=0.1,

    fill_mode = 'nearest')



testgen = ImageDataGenerator( 

    rescale=1./255

    )

datagen.fit(X_train)

batch_size = 64
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):

    for i in range(0, 9): 

        pyplot.axis('off') 

        pyplot.subplot(330 + 1 + i) 

        # print(np.where(y_batch[i] == 1)[0][0])

        pyplot.title(emotion_labels[np.where(y_batch[i] == 1)[0][0]])

        pyplot.imshow(X_batch[i].reshape(48, 48), cmap=pyplot.get_cmap('gray'))

    pyplot.axis('off') 

    pyplot.show() 

    break 
import keras
train_flow = datagen.flow(X_train, y_train, batch_size=batch_size) 

val_flow = testgen.flow(X_val, y_val, batch_size=batch_size) 

test_flow = testgen.flow(X_test, y_test, batch_size=batch_size) 
conv5_model = Sequential()



conv5_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))

conv5_model.add(BatchNormalization())

conv5_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',name = 'conv1'))

conv5_model.add(BatchNormalization())

conv5_model.add(MaxPooling2D(pool_size=(2, 2)))

conv5_model.add(Dropout(0.25))



conv5_model.add(Conv2D(128, kernel_size=(3, 3),padding="same", activation='relu', name = 'conv2'))

conv5_model.add(BatchNormalization())

conv5_model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv3'))

conv5_model.add(BatchNormalization())

conv5_model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv4'))

conv5_model.add(BatchNormalization())

conv5_model.add(MaxPooling2D(pool_size=(2, 2)))

conv5_model.add(Dropout(0.25))



conv5_model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv5'))

conv5_model.add(BatchNormalization())

conv5_model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv6'))

conv5_model.add(BatchNormalization())

conv5_model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv7'))

conv5_model.add(BatchNormalization())

conv5_model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv8'))

conv5_model.add(BatchNormalization())

conv5_model.add(MaxPooling2D(pool_size=(2, 2)))

conv5_model.add(Dropout(0.25))



conv5_model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv9'))

conv5_model.add(BatchNormalization())

conv5_model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv10'))

conv5_model.add(BatchNormalization())

conv5_model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv11'))

conv5_model.add(BatchNormalization())

conv5_model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv12'))

conv5_model.add(BatchNormalization())

conv5_model.add(MaxPooling2D(pool_size=(2, 2)))

conv5_model.add(Dropout(0.25))



conv5_model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv13'))

conv5_model.add(BatchNormalization())

conv5_model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv14'))

conv5_model.add(BatchNormalization())

conv5_model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation='relu',  name = 'conv16'))

conv5_model.add(BatchNormalization())

conv5_model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv17'))

conv5_model.add(BatchNormalization())

conv5_model.add(MaxPooling2D(pool_size=(2, 2)))

conv5_model.add(Dropout(0.25))





conv5_model.add(Flatten())

conv5_model.add(Dense(num_classes, activation='softmax'))

print(conv5_model.summary())

opt = Adam(lr=0.0001, decay=1e-6)

conv5_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) 
plot_model(conv5_model, to_file='model.png')
from keras.callbacks import ModelCheckpoint

filepath="weights_min_loss.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint]
num_epochs = 100  

history = conv5_model.fit_generator(train_flow, 

                    steps_per_epoch=len(X_train) / batch_size, 

                    epochs=num_epochs,  

                    verbose=2,  

                    callbacks=callbacks_list,

                    validation_data=val_flow,  

                    validation_steps=len(X_val) / batch_size) 
%matplotlib inline

def visualize_acc(history):



  train_loss=history.history['loss']

  val_loss=history.history['val_loss']

  train_acc=history.history['accuracy']

  val_acc=history.history['val_accuracy']



  epochs = range(len(train_acc))



  plt.plot(epochs,train_loss,'r', label='train_loss')

  plt.plot(epochs,val_loss,'b', label='val_loss')

  plt.title('train_loss vs val_loss')

  plt.xlabel('epoch')

  plt.ylabel('loss')

  plt.legend()

  plt.figure()



  plt.plot(epochs,train_acc,'r', label='train_acc')

  plt.plot(epochs,val_acc,'b', label='val_acc')

  plt.title('train_acc vs val_acc')

  plt.xlabel('epoch')

  plt.ylabel('accuracy')

  plt.legend()

  plt.figure()
visualize_acc(history)
loss = conv5_model.evaluate_generator(test_flow, steps=len(X_test) / batch_size) 

print("Test Loss " + str(loss[0]))

print("Test Acc: " + str(loss[1]))
loss = conv5_model.evaluate(X_val/255., y_val) 

print("Test Loss " + str(loss[0]))

print("Test Acc: " + str(loss[1]))
y_pred_ = conv5_model.predict(X_test/255., verbose=1)

y_pred = np.argmax(y_pred_, axis=1)

t_te = np.argmax(y_test, axis=1)
def plot_confusion_matrix(y_test, y_pred, classes,

                          normalize=False,

                          title='Unnormalized confusion matrix',

                          cmap=plt.cm.Blues):

    cm = confusion_matrix(y_test, y_pred)

    

    if normalize:

        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)

        

    np.set_printoptions(precision=2)

        

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    thresh = cm.min() + (cm.max() - cm.min()) / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True expression')

    plt.xlabel('Predicted expression')

    plt.show()
fig = plot_confusion_matrix(y_test=t_te, y_pred=y_pred,

                      classes=classes,

                      normalize=True,

                      cmap=plt.cm.Greys,

                      title='Average accuracy: ' + str(np.sum(y_pred == t_te)/len(t_te)) + '\n')
#let's try smaller Conv model

model_conv = Sequential()



model_conv.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))

model_conv.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model_conv.add(MaxPooling2D(pool_size=(2, 2)))

model_conv.add(Dropout(0.25))



model_conv.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

model_conv.add(MaxPooling2D(pool_size=(2, 2)))

model_conv.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

model_conv.add(MaxPooling2D(pool_size=(2, 2)))

model_conv.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

model_conv.add(MaxPooling2D(pool_size=(2, 2)))

model_conv.add(Dropout(0.25))



model_conv.add(Flatten())

model_conv.add(Dense(1024, activation='relu'))

model_conv.add(Dropout(0.5))

model_conv.add(Dense(7, activation='softmax'))
filepath="weights_min_loss_conv.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint]

# Do not forget to compile it

model_conv.compile(loss='categorical_crossentropy',

                     optimizer='rmsprop',

                     metrics=['accuracy'])
num_epochs = 100 

history = model_conv.fit_generator(train_flow, 

                    steps_per_epoch=len(X_train) / batch_size, 

                    epochs=num_epochs,  

                    verbose=2,  

                    callbacks=callbacks_list,

                    validation_data=val_flow,  

                    validation_steps=len(X_val) / batch_size)
plot_model(model_conv, to_file='model.png')
visualize_acc(history)
loss = model_conv.evaluate_generator(test_flow, steps=len(X_test) / batch_size) 

print("Test Loss " + str(loss[0]))

print("Test Acc: " + str(loss[1]))
conv5_model.save('conv5_model')
loss = model_conv.evaluate(X_test/255., y_test) 

print("Test Loss " + str(loss[0]))

print("Test Acc: " + str(loss[1]))
loss = model_conv.evaluate(X_val/255., y_val) 

print("Test Loss " + str(loss[0]))

print("Test Acc: " + str(loss[1]))
y_pred_ = model_conv.predict(X_test/255., verbose=1)

y_pred = np.argmax(y_pred_, axis=1)

t_te = np.argmax(y_test, axis=1)
fig = plot_confusion_matrix(y_test=t_te, y_pred=y_pred,

                      classes=classes,

                      normalize=True,

                      cmap=plt.cm.Greys,

                      title='Average accuracy: ' + str(np.sum(y_pred == t_te)/len(t_te)) + '\n')