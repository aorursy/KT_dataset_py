import numpy as np

np.random.seed(1996)



import matplotlib.pyplot as plt



import json

import time

import os

from tpot import TPOTRegressor

from sklearn.model_selection import train_test_split





from keras.layers import Input, Lambda, Dense, Flatten, Dropout, BatchNormalization

from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D

from keras.models import Model, Sequential

from keras import optimizers

from keras.regularizers import l2



from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator



from os.path import split as splt 



from sklearn.metrics import confusion_matrix



from glob import glob





# re-size all the images to this

IMAGE_SIZE = [90, 120]



# training config:

epochs = 25

batch_size = 32



# https://www.kaggle.com/paultimothymooney/blood-cells

train_path = '../input/dataset2-master/dataset2-master/images/TRAIN/'

valid_path = '../input/dataset2-master/dataset2-master/images/TEST/'



# useful for getting number of files

image_files = glob(train_path + '/*/*.jp*g')

valid_image_files = glob(valid_path + '/*/*.jp*g')



# useful for getting number of classes

folders = glob(train_path + '/*')


# define the layers:

reg = 0.5e-3 # l2-penalty

inp = Input(shape=IMAGE_SIZE+[3])



x = Conv2D(filters=32, kernel_size=(3, 3), use_bias=True, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), 

  activation='relu')(inp)

x = Conv2D(filters=32, kernel_size=(3, 3), use_bias=True, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), 

  activation='relu')(x)

x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)



x = Conv2D(filters=64, kernel_size=(3, 3), use_bias=True, kernel_regularizer=l2(reg), bias_regularizer=l2(reg),

  activation='relu')(x)

x = Conv2D(filters=64, kernel_size=(3, 3), use_bias=True, kernel_regularizer=l2(reg), bias_regularizer=l2(reg),

  activation='relu')(x)

x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)



x = Conv2D(filters=128, kernel_size=(3, 3), use_bias=True, kernel_regularizer=l2(reg), bias_regularizer=l2(reg),

  activation='relu')(x)



x = Conv2D(filters=128, kernel_size=(3, 3), use_bias=True, kernel_regularizer=l2(reg), bias_regularizer=l2(reg),

  activation='relu')(x)

x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)





x = Conv2D(filters=64, kernel_size=(3, 3), use_bias=True, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), 

  activation='relu')(x)

# x = BatchNormalization()(x)

x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)



x = Flatten()(x)

x = Dropout(0.3)(x)



prediction = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=inp, outputs=prediction)

model.summary()

# tell the model what cost and optimization method to use:

#rmsprop = optimizers.RMSprop(lr=0.001, rho=0.95, epsilon=1e-08, decay=0.1)

adam = optimizers.adam(lr=0.0001, beta_1=0.9, beta_2=0.99)

model.compile(

  loss='categorical_crossentropy',

  optimizer=adam,

  metrics=['accuracy']

)

# create an instance of ImageDataGenerator

gen = ImageDataGenerator(

  rotation_range=20,

  width_shift_range=0.1,

  height_shift_range=0.1,

  shear_range=0.1,

  zoom_range=0.2,

  preprocessing_function=lambda x: x/255.0

)



# get label mapping for confusion matrix plot later

test_gen = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE)

print(test_gen.class_indices)

labels = [None] * len(test_gen.class_indices)

for k, v in test_gen.class_indices.items():

  labels[v] = k



i = 0

for x, y in test_gen:

  print("min:", x[0].min(), "max:", x[0].max())

  plt.title(labels[np.argmax(y[0])])

  plt.imshow(x[0])

  plt.show()

  i+=1

  if i==5:

    break





# create generators

train_generator = gen.flow_from_directory(

  train_path,

  target_size=IMAGE_SIZE,

  shuffle=True,

  batch_size=batch_size,

)

valid_generator = gen.flow_from_directory(

  valid_path,

  target_size=IMAGE_SIZE,

  shuffle=True,

  batch_size=batch_size,

)





# fit the model

r = model.fit_generator(

  train_generator,

  validation_data=valid_generator,

  epochs=epochs,

  steps_per_epoch=len(image_files) // batch_size,

  validation_steps=len(valid_image_files) // batch_size,

)



score = model.evaluate_generator(valid_generator)

print('final val_loss:', score[0])

print('final val_acc:', score[1])

def get_confusion_matrix(data_path, N):

  # we need to see the data in the same order

  # for both predictions and targets

  print("Confusion Matrix", N)

  predictions = []

  targets = []

  i = 0

  for x, y in gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False, batch_size=batch_size * 2):

    i += 1

    if i % 50 == 0:

      print(i)

    p = model.predict(x)

    p = np.argmax(p, axis=1)

    y = np.argmax(y, axis=1)

    predictions = np.concatenate((predictions, p))

    targets = np.concatenate((targets, y))

    if len(targets) >= N:

      break



  cm = confusion_matrix(targets, predictions)

  return cm





cm = get_confusion_matrix(train_path, len(image_files))

print(cm)

valid_cm = get_confusion_matrix(valid_path, len(valid_image_files))

print(valid_cm)





# plot some data:



# loss:

plt.plot(r.history['loss'], label='train error')

plt.plot(r.history['val_loss'], label='test error')

plt.legend()

plt.xlabel('epoch')

plt.ylabel('error')

plt.show()



# accuracies:

plt.plot(r.history['acc'], label='train acc')

plt.plot(r.history['val_acc'], label='test acc')

plt.legend()

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.show()

import itertools

import numpy as np

import matplotlib.pyplot as plt

import os





def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion Matrix',

                          cmap=plt.cm.Blues, 

                          sv_dir=None):

  """

  This function prints and plots the confusion matrix.

  Normalization can be applied by setting `normalize=True`.

  """

  if normalize:

      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

      print("Normalized Confusion Matrix")

  else:

      print('Confussion Matrix')



  print(cm)



  plt.imshow(cm, interpolation='nearest', cmap=cmap)

  plt.title(title)

  plt.colorbar()

  tick_marks = np.arange(len(classes))

  plt.xticks(tick_marks, classes, rotation=45)

  plt.yticks(tick_marks, classes)



  fmt = '.2f' if normalize else 'd'

  thresh = cm.max() / 2.

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

      plt.text(j, i, format(cm[i, j], fmt),

               horizontalalignment="center",

               color="white" if cm[i, j] > thresh else "black")



  plt.tight_layout()

  plt.ylabel('Real Values')

  plt.xlabel('Predicted Values')

  if sv_dir is not None:

    plt.savefig(sv_dir)

  plt.show()





def y2indicator(Y):

  K = len(set(Y))

  N = len(Y)

  I = np.empty((N, K))

  I[np.arange(N), Y] = 1

  return I

plot_confusion_matrix(

  cm, 

  labels, 

  title='Training Confusion matrix'

)



plot_confusion_matrix(

  valid_cm, 

  labels, 

  title='Test Confusion Matrix'

)
