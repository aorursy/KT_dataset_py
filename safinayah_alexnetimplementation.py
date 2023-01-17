from __future__ import division, print_function, absolute_import

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import keras
import keras.layers as layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard
# read training & testing data


#  trainImg = pd.read_csv("/Your/dataset/training/file/in/csv/form", header=None)
# trainLabel = pd.read_csv("/Your/dataset/training/labels/in/csv/form", header=None)
# testImg = pd.read_csv("/Your/dataset/teastinf/file/in/csv/form", header=None)
# testLabel = pd.read_csv("/Your/dataset/testing/labels/in/csv/form", header=None)

trainImg.head()
testImg.head()
# Split data into training set and validation set
#training images
trainImg = trainImg.values.astype('float32') /255.0
#training labels
trainLabel = trainLabel.values.astype('int32') 

#testing images
testImg = testImg.values.astype('float32')/255.0
#testing labels
testLabel = testLabel.values.astype('int32')
trainImg[0]
#One Hot encoding of train labels.
trainLabel = to_categorical(trainLabel,10)

#One Hot encoding of test labels.
testLabel = to_categorical(testLabel,10)
trainLabel[0]
print(trainImg.shape, trainLabel.shape, testImg.shape, testLabel.shape)
# reshape input images to 28x28x1
trainImg = trainImg.reshape([-1, 28, 28, 1])
testImg = testImg.reshape([-1, 28, 28, 1])
print(trainImg.shape, trainLabel.shape, testImg.shape, testLabel.shape)
trainImg[0]
def alexnet(input_shape, n_classes):
  input = Input(input_shape)
  
  # actually batch normalization didn't exist back then
  # they used LRN (Local Response Normalization) for regularization
  x = Conv2D(96, 11, strides=4, padding='same', activation='relu')(input)
  x = BatchNormalization()(x)
  x = MaxPool2D(3, strides=2)(x)
  
  x = Conv2D(256, 5, padding='same', activation='relu')(x)
  x = BatchNormalization()(x)
  x = MaxPool2D(3, strides=2)(x)
  
  x = Conv2D(384, 3, strides=1, padding='same', activation='relu')(x)
  
  x = Conv2D(384, 3, strides=1, padding='same', activation='relu')(x)
  
  x = Conv2D(256, 3, strides=1, padding='same', activation='relu')(x)
  x = BatchNormalization()(x)
  x = MaxPool2D(3, strides=2)(x)
  
  x = Flatten()(x)
  x = Dense(4096, activation='relu')(x)
  x = Dense(4096, activation='relu')(x)
  
  output = Dense(n_classes, activation='softmax')(x)
  
  model = Model(input, output)
  return model
input_shape = 28, 28, 1
n_classes = 10

K.clear_session()
model = alexnet(input_shape, n_classes)
model.summary()
repetitions = 10
input = np.random.randn(1, *input_shape)

output = model.predict(input)
start = time()
for _ in range(repetitions):
  output = model.predict(input)
  
print((time() - start) / repetitions)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(trainImg, trainLabel, 
          batch_size=100, epochs=1, verbose=1)
# print('Predict the classes: ')
# prediction = model.predict_classes(trainImg)
# print('Predicted classes: ', prediction)
# Evaluate model
score = model.evaluate(testImg, testLabel)
print('Loss on Test set: %0.2f%%' % (score[0] * 100))
print('Test accuarcy: %0.2f%%' % (score[1] * 100))