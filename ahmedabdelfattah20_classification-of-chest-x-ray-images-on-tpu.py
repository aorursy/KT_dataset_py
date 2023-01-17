import os

import matplotlib.pyplot as plt

import cv2

import numpy as np

import random

import time

from keras.layers import *

from keras.models import Input, Model

from keras.utils import to_categorical

from keras.callbacks import callbacks

from keras.callbacks import ModelCheckpoint

from keras.models import load_model

import tensorflow as tf

from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
numberOfNormal = len(os.listdir('/kaggle/input/chest-xray-images-for-classification-pneumonia/pneumonia/train/NORMAL')) + len(os.listdir('/kaggle/input/chest-xray-images-for-classification-pneumonia/pneumonia2/train/NORMAL'))

numberOfPneumonia = len(os.listdir('/kaggle/input/chest-xray-images-for-classification-pneumonia/pneumonia/train/PNEUMONIA')) + len(os.listdir('/kaggle/input/chest-xray-images-for-classification-pneumonia/pneumonia2/train/PNEUMONIA'))



print('number of normal samples is {0} and number of pneumonia samples is {1}'.format(numberOfNormal, numberOfPneumonia))

print('Ratio of Normal samples over pneumonia samples is ', numberOfNormal / numberOfPneumonia)
X_train, y_train= [], []

X_test, y_test = [], []

X_validation, y_validation = [], []

def prepareData(mainPath) : 

  global X_train, y_train, X_test, y_test, X_validation, y_validation

  for dirname, _, filenames in os.walk(mainPath):

      for filename in filenames:

          path = os.path.join(dirname, filename)

          img = cv2.imread(path, 0) #read each image in dataset

          img = cv2.resize(img, (480, 360))  

            

          if path.find('/NORMAL/') != -1:

              classification = 0

          elif path.find('/PNEUMONIA/') != -1:

              classification = 1

          if path.find('train') != -1:

              X_train.append(img)              

              y_train.append(classification)

            

              if classification == 0:

                  # data augmentation for normal classes

                

                  #Rotation 

                  rows, cols = img.shape

                  Matrix = cv2.getRotationMatrix2D((int(cols/2), int(rows/2)), random.randint(-10, 10) , 1)

                  img_1 = cv2.warpAffine(img,Matrix,(cols,rows))

                  X_train.append(img_1)

                  y_train.append(classification)

                  # shifting in x and y axis

                  Matrix = np.float32([[1,0,random.randint(10, 20)],[0,1,random.randint(10, 20)]])

                  img_2 = cv2.warpAffine(img,Matrix,(cols,rows))



                  X_train.append(img_2)

                  y_train.append(classification)

          elif path.find('test') != -1:

              X_test.append(img)

              y_test.append(classification)

          elif path.find('validation') != -1:

              X_validation.append(img)

              y_validation.append(classification)



def shuffle_data(x, y):

  tmp = list(zip(x, y)) 

  random.shuffle(tmp)

  x, y = zip(*tmp)

  x, y = list(x), list(y)

  del(tmp)

  return x, y



# append data into X_train, y_train, X_test, y_test, X_validation and y_validation lists.



prepareData('/kaggle/input/chest-xray-images-for-classification-pneumonia/pneumonia')

prepareData('/kaggle/input/chest-xray-images-for-classification-pneumonia/pneumonia2')

# Shuffle training dataset

X_train, y_train = shuffle_data(X_train, y_train)



X_train = np.array(X_train).astype('float32')/255

X_test = np.array(X_test).astype('float32')/255

X_validation = np.array(X_validation).astype('float32')/255



y_train = np.array(y_train)

y_test = np.array(y_test)

y_validation = np.array(y_validation)



y_train = to_categorical(y_train)

y_test = to_categorical(y_test)

y_validation = to_categorical(y_validation)



X_train = X_train[:, :, :, np.newaxis]

X_test = X_test[:, :, :, np.newaxis]

X_validation = X_validation[:, :, :, np.newaxis]



#concatenate validation to testing data

X_test = np.concatenate((X_test, X_validation))

del(X_validation)

y_test = np.concatenate((y_test, y_validation))

del(y_validation)
numberofZeroClassified = np.count_nonzero(y_train, axis=0)[0]

numberofOneClassified = np.count_nonzero(y_train, axis=0)[1]

print('number of normal samples is {0} and number of pneumonia samples is {1}'.format(numberofZeroClassified, numberofOneClassified))

print('Ratio of Normal samples over pneumonia samples is ', numberofZeroClassified/ numberofOneClassified)
for i, sample in enumerate(X_train[:4]):

    plt.subplot(2, 2, i+1)



    plt.imshow(cv2.resize(sample, (48, 36)), cmap='gray', interpolation='none')

    plt.title('Class '+ str(y_train[i][0]))

with strategy.scope():

    filters = [32, 64, 128]



    input_layer = Input(shape=(360, 480, 1), name='Input_Layer')

    layer = input_layer



    for filter_size in filters:

        layer = Conv2D(filter_size, kernel_size = 3,strides = 2, activation='relu')(layer)

        layer = BatchNormalization()(layer)

        layer = MaxPool2D()(layer)

        layer = Dropout(.25)(layer)

    layer = Flatten()(layer)

    layer = Dense(512, activation='relu')(layer)

    layer = Dropout(.4)(layer)

    output_layer = Dense(2, activation='softmax')(layer)



    model = Model(input_layer, output_layer)

    model.summary()

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
epochs_number = 30

# check point to save model which has best accuracy

best_accuracy = ModelCheckpoint('best_accuracy.hdf5',monitor='val_accuracy', verbose=0, save_best_only=True ) 

# check point to save model which has best loss  (Lowest loss value)

best_loss = ModelCheckpoint('best_loss.hdf5',monitor='val_loss', verbose=0, save_best_only=True )

history = model.fit(X_train, y_train, batch_size=32, epochs=epochs_number, validation_data=(X_test, y_test), callbacks=[best_accuracy, best_loss])
plt.style.use('ggplot')

plt.plot(range(epochs_number),history.history['val_loss'], c='b' ,label='Validation Loss')

plt.plot(range(epochs_number),history.history['loss'], c='r' ,label='Training Loss')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()

plt.show()
plt.style.use('ggplot')

plt.plot(range(epochs_number),history.history['val_accuracy'], 'b' ,label='Validation Accuracy')

plt.plot(range(epochs_number),history.history['accuracy'], 'r' ,label='Training Accuracy')

plt.xlabel('epochs')

plt.ylabel('Acurracy')

plt.legend()

plt.show()
best_lossModel = load_model('best_accuracy.hdf5')

best_accuracyModel = load_model('best_loss.hdf5')
y = np.argmax(y_test, axis=1) #original test data 

'''

output for softmax is a probaility for which class the input belog to for example

if output is [.2 .8]

.2 for zero class

and .8 for 1 class



np.argmax will return 1 because .8 has index 1

so we can use thic concept to do the inverse operation of categorization

'''

y_best_lossModel = best_lossModel.predict(X_test)

y_best_lossModel = np.argmax(y_best_lossModel, axis=1) 

y_best_accuracyModel = best_accuracyModel.predict(X_test)

y_best_accuracyModel = np.argmax(y_best_accuracyModel, axis=1)
print("Best Loss model confusion Matrix ")

conf_mat_Loss = confusion_matrix(y, y_best_lossModel)

print(conf_mat_Loss)

print("Best Accuracy model confusion Matrix ")

conf_mat_accuracy = confusion_matrix(y, y_best_accuracyModel)

print(conf_mat_accuracy)



print("ROC AUC best accuracy model Score --> ", roc_auc_score(y, y_best_accuracyModel))

print("ROC AUC best less model Score --> ", roc_auc_score(y, y_best_lossModel))



print("F1 Score best accuracy model Score --> ", f1_score(y, y_best_accuracyModel))

print("F1 Score best loss model Score --> ", f1_score(y, y_best_lossModel))