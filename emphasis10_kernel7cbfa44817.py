from keras.utils import np_utils

from keras.models import Sequential, load_model

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.layers import BatchNormalization

from keras import applications



import matplotlib.pyplot as plt

import os, platform, time

import os.path

from os import walk

import numpy as np

import pandas as pd

import random

import cv2

from PIL import Image, ImageDraw

import keras

from keras.utils import to_categorical

from keras.optimizers import SGD

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import preprocess_input

import cv2

import random

from sklearn import preprocessing



def draw_it(raw_strokes):

    image = Image.new("P", (255,255), color=255)

    image_draw = ImageDraw.Draw(image)



    for stroke in eval(raw_strokes):

        for i in range(len(stroke[0])-1):



            image_draw.line([stroke[0][i], 

                             stroke[1][i],

                             stroke[0][i+1], 

                             stroke[1][i+1]],

                            fill=0, width=6)

    return np.array(image)



#DATA_PATH = "D:\\juptest\\datasets\\train_simplified"



if platform.system() == 'Linux':

    #from google.colab import drive #for mounting google drive

    #drive.mount('/content/gdrive')

    DATA_PATH = '/kaggle/input/quickdraw-doodle-recognition/train_simplified/'

else:

    DATA_PATH = "D:\\juptest\\datasets\\train_simplified\\"
#os.chdir('D:\\juptest\\datasets\\')

FILE_PATH = [] #now 10 files

for (dirpath, dirnames, filenames) in walk(DATA_PATH):

    FILE_PATH.extend(filenames)

    break



word_set = []

for i in FILE_PATH:

  word_set.append(i[0:-4])



IMG_WIDTH = 48

IMG_HEIGHT = 48

IMG_DEPTH = 3

BATCH_SIZE = 32

ext_num = 600
model = applications.ResNet50V2(include_top=True, weights=None, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), pooling='max', classes=len(word_set))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



le = preprocessing.LabelEncoder()

le.fit(word_set)
for i in range(10):

  train_set = [] #set of X

  label_set = [] #set of Y



  ext_class = random.sample(range(len(FILE_PATH)), int(0.42*len(FILE_PATH)))



  for i in ext_class:

    thisfile = pd.read_csv(DATA_PATH + FILE_PATH[i])

    #word_set.append(thisfile['word'][0])

    extlist = random.sample(range(len(thisfile)), ext_num)

    for j in extlist:

        train_set.append(cv2.resize(draw_it(thisfile['drawing'][j]), dsize=(IMG_WIDTH,IMG_HEIGHT)))

        label_set.append(thisfile['word'][j])



  

  label_set = le.transform(label_set)



  train_set = np.array(train_set)

  label_set = np.array(label_set)

  X_train, X_validation, Y_train, Y_validation = train_test_split(train_set, label_set, test_size = 0.2, random_state=223)



  X_train = X_train.astype('float32') / 255

  X_validation = X_validation.astype('float32') / 255



  X_train = np.stack([X_train] * 3, axis = 3)

  X_validation = np.stack([X_validation] * 3, axis = 3)



  Y_train = np_utils.to_categorical(Y_train, len(word_set))

  Y_validation = np_utils.to_categorical(Y_validation, len(word_set))

 



  history = model.fit(X_train, Y_train,

            epochs=4,

            batch_size=BATCH_SIZE,

            validation_data=(X_validation, Y_validation))

  #print('\nAccuracy: {:.4f}'.format(model.evaluate(validation_data, validation_labels)[1]))



  y_vloss = history.history['val_loss']

  y_loss = history.history['loss']



  x_len = np.arange(len(y_loss))

  plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")

  plt.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")



  plt.legend(loc='upper right')

  plt.grid()

  plt.xlabel('epoch')

  plt.ylabel('loss')

  plt.show()
test_X = [] #set of X

test_Y = [] #set of Y



thisfile = pd.read_csv('/kaggle/input/quickdraw-doodle-recognition/test_simplified.csv')



for j in range(len(thisfile)):

    test_X.append(cv2.resize(draw_it(thisfile['drawing'][j]), dsize=(IMG_WIDTH,IMG_HEIGHT)))

    #test_Y.append(thisfile['word'][j])


test_X = np.array(test_X)

#test_Y = np.array(test_Y)



#X_train, X_validation, Y_train, Y_validation = train_test_split(text_X, test_Y, test_size = 0.2, random_state=223)



test_X = test_X.astype('float32') / 255

test_X = np.stack([test_X] * 3, axis = 3)





test_Y = model.predict(test_X)

print(len(test_Y))

X = []

for i in range(len(test_Y)):

    maxv = -1

    idx = 0

    for j in range(len(test_Y[0])):

        if maxv < test_Y[i][j]:

            maxv = test_Y[i][j]

            idx = j

    X.append(idx)
y = le.inverse_transform(X)

ans = []

for i in range(len(thisfile['key_id'])):

    ans.append([thisfile['key_id'][i], y[i]])
ans = pd.DataFrame(ans)

ans.columns = ['key_id', 'word']

ans.to_csv('/kaggle/working/ans.csv', index=False)