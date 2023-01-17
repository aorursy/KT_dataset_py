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
import cv2
import tensorflow as tf



import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from keras.layers import Dense, Activation, Dropout, Flatten

from tensorflow.keras.layers import BatchNormalization



from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import array_to_img

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier 

from sklearn.model_selection import GridSearchCV



from sklearn.pipeline import make_pipeline
 

width = 64

height = 64

dim = (width, height)

 
test_file='../input/challenges-in-representation-learning-facial-expression-recognition-challenge/test.csv'

train_file='../input/challenges-in-representation-learning-facial-expression-recognition-challenge/train.csv'

label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

names_train=['emotion','pixels']

names_test=['pixels']

df_test=pd.read_csv('../input/challenges-in-representation-learning-facial-expression-recognition-challenge/test.csv',names=names_test, na_filter=False)

#im=df['pixels']

df_test=df_test.drop([0],axis=0)

df_test.head(10)

df_train=pd.read_csv('../input/challenges-in-representation-learning-facial-expression-recognition-challenge/train.csv',names=names_train, na_filter=False)

#im=df['pixels']

df_train=df_train.drop([0],axis=0)

df_train.head(10)

df_train=df_train.drop([1],axis=0)



def gray_to_rgb(im):

  '''

  converts images from single channel images to 3 channels

  '''



  w, h = im.shape

  ret = np.empty((w, h, 3), dtype=np.uint8)

  ret[:, :, 2] =  ret[:, :, 1] =  ret[:, :, 0] =  im

  return ret



def convert_to_image(pixels, mode="save", t="gray"):



  if type(pixels) == str:

      pixels = np.array([int(i) for i in pixels.split()])

  if mode == "show":

    if t == "gray":

      return pixels.cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    else:

      return gray_to_rgb(pixels.cv2.resize(img, dim, interpolation = cv2.INTER_AREA))

  else:

      return pixels

df_train["pixels"] = df_train["pixels"].apply(lambda x : convert_to_image(x, mode="show", t="gray"))

df_test["pixels"] = df_test["pixels"].apply(lambda x : convert_to_image(x, mode="show", t="gray"))
X_train, X_val, y_train, y_val = train_test_split(df_train["pixels"],  df_train["emotion"], test_size=0.2, random_state=1)



X_train = np.array(list(X_train[:]), dtype=np.float)

X_val = np.array(list(X_val[:]), dtype=np.float)



y_train = np.array(list(y_train[:]), dtype=np.float)

y_val = np.array(list(y_val[:]), dtype=np.float)



X_train = X_train.reshape(X_train.shape[0], 48, 48, 1) 

X_val = X_val.reshape(X_val.shape[0], 48, 48, 1)
X_train.shape
X_test=np.array(list(df_test['pixels']), dtype=np.float)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
X_val.shape
X_test.shape
IMG_SIZE=48
model = Sequential()



model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', 

                 input_shape=(IMG_SIZE,IMG_SIZE,1)))

model.add(BatchNormalization(axis=1))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(BatchNormalization(axis=1))

model.add(MaxPooling2D(pool_size=(2, 2)))



#model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

#model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(BatchNormalization(axis=1))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

model.add(BatchNormalization(axis=1))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))



opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

model.summary()
estimators = []

estimators.append(('standardize', StandardScaler()))

estimators.append(('mlp', KerasRegressor(build_fn=model, epochs=50, batch_size=5, verbose=0)))

pipeline = Pipeline(estimators)

kfold = KFold(n_splits=10)

results = cross_val_score(pipeline, X_train, y_train, cv=kfold)

print("Standard: %.2f (%.2f) MSE" % (results.mean(), results.std()))
plt.plot(custom.history['loss'])

plt.plot(custom.history['val_loss'])

plt.title("Model Loss")

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(['Train', 'Test'])

plt.show()
plt.plot(custom.history['accuracy'])

plt.plot(custom.history['val_accuracy'])

plt.title("Model Accuracy")

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(['Train', 'Test'])

plt.show()





