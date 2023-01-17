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
import tensorflow as tf

from sklearn.model_selection import train_test_split

import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.layers import LeakyReLU

from tensorflow.keras.callbacks import LearningRateScheduler

from tensorflow.keras.callbacks import EarlyStopping

from keras.optimizers import Adam

import matplotlib.pyplot as plt
train = pd.read_csv("../input/Kannada-MNIST/train.csv")

test = pd.read_csv("../input/Kannada-MNIST/test.csv")

DigMNIST = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")

samplesubmission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")
print(train.shape)

train.head()
print(test.shape)

test.head()
print(DigMNIST.shape)

DigMNIST.head()
X=train.iloc[:,1:].values 

Y=train.iloc[:,0].values 

Y[:10]
X = X.reshape(X.shape[0], 28, 28,1) 

Y = keras.utils.to_categorical(Y, 10) 

x_test=test.drop('id', axis=1).iloc[:,:].values

x_test = x_test.reshape(x_test.shape[0], 28, 28,1)

x_test.shape
dig_x=DigMNIST.drop('label',axis=1).iloc[:,:].values

dig_x = dig_x.reshape(dig_x.shape[0], 28, 28,1)

dig_x.shape
dig_y=DigMNIST.label

dig_y.shape
y_dig=DigMNIST.label

y_dig.shape
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.10, random_state=42) 

def lr_decay(epoch):

    return initial_learningrate * 0.99 ** epoch
main_datagen = ImageDataGenerator(

        featurewise_center=False, 

        samplewise_center=False,  

        rotation_range=8,  

        zoom_range = 0.25, 

        width_shift_range=0.25,  

       featurewise_std_normalization=False,  

        samplewise_std_normalization=False,  

        zca_whitening=False,

        height_shift_range=0.15, 

        horizontal_flip=False,  

        vertical_flip=False)  

main_datagen.fit(X_train)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (3,3), padding='same', input_shape=(28, 28, 1)),

    tf.keras.layers.BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(64,  (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(128, (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(128, (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.2),    

    tf.keras.layers.Conv2D(256, (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(256, (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(10, activation='softmax')

])

model.summary()
class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):

    if(logs.get('accuracy')>1.000):

      print("\Reached 100 % accuracy so cancelling training!")

      self.model.stop_training= True

callbacks=myCallback()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

gaurav= model.fit(X, Y, batch_size=100, epochs=3, callbacks=[callbacks] )
from sklearn import metrics
plt.plot(gaurav.history['accuracy'],label='train')

plt.plot(gaurav.history['accuracy'],label='validation')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend()
val_loss,val_acc = model.evaluate(X,Y)
predictions = model.predict_classes(x_test/255.)
samplesubmission['label'] = predictions
samplesubmission.head()
samplesubmission.to_csv("submission.csv",index=False)