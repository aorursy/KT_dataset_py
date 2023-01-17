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
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.regularizers import l2
class VGG16( Model ):
  def __init__( self, num_classes ):
    super( VGG16, self ).__init__( )
    self.conv1     = Conv2D( 64, kernel_size = (3, 3), padding = 'same', activation = 'relu' )
    self.conv2     = Conv2D( 64, kernel_size = (3, 3), padding = 'same', activation = 'relu' )
    self.max_pool1 = MaxPooling2D( pool_size=( 2, 2 ), strides = (2, 2) )
    self.conv3     = Conv2D( 128, kernel_size = (3, 3), padding = 'same', activation = 'relu' )
    self.conv4     = Conv2D( 128, kernel_size = (3, 3), padding = 'same', activation = 'relu' )
    self.max_pool2 = MaxPooling2D( pool_size=( 2, 2 ), strides = (2, 2) )
    self.conv5     = Conv2D( 256, kernel_size = (3, 3), padding = 'same', activation = 'relu' )
    self.conv6     = Conv2D( 256, kernel_size = (3, 3), padding = 'same', activation = 'relu' )
    self.conv7     = Conv2D( 256, kernel_size = (3, 3), padding = 'same', activation = 'relu' )
    self.max_pool3 = MaxPooling2D( pool_size=( 2, 2 ), strides = (2, 2) )
    self.conv8     = Conv2D( 512, kernel_size = (3, 3), padding = 'same', activation = 'relu' )
    self.conv9     = Conv2D( 512, kernel_size = (3, 3), padding = 'same', activation = 'relu' )
    self.conv10    = Conv2D( 512, kernel_size = (3, 3), padding = 'same', activation = 'relu' )
    self.max_pool4 = MaxPooling2D( pool_size=( 2, 2 ), strides = (2, 2) )
    self.flatten   = Flatten(  ) 
    self.dense1    = Dense( 4096, activation = 'relu' ) 
    self.dropout1  = Dropout( 0.2 )
    self.dense2    = Dense( 4096, activation = 'relu' ) 
    self.dropout2  = Dropout( 0.2 )
    self.dense3    = Dense( num_classes, activation = 'softmax' ) 
    
  def call( self, x ):
    x = self.max_pool1( self.conv2( self.conv1( x ) ) )
    x = self.max_pool2( self.conv4( self.conv3( x ) ) )
    x = self.max_pool3( self.conv7( self.conv6( self.conv5( x ) ) ) )
    x = self.max_pool4( self.conv10( self.conv9( self.conv8( x ) ) ) ) 
    x = self.flatten( x )
    x = self.dense3( self.dropout2 ( self.dense2( self.dropout1( self.dense1( x ) ) ) ) )
    return x
# load data
train = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")
test = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_test.csv")
train.head()
test.head()
# get data and transform to numpy
y_test = (test.iloc[:,0]).to_numpy()
X_test = (test.iloc[:,1:]).to_numpy()
y_train = (train.iloc[:,0]).to_numpy()
X_train = (train.iloc[:,1:]).to_numpy()
num_classes = 10
img_width = 28
img_heigh = 28
img_ch = 1
input_shape = (img_width, img_heigh, img_ch)

# normalize data
X_train, X_test = X_train / 255, X_test / 255

# reshape input 
X_train = X_train.reshape(X_train.shape[0], *input_shape)
X_test = X_test.reshape(X_test.shape[0], *input_shape)

# one-hot
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
print('Train shape: {}, {}'.format(X_train.shape,y_train.shape))
print('Test shape: {}, {}'.format(X_test.shape,y_test.shape))
# compile model
model = VGG16(num_classes)
model.compile(  tf.keras.optimizers.Adam(learning_rate=1e-4), loss = 'categorical_crossentropy',
                metrics = 'accuracy' )

# create callbacks
callbacks = [ 
  tf.keras.callbacks.ModelCheckpoint('best_model', monitor='val_accuracy', verbose=1, save_best_only=True),
  tf.keras.callbacks.ReduceLROnPlateau( factor = 0.1, patience = 3, min_lr = 0.00001, verbose = 1 )
 ]

history = model.fit(X_train, y_train, epochs = 25, batch_size = 256,
                    callbacks = callbacks, verbose = 1,
                    validation_data = (X_test,y_test) )
# Diffining Figure
f = plt.figure(figsize=(20,7))

#Adding Subplot 1 (For Accuracy)
f.add_subplot(121)

plt.plot(history.epoch,history.history['accuracy'],label = "accuracy") # Accuracy curve for training set
plt.plot(history.epoch,history.history['val_accuracy'],label = "val_accuracy") # Accuracy curve for validation set

plt.title("Accuracy Curve",fontsize=18)
plt.xlabel("Epochs",fontsize=15)
plt.ylabel("Accuracy",fontsize=15)
plt.grid(alpha=0.3)
plt.legend()

#Adding Subplot 1 (For Loss)
f.add_subplot(122)

plt.plot(history.epoch,history.history['loss'],label="loss") # Loss curve for training set
plt.plot(history.epoch,history.history['val_loss'],label="val_loss") # Loss curve for validation set

plt.title("Loss Curve",fontsize=18)
plt.xlabel("Epochs",fontsize=15)
plt.ylabel("Loss",fontsize=15)
plt.grid(alpha=0.3)
plt.legend()

plt.show()
scores = model.evaluate(X_test, y_test)