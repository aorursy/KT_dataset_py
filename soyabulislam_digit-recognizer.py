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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical
train_set= pd.read_csv("../input/digit-recognizer/train.csv")
train_set
test=pd.read_csv("../input/digit-recognizer/test.csv")
test
sub=pd.read_csv("../input/digit-recognizer/sample_submission.csv")
sub
X= train_set.drop('label', axis=1)
y=train_set['label']
print(X)
print(y)
print(X.shape)
print(y.shape)
X = X/255.0
test =test/255.0
X= X.values.reshape(-1,28,28,1)
test= test.values.reshape(-1,28,28,1)
test= to_categorical(test, num_classes=10)
X_train, X_test, y_train, y_val= train_test_split(X,y, test_size=0.15, random_state=42)

fig = plt.figure()
for i in range(30):
  plt.subplot(10,3,i+1)
  plt.tight_layout()
  plt.imshow(X_train[i][:,:,0], cmap='gray', interpolation='none')
  #plt.title("Digit: {}".format(y_train[i][:,:,0]))
  plt.xticks([])
  plt.yticks([])



model.summary()

def digit_recognizer():
    import tensorflow as tf
    model=tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16,(3,3), activation='relu', input_shape=(28,28,1),padding="same"),
        tf.keras.layers.BatchNormalization( scale=True),
        tf.keras.layers.Conv2D(16,(3,3), activation='relu'),
        tf.keras.layers.BatchNormalization( scale=True),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization( scale=True),
        tf.keras.layers.Conv2D(32,(3,3), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization( scale=True),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization( scale=True),
        tf.keras.layers.Conv2D(64,(3,3), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization( scale=True),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation='sigmoid')
        
    ])
    
    from tensorflow.keras.optimizers import RMSprop

    #model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False,
                                       featurewise_std_normalization=False, samplewise_std_normalization=False,
                                       zca_whitening=False, zca_epsilon=1e-06, rotation_range=10, width_shift_range=0.1,
                                       height_shift_range=0.1, brightness_range=None,  zoom_range=0.1,
                                       channel_shift_range=0.1, fill_mode='nearest',  horizontal_flip=False,
                                       vertical_flip=False, rescale=None )
    history=model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=80), epochs=30,validation_data=(X_test,y_val),
                                                   verbose=2, steps_per_epoch=X_train.shape[0]//80
                                            )
    return history

digit_recognizer()
