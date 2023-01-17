# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns



import tensorflow as tf

from keras.utils.np_utils import to_categorical

from tensorflow.keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



%matplotlib inline

sns.set(style='white', context='notebook',palette='deep')
df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

df.head()

X = df.drop(['label'],axis=1)

Y = df['label']
Y.value_counts()
sns.countplot(Y)
X.isnull().any().describe()
X = X / 255.0

test = test / 255.0

X = np.array(X)

test = np.array(test)

print(type(X))
X2 = X.reshape(-1,28,28,1)

test = test.reshape(-1,28,28,1)

print(X2)
Y2 = to_categorical(Y, num_classes=10)

print(Y)
[X_train, X_test, Y_train, Y_test] = train_test_split(X2, Y2, test_size = 0.1, random_state = 3)
print(X_train.shape[:])

print(Y_train.shape[:])

print(X_test.shape[:])

print(Y_test.shape[:])
model = tf.keras.models.Sequential([

    

    tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu', input_shape=(28,28,1)),

    tf.keras.layers.Conv2D(32, (3,3), padding='same',activation='relu'),

    tf.keras.layers.MaxPool2D((2,2)),

    tf.keras.layers.Dropout(0.25),

    

    tf.keras.layers.Conv2D(32, (3,3),padding='same', activation='relu'),

    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),

    tf.keras.layers.MaxPool2D((2,2), strides=(2,2)),

    tf.keras.layers.Dropout(0.35),

    

    tf.keras.layers.Conv2D(64,(3,3), padding='same',activation='relu'),

    tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),

    tf.keras.layers.Conv2D(256,(3,3), padding='same', activation='relu'),

    tf.keras.layers.MaxPool2D((2,2),),

    tf.keras.layers.Dropout(0.50),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dropout(.50),

    tf.keras.layers.Dense(10, activation='softmax'),

    

])



optimizer = RMSprop(lr=0.0008,rho=0.912, epsilon=1e-08,decay=0.0001)



model.compile(loss = 'categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()
learning_rate_reduction = ReduceLROnPlateau(moniter='val_acc', patience = 3,verbose=2, factor=0.5, min_lr=0.00001)
epochs = 3

batch_size = 78
datagen = ImageDataGenerator(

    featurewise_center= False,

    samplewise_center = False,

    featurewise_std_normalization = False,

    samplewise_std_normalization = False,

    rotation_range = 9,

    zoom_range = 0.15,

    width_shift_range = 0.1,

    height_shift_range = 0.1,

    horizontal_flip = False,

    vertical_flip = False

)

datagen.fit(X_train)
history1 = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs,

                   validation_data = (X_test, Y_test), verbose=2)
history2 = model.fit_generator(datagen.flow(X_train, Y_train ,batch_size=batch_size),

                             epochs=epochs, 

                             validation_data = (X_test, Y_test),

                             verbose = 2, 

                             steps_per_epoch = X_train.shape[0]//batch_size,

                             callbacks = [learning_rate_reduction])
figure , ax = plt.subplots(2,1)

ax[0].plot(history.history['accuracy'], color='g', label = 'Training_accuracy')

ax[0].plot(history.history['val_accuracy'], color='b',label = 'validation_accuracy')



ax[0].legend(loc='best', shadow = True)



ax[1].plot(history.history['loss'], color = 'g', label = 'Training_loss')

ax[1].plot(history.history['val_loss'], color = 'b' , label = 'validation_loss')

ax[1].legend(loc='best', shadow = False)