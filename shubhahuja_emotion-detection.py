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
import os
import matplotlib.pyplot as plt
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils
from tensorflow.keras import datasets, layers, models
path = '/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/'
os.listdir(path)
data = pd.read_csv(path+'icml_face_data.csv')

def data_prep(data):

    
    image_array = np.zeros(shape=(len(data), 48, 48))
    image_label = np.array(list(map(int, data['emotion'])))
    
    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, ' pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image
        
    return image_array, image_label
train_image, train_label = data_prep(data[data[' Usage']=='Training'])
val_image, val_label = data_prep(data[data[' Usage']=='PrivateTest'])
test_image, test_label = data_prep(data[data[' Usage']=='PublicTest'])


train_images = train_image.reshape((train_image.shape[0], 48, 48, 1))
train_images = train_images/255

val_images = val_image.reshape((val_image.shape[0], 48, 48, 1))
val_images = val_images/255

test_images = test_image.reshape((test_image.shape[0], 48, 48, 1))
test_images = test_images/255


model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64,kernel_size= (5, 5), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))
model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128,kernel_size= (3, 3), activation='relu'))
model.add(Conv2D(128,kernel_size= (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))


model.add(layers.Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))
model.summary()


train_label=np_utils.to_categorical(train_label, num_classes=7)
test_label=np_utils.to_categorical(test_label, num_classes=7)
val_label=np_utils.to_categorical(val_label, num_classes=7)
model.compile(loss=categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
history=model.fit(train_images, train_label,
          batch_size=128,
          epochs=50,
          verbose=1,
          validation_data=(test_images, test_label),
          shuffle=True)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()