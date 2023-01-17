# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import sweetviz as sv

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/digit-recognizer/train.csv')

test_data = pd.read_csv('../input/digit-recognizer/test.csv')

print(train.info())
print(np.shape(train))
X_train = train.iloc[:,1:]

y_train = train.iloc[:,0]
X_train = X_train/255.0

test_data = test_data/255.0

X_train = X_train.values.reshape(-1,28,28,1)

test_data = test_data.values.reshape(-1,28,28,1)
X_train.shape,test_data.shape
import tensorflow as tf

from tensorflow import keras

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Lambda, Flatten, Dense

from keras.utils.np_utils import to_categorical 
y_train = to_categorical(y_train,num_classes=10)
y_train
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
from matplotlib import pyplot as plt

plt.imshow(X_train[0][:,:,0], cmap='gray')
model = Sequential()

model.add(Conv2D(32, (5,5),padding='Same', activation='relu', kernel_initializer='glorot_uniform', input_shape=(28, 28, 1)))

model.add(Conv2D(32, (5,5),padding='Same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

          

model.add(Conv2D(64, (3,3), padding='Same', activation='relu'))

model.add(Conv2D(64, (3,3), padding='Same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

          

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
model.summary()
from keras.optimizers import RMSprop

opt = RMSprop(lr=0.001)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img

import itertools
datagen = ImageDataGenerator(rotation_range=10, zoom_range = 0.1, width_shift_range=0.1, height_shift_range=0.1)

datagen.fit(X_train)

history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=86),

                              epochs = 30, validation_data = (X_test,y_test),verbose = 2,

                              steps_per_epoch=X_train.shape[0]/86)
# fitted = model.fit(X_train,y_train,epochs=20,batch_size=32,validation_data=(X_test, y_test))

fig, ax = plt.subplots()





ax.plot(history.history['accuracy'], color='b', label="Training accuracy")

ax.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax.legend(loc='best', shadow=True)
pred = model.predict(test_data)

pred
prediction = np.argmax(pred, axis = 1)

prediction
submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

submission['Label'] = prediction

submission.head(10)
submission.to_csv("submission.csv", index=False, header=True)