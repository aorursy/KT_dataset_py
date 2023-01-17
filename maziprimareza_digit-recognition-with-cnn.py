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
digits = np.array(pd.read_csv('/kaggle/input/digit-recognizer/train.csv'))
digit_images = digits[:,1:].reshape(digits.shape[0],28,28)

digit_targets = digits[:,0]
import matplotlib.pyplot as plt



plt.gray()

fig,axes = plt.subplots(2,5, figsize=(20,10))



for i in range(0,2):

    for t in range(0,5):

        r = np.random.randint(1, digits[1:,:].shape[0])

        axes[i,t].matshow(digit_images[r])

        axes[i,t].set_title('Actual digit: {}'.format(digit_targets[r]))

        axes[i,t].yaxis.set_major_locator(plt.NullLocator())

        axes[i,t].xaxis.set_major_formatter(plt.NullFormatter())

plt.show()
import numpy as np

import pandas as pd



import tensorflow as tf

import keras

from keras.models import Sequential

from keras.layers.core import Dense, Activation, Flatten

from keras.callbacks import ModelCheckpoint

from keras.callbacks import EarlyStopping

from keras.models import Sequential

from keras.layers.core import Dense, Activation, Flatten

from keras.optimizers import Adam

from keras.layers import Conv1D, Conv2D, MaxPooling2D

from datetime import datetime

import matplotlib.pyplot as plt



physical_devices = tf.config.experimental.list_physical_devices('GPU')

print("physical_devices-------------", len(physical_devices))

tf.config.experimental.set_memory_growth(physical_devices[0], True)
X = digits[:,1:].reshape(digits.shape[0],1,28,28)

y = np.array(pd.get_dummies(digits[:,0]))
cnn = Sequential()



cnn.add(Conv2D(32, kernel_size=(8, 8),activation='relu',input_shape=(1,28,28), padding='same'))



cnn.add(Conv2D(64, (8, 8), activation='relu', padding='same'))



cnn.add(MaxPooling2D(pool_size=(1, 10)))

cnn.add(Flatten())

cnn.add(Dense(128, activation='sigmoid'))

cnn.add(Dense(y.shape[1]))

cnn.compile(loss="mse", optimizer="adam", metrics=["accuracy"])



cnn.summary()
from datetime import datetime



tic = datetime.now()

J = cnn.fit(X,y,validation_split=0.001,verbose=0,epochs = 50, batch_size=128)

toc = datetime.now()

print("Time needed for training: ", toc-tic)
score = cnn.evaluate(X,y, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
yhat= cnn.predict(X)

plt.gray()

fig,axes = plt.subplots(2,5, figsize=(20,10))



for i in range(0,2):

    for t in range(0,5):

        r = np.random.randint(1, digits[1:,:].shape[0])

        axes[i,t].matshow(digit_images[r])

        axes[i,t].set_title('Predict: {} vs Actual: {}' .format(np.argmax(yhat[r]), np.argmax(y[r])))

        axes[i,t].yaxis.set_major_locator(plt.NullLocator())

        axes[i,t].xaxis.set_major_formatter(plt.NullFormatter())

plt.show()
plt.plot(J.history['loss'],label = 'Training Loss')

plt.plot(J.history['val_loss'],label = 'Validation Loss')

plt.legend()

plt.show()
plt.plot(J.history['accuracy'],label = 'Training Accuracy')

plt.plot(J.history['val_accuracy'],label = 'Validation Accuracy')

plt.legend()

plt.show()
test = np.array(pd.read_csv('/kaggle/input/digit-recognizer/test.csv')) #read the test data
digit_images_test = test.reshape(test.shape[0],28,28) #reshape the data to show the image
X_test = test.reshape(test.shape[0],1,28,28)

y_pred = cnn.predict(X_test)
plt.gray()

fig,axes = plt.subplots(2,5, figsize=(20,10))



for i in range(0,2):

    for t in range(0,5):

        r = np.random.randint(1, test[1:,:].shape[0])

        axes[i,t].matshow(digit_images_test[r])

        axes[i,t].set_title('Predict: {}' .format(np.argmax(y_pred[r])))

        axes[i,t].yaxis.set_major_locator(plt.NullLocator())

        axes[i,t].xaxis.set_major_formatter(plt.NullFormatter())

plt.show()
y_result=[]

for yy in y_pred:

    y_result.append(np.argmax(yy))
#save the result to csv file

result = pd.DataFrame(y_result)

result = pd.DataFrame(

    {"ImageId": result.index+1,

     "Label": y_result

    })

result.to_csv('result.csv', index=False)
result