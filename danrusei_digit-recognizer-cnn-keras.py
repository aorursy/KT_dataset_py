# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Check if GPU is avialable

import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
%matplotlib inline

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense , Dropout , Lambda, Flatten, Conv2D
from tensorflow.python.keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as K

img_rows, img_cols = 28, 28
num_classes = 10

train = pd.read_csv("../input/train.csv")
print(train.shape)
train.head()
test= pd.read_csv("../input/test.csv")
print(test.shape)
test.head()
X = train.iloc[:,1:].values.astype('float32') # all pixel values
y = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits
X_test = test.values.astype('float32')
# apply one hot encoding for label

from keras.utils.np_utils import to_categorical
y = to_categorical(y)
num_classes = y.shape[1]
num_classes
#Convert train datset to (num_images, img_rows, img_cols, colour channel) format 
X = X.reshape(X.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
# split initial to train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42)
#Fit the model using Data Augemnetation, will improve accuracy of the model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
data_generator_with_aug = ImageDataGenerator(rotation_range=8,
                                             width_shift_range = 0.08,
                                             height_shift_range = 0.08)
            
data_generator_no_aug = ImageDataGenerator()
# apply data augementation to train and validation datasete

train_generator = data_generator_with_aug.flow(X_train, y_train,batch_size=64)

validation_generator = data_generator_no_aug.flow(X_val, y_val, batch_size=64)

# Strides as an option to MaxPooling
# Dropout to combat overfitting

digit_model = Sequential()
digit_model.add(Conv2D(24, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
digit_model.add(Dropout(0.5))
digit_model.add(Conv2D(24, kernel_size=(3, 3), strides=2, activation='relu'))
digit_model.add(Dropout(0.5))
digit_model.add(Conv2D(24, kernel_size=(3, 3), strides=2, activation='relu'))
digit_model.add(Dropout(0.5))
digit_model.add(Flatten())
digit_model.add(Dense(128, activation='relu'))
digit_model.add(Dense(num_classes, activation='softmax'))
print("input shape ",digit_model.input_shape)
print("output shape ",digit_model.output_shape)
# Your code to compile the model in this cell
digit_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# Your code to fit the model here
with tf.device("/device:GPU:0"):
    history = digit_model.fit_generator(
        train_generator,
        epochs=3,
        validation_data=validation_generator,
        validation_steps=1)
import matplotlib.pylab as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')

plt.legend()

plt.show()
gen = ImageDataGenerator(rotation_range=8,
                        width_shift_range = 0.08,
                        height_shift_range = 0.08)
batches = gen.flow(X, y, batch_size=64)
with tf.device("/device:GPU:0"):
    history_subm=digit_model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3)
predictions = digit_model.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("DigitRecognizer.csv", index=False, header=True)