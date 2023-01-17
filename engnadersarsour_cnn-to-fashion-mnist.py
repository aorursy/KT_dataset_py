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

from tensorflow import keras

import matplotlib.pyplot as plt

tf.__version__

from keras.utils import np_utils

import numpy as np

from keras.models import Sequential

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization

from keras.utils import to_categorical

from keras.callbacks import LearningRateScheduler

from sklearn.model_selection import train_test_split

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()
print('train_x shape => ', train_x.shape)

print('train_y shape => ', train_y.shape)

print('')

print('test_x shape => ', test_x.shape)

print('test_y shape => ', test_y.shape)
# Reshaping the data

train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)

test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)

print('train_shape => ', train_x.shape)

print('test_shape => ', test_x.shape)
# Normalizing data

train_x = train_x/255

test_x = test_x/255

train_y = to_categorical(train_y)

test_y = to_categorical(test_y)
model1 = Sequential()

from keras.layers import LeakyReLU

model1.add(Conv2D(32, kernel_size=(3, 3), padding = 'same' ,kernel_initializer='he_normal', input_shape=(28, 28, 1),name = 'conv1'))

model1.add(LeakyReLU(alpha = 0.2))

model1.add(MaxPooling2D((2, 2),name='pool1'))

model1.add(Dropout(0.25, name = 'dropout1'))

model1.add(BatchNormalization(name='batchnorm1'))

model1.add(Conv2D(64, (3, 3), padding = 'same', name='conv2'))

model1.add(LeakyReLU(alpha = 0.2))

model1.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))

model1.add(Dropout(0.25, name='dropout2'))

model1.add(BatchNormalization(name='batchnorm2'))

model1.add(Conv2D(128, (3, 3), padding = 'same', name='conv3'))

model1.add(LeakyReLU(alpha = 0.2))

model1.add(MaxPooling2D(name='pool3'))

#model1.add(Dropout(0.2, name='dropout3'))

model1.add(BatchNormalization(name='batchnorm3'))

model1.add(Flatten())

model1.add(Dense(512, name='dense0', activation = 'relu'))

model1.add(Dense(128, name='dense1', activation = 'relu'))

#model1.add(Dropout(0.3, name='dropout4'))

model1.add(Dense(10, activation='softmax', name='output'))
model1.compile(loss=keras.losses.categorical_crossentropy,

               optimizer=keras.optimizers.Adagrad(learning_rate=0.05),

               metrics=['accuracy'])
model1.summary()
from keras.callbacks import ModelCheckpoint

#callback = ModelCheckpoint('checkpoint.h5', save_best_only = True, verbose=1)

history =model1.fit(train_x, train_y, batch_size = 1000, epochs=80, validation_data = (test_x, test_y), verbose = 1)
plt.plot(history.history['accuracy'], label='accuracy')

plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.ylim([0.5, 1])

plt.legend(loc='lower right')



test_loss, test_acc = model1.evaluate(test_x, test_y, verbose=2)
test_loss, test_acc = model1.evaluate(test_x,  test_y, verbose=2)

print('\nTest accuracy:', test_acc)
probability_model1 = tf.keras.Sequential([model1, 

                                   tf.keras.layers.Softmax()])

predictions = probability_model1.predict(test_x)

preds = pd.Series(np.argmax(model1.predict(test_x)).tolist())

out_df = pd.DataFrame({

    'Id': pd.Series(range(1,28001)),

    'target': preds

})

out_df
out_df.to_csv('my_submission.csv', index=False)