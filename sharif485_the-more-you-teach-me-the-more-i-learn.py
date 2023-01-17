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
import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn import metrics

import time

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization, LeakyReLU

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.callbacks import LearningRateScheduler

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler



import matplotlib.pyplot as plt
train_data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test_data = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

dig_data = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")
print("Train set shape = " +str(train_data.shape))

print("Test set shape = " +str(test_data.shape))

print("Dif set shape = " +str(dig_data.shape))
val_count=train_data.label.value_counts()

sns.barplot(val_count.index, val_count)
#Slicing and Reshaping training images

data_train = train_data.iloc[:,1:].values 

x_train = data_train.reshape(data_train.shape[0], 28, 28, 1)

#Slicing training labels and applying one hot encoding

train_label = train_data.iloc[:,0].values 

y_train = tf.keras.utils.to_categorical(train_label, 10)

print(x_train.shape, y_train.shape)
#Slicing and Reshaping validation images

data_val=dig_data.drop('label',axis=1).iloc[:,:].values

x_val = data_val.reshape(data_val.shape[0], 28, 28,1)

#Slicing validation labels and applying one hot encoding

val_label=dig_data.label

y_val = tf.keras.utils.to_categorical(val_label, 10) 

print(x_val.shape, y_val.shape)
#procesing  test data

x_test=test_data.drop('id', axis=1).iloc[:,:].values

x_test = x_test.reshape(x_test.shape[0], 28, 28,1)

x_test.shape
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation = 'relu', input_shape = (28,28,1)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.2))



model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'Same', activation = 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'Same', activation = 'relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(64, activation = 'relu'))

model.add(Dense(32, activation = 'relu'))

model.add(Dropout(0.3))

model.add(Dense(10, activation = 'softmax'))
model.summary()
input_shape = (28, 28, 1)

learning_rate=0.001

batch_size = 512

epochs = 280

optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
def lr_decay(epoch):

    return learning_rate * 0.99 ** epoch
train_data_generator = ImageDataGenerator(rescale = 1./255.,

                                        rotation_range = 20,

                                        width_shift_range = 0.1,

                                        height_shift_range = 0.1,

                                        shear_range = 0.1,

                                        zoom_range = [0.2, 1.2],

                                        horizontal_flip = False)

val_data_generator = ImageDataGenerator(rescale=1./255)
model.compile(optimizer=optimizer, loss=['categorical_crossentropy'], metrics=['accuracy'])
#Featching the start time and tracking it to check if the training hour passed more than 2 hours or not

start = time.time()

time_spent = 0

accuracy = []

val_accuracy = []

for epoch in range (220):

    time_spent = time.time()-start

    if time_spent >= 7150:  # as training time limit >=2 hours (2 * 60 * 60 = 7200s)

        break

    else:

        epoch += 1

        print('epoch:', epoch)

        history = model.fit_generator(

          train_data_generator.flow(x_train,y_train, batch_size=batch_size),

          steps_per_epoch=100,

          epochs=1,

          callbacks=[LearningRateScheduler(lr_decay)],

          validation_data=val_data_generator.flow(x_val,y_val),

          validation_steps=50,  

          verbose=1, shuffle = True)

        accuracy.append(history.history['accuracy'])

        val_accuracy.append(history.history['val_accuracy'])



def plot_accuracy(accuracy, val_accuracy):

    epochs = range(len(accuracy))

    plt.plot(epochs, accuracy, 'r')

    plt.plot(epochs, val_accuracy, 'b')

    plt.title('Training accuracy')

    plt.legend(['train', 'val'], loc='lower right')

    plt.grid()

    plt.show()
plot_accuracy(accuracy, val_accuracy)
x_test = x_test/255



predictions = model.predict_classes(x_test)

submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

submission['label'] = predictions

submission.head()
submission.to_csv("submission.csv",index=False)
print('end...')