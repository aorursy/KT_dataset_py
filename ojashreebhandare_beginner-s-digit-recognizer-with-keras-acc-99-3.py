# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Flatten, Dense, Dropout

import time



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import RMSprop, Adam

from keras.callbacks import ReduceLROnPlateau



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools

from keras.utils.np_utils import to_categorical

from keras.callbacks import LearningRateScheduler





# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.shape
y_train = train['label']

X_train  = train.drop(labels = "label", axis=1)
X_train.head(10)
del train
sns.countplot(y_train)

plt.show()
print(X_train.isnull().any().describe())

print(test.isnull().any().describe())
X_train = X_train/255.0

test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
y_train= to_categorical(y_train)

count_classes = y_train.shape[1]

count_classes
random_seed = 2

# Split the train and the validation set for the fitting

x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state=random_seed)
cnn = Sequential()

#Layer 1 : 16 filters

cnn.add(Conv2D(filters=16,

               kernel_size=(5,5),

               strides=(1,1),

               padding='same',

               input_shape=(28,28,1),

               data_format='channels_last'))

cnn.add(Activation('relu'))



cnn.add(MaxPooling2D(pool_size=(2,2),

                     strides=2))



cnn.add(Dropout(0.25))





# Layer 2: 32 Filters

cnn.add(Conv2D(filters=32,

               kernel_size=(5,5),

               strides=(1,1),

               padding='same'))

cnn.add(Activation('relu'))

cnn.add(MaxPooling2D(pool_size=(2,2),

                     strides=2))



cnn.add(Dropout(0.2))





# Layer 3: 64 Filters

cnn.add(Conv2D(filters=64,

               kernel_size=(5,5),

               strides=(1,1),

               padding='same'))

cnn.add(Activation('relu'))

cnn.add(MaxPooling2D(pool_size=(2,2),

                     strides=2))



cnn.add(Dropout(0.2))





# Layer : 128 Filters

cnn.add(Conv2D(filters=128,

               kernel_size=(5,5),

               strides=(1,1),

               padding='same'))

cnn.add(Activation('relu'))

cnn.add(MaxPooling2D(pool_size=(2,2),

                     strides=2))



cnn.add(Dropout(0.2))



cnn.add(BatchNormalization())





# Output of layer 4 is flattened, to send to fully connected layers

cnn.add(Flatten())



# Fully connected layer 1

cnn.add(Dense(1024))

cnn.add(Activation('relu'))



cnn.add(Dropout(0.25))



# Fully connected layer 2

cnn.add(Dense(1024))

cnn.add(Activation('relu'))



cnn.add(Dropout(0.5))



# Final output layer to predict 10 target classes

cnn.add(Dense(10))

cnn.add(Activation('softmax'))
shift = 0.2

datagen = ImageDataGenerator(zoom_range = 0.1,

                            height_shift_range = shift,

                            width_shift_range = shift)
cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

# Fit model with training set and check on validation set, to avoid overfitting

start = time.time()

cnn_result=cnn.fit_generator(

    datagen.flow(x_train, y_train, batch_size=16),

    steps_per_epoch = 1000,

    epochs=30,

    validation_data=(x_val[:1000,:], y_val[:1000,:]),

    validation_steps = 100, callbacks= [annealer])

end = time.time()

print('Training processing time:',(end - start)/60)
final_loss, final_acc = cnn.evaluate(x_val, y_val, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
plt.plot(cnn_result.history['loss'], color='b')

plt.plot(cnn_result.history['val_loss'], color='r')

plt.show()

plt.plot(cnn_result.history['acc'], color='b')

plt.plot(cnn_result.history['val_acc'], color='r')

plt.show()
test_set = test.reshape(-1, 28, 28 , 1).astype('float32')



result = cnn.predict(test_set)

result = np.argmax(result,axis = 1)

result = pd.Series(result, name="Label")

submit = pd.concat([pd.Series(range(1 ,28001) ,name = "ImageId"),   result],axis = 1)

submit.to_csv("MNIST_Test.csv",index=False)

submit.head(10)