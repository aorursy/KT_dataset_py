# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

%matplotlib inline



from keras.models import Sequential

from keras.layers import Dense, Dropout, Lambda, Flatten

from keras.optimizers import Adam, RMSprop

from sklearn.model_selection import train_test_split

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/digit-recognizer/train.csv")

print(train.shape)

train.head()
test = pd.read_csv("../input/digit-recognizer/test.csv")

print(test.shape)

test.head()
x_train = (train.iloc[:,1:].values).astype('float32')

# all pixel values

y_train = train.iloc[:,0].values.astype('int32') 

# only labels i.e targets digits

x_test = test.values.astype('float32')
x_train
y_train
# data visualization

# conveting datast to (num_imager, img_rows, img_cols) format

x_train = x_train.reshape(x_train.shape[0], 28, 28)



for i in range(6, 9):

    plt.subplot(330 + (i+1))

    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))

    plt.title(y_train[i]);
# expanding 1 more dimension as 1 for color channel gray

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_train.shape
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_test.shape
# preprocessing the digit images

# feature standardization



mean_px = x_train.mean().astype(np.float32)

std_px = x_train.std().astype(np.float32)



def standardize(x):

    return(x-mean_px)/std_px
# one hot encoding of labels



from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train)

num_classes = y_train.shape[1]

num_classes
# plotting the 10th label



plt.title(y_train[9])

plt.plot(y_train[9])

plt.xticks(range(10));
# designing the neural network architecture

# fix random seed for reproducibility

seed = 43

np.random.seed(seed)
# linear model



from keras.models import Sequential

from keras.layers.core import Lambda, Dense, Flatten, Dropout

from keras.callbacks import EarlyStopping

from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D
model = Sequential()

model.add(Lambda(standardize, input_shape=(28, 28, 1)))

model.add(Flatten())

model.add(Dense(10, activation = 'softmax'))

print("input shape ", model.input_shape)

print("output shape ", model.output_shape)
# compiling the network

from keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),

             loss='categorical_crossentropy',

             metrics=['accuracy'])
from keras.preprocessing import image

gen = image.ImageDataGenerator()
# Cross Validation



from sklearn.model_selection import train_test_split

x = x_train

y = y_train

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=42)

batches=gen.flow(x_train, y_train, batch_size=64)

val_batches=gen.flow(x_val, y_val, batch_size=64)
history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3,

                           validation_data=val_batches, validation_steps=val_batches.n)
history_dict = history.history

history_dict.keys()
import matplotlib.pyplot as plt

%matplotlib inline

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)



# "bo" is for "blue dot"

plt.plot(epochs, loss_values, 'bo')

# b+ is for "blue crosses"

plt.plot(epochs, val_loss_values, 'b+')

plt.xlabel('Epochs')

plt.ylabel('Loss')



plt.show()
plt.clf()   # clear figure

acc_values = history_dict['accuracy']

val_acc_values = history_dict['val_accuracy']



plt.plot(epochs, acc_values, 'bo')

plt.plot(epochs, val_acc_values, 'b+')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')



plt.show()
def get_fc_model():

    model = Sequential([

        Lambda(standardize, input_shape=(28,28,1)),

        Flatten(),

        Dense(512, activation='relu'),

        Dense(10, activation='softmax')

        ])

    model.compile(optimizer='Adam', loss='categorical_crossentropy',

                  metrics=['accuracy'])

    return model
fc = get_fc_model()

fc.optimizer.lr=0.01
history=fc.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 

                    validation_data=val_batches, validation_steps=val_batches.n)
from keras.layers import Convolution2D, MaxPooling2D



def get_cnn_model():

    model = Sequential([

        Lambda(standardize, input_shape=(28,28,1)),

        Convolution2D(32,(3,3), activation='relu'),

        Convolution2D(32,(3,3), activation='relu'),

        MaxPooling2D(),

        Convolution2D(64,(3,3), activation='relu'),

        Convolution2D(64,(3,3), activation='relu'),

        MaxPooling2D(),

        Flatten(),

        Dense(512, activation='relu'),

        Dense(10, activation='softmax')

        ])

    model.compile(Adam(), loss='categorical_crossentropy',

                  metrics=['accuracy'])

    return model
model = get_cnn_model()

model.optimizer.lr=0.01
history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 

                    validation_data=val_batches, validation_steps=val_batches.n)
# Data Augmentation

gen =ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,

                               height_shift_range=0.08, zoom_range=0.08)

batches = gen.flow(x_train, y_train, batch_size=64)

val_batches = gen.flow(x_val, y_val, batch_size=64)
model.optimizer.lr=0.001

history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 

                    validation_data=val_batches, validation_steps=val_batches.n)
# adding batch normalization

from keras.layers.normalization import BatchNormalization



def get_bn_model():

    model = Sequential([

        Lambda(standardize, input_shape=(28,28,1)),

        Convolution2D(32,(3,3), activation='relu'),

        BatchNormalization(axis=1),

        Convolution2D(32,(3,3), activation='relu'),

        MaxPooling2D(),

        BatchNormalization(axis=1),

        Convolution2D(64,(3,3), activation='relu'),

        BatchNormalization(axis=1),

        Convolution2D(64,(3,3), activation='relu'),

        MaxPooling2D(),

        Flatten(),

        BatchNormalization(),

        Dense(512, activation='relu'),

        BatchNormalization(),

        Dense(10, activation='softmax')

        ])

    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
model= get_bn_model()

model.optimizer.lr=0.01

history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 

                    validation_data=val_batches, validation_steps=val_batches.n)
model.optimizer.lr=0.01

gen = image.ImageDataGenerator()

batches = gen.flow(x, y, batch_size=64)

history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3)
predictions = model.predict_classes(x_test, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)