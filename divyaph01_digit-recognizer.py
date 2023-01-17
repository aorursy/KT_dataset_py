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
train_set = pd.read_csv("/kaggle/input/train.csv", delimiter = ',')

train_set.head()
train_X = np.asarray(train_set.drop(["label"], axis = 1))

train_X.shape
train_Y = np.asarray(train_set["label"])

train_Y.shape
test_set = pd.read_csv("/kaggle/input/test.csv", delimiter = ',') 

test_set.head()  
test_X = np.asarray(test_set)

test_X.shape
train_X = train_X.reshape(train_X.shape[0], 28, 28)

import matplotlib.pyplot as plt 

%matplotlib inline 



for i in range(6, 9):

    plt.subplot(330 + (i+1))

    plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))

    plt.title(train_Y[i]);
#expand 1 more dimention as 1 for colour channel gray

train_X = train_X.reshape(train_X.shape[0], 28, 28,1)

train_X.shape
test_X = test_X.reshape(test_X.shape[0], 28, 28,1)

test_X.shape
mean_px = train_X.mean().astype(np.float32)

std_px = train_X.std().astype(np.float32)



def standardize(x): 

    return (x-mean_px)/std_px
from keras.utils.np_utils import to_categorical

train_Y = to_categorical(train_Y)

num_classes = train_Y.shape[1]

num_classes
plt.title(train_Y[9])

plt.plot(train_Y[9])

plt.xticks(range(10));
# fix random seed for reproducibility

seed = 43

np.random.seed(seed)
from keras.models import  Sequential

from keras.layers.core import  Lambda , Dense, Flatten, Dropout

from keras.callbacks import EarlyStopping

from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D
model= Sequential()

model.add(Lambda(standardize,input_shape=(28,28,1)))

model.add(Flatten())

model.add(Dense(10, activation='softmax'))

print("input shape ",model.input_shape)

print("output shape ",model.output_shape)
from keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),

 loss='categorical_crossentropy',

 metrics=['accuracy'])
from keras.preprocessing import image

gen = image.ImageDataGenerator()
from sklearn.model_selection import train_test_split

X = train_X

y = train_Y

train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.10, random_state=42)

batches = gen.flow(train_X, train_Y, batch_size=64)

val_batches=gen.flow(val_X, val_Y, batch_size=64)
history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3, 

                    validation_data=val_batches, validation_steps=val_batches.n)
history_dict = history.history

history_dict.keys()
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

from keras.optimizers import Adam ,RMSprop



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
model= get_cnn_model()

model.optimizer.lr=0.01
history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 

                    validation_data=val_batches, validation_steps=val_batches.n)
from keras.preprocessing import image

#gen = image.ImageDataGenerator()

gen =image.ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,

                               height_shift_range=0.08, zoom_range=0.08)

batches = gen.flow(train_X, train_Y, batch_size=64)

val_batches = gen.flow(val_X, val_Y, batch_size=64)
model.optimizer.lr=0.001

history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 

                    validation_data=val_batches, validation_steps=val_batches.n)
model.optimizer.lr=0.01

gen = image.ImageDataGenerator()

batches = gen.flow(X, y, batch_size=64)

history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3)
predictions = model.predict_classes(test_X, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("DR_submission.csv", index=False, header=True)
result = pd.read_csv("DR_submission.csv", delimiter = ",")

result.shape