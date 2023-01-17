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
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

print(train.shape)

train.head()
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

print(test.shape)

test.head()
X_train = train.iloc[:, 1:].values.astype('float32')

y_train = train.iloc[:, 0].values.astype('int32')

X_test = test.values.astype('float32')
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)
X_train.shape[0]
X_train = X_train.reshape(X_train.shape[0], 28, 28)

X_train.shape
import matplotlib.pyplot as plt

%matplotlib inline
for i in range(0, 9):

    plt.subplot(3, 3, i + 1)

    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

    plt.title(y_train[i])

    plt.axis('off')
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_train.shape
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_test.shape
X_train.min(), X_train.max()
mean_px = X_train.mean().astype(np.float32)

std_px = X_train.std().astype(np.float32)



print(mean_px, std_px)



def standardize(x):

    return (x - mean_px) / std_px
from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train)

y_train.shape
num_classes = y_train.shape[1]

num_classes
# fix the random seed for reproducibility

seed = 43

np.random.seed(seed)
from keras.models import Sequential

from keras.layers.core import Lambda, Dense, Flatten, Dropout

from keras.callbacks import EarlyStopping

from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D
model = Sequential()

model.add(Lambda(standardize, input_shape=(28, 28, 1)))

model.add(Flatten())

model.add(Dense(10, activation='softmax'))
model.summary()
print(model.input_shape)

print(model.output_shape)
from keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),

              loss='categorical_crossentropy',

              metrics=['accuracy'])
from keras.preprocessing import image

gen = image.ImageDataGenerator()
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

batches = gen.flow(X_train, y_train, batch_size=64)

val_batches = gen.flow(X_val, y_val, batch_size=64)
print(X_train.shape)
print(len(batches))

print(batches.n // 64)
history = model.fit_generator(generator=batches,

                              steps_per_epoch=len(batches),

                              epochs=30,

                              validation_data=val_batches,

                              validation_steps=len(val_batches))
history_dict = history.history

history_dict.keys()
loss = history_dict['loss']

val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo')

plt.plot(epochs, val_loss, 'b+')

plt.xlabel('Epochs')

plt.ylabel('Loss')
plt.clf()

accuracy = history_dict['accuracy']

val_accuracy = history_dict['val_accuracy']



plt.plot(epochs, accuracy, 'bo')

plt.plot(epochs, val_accuracy, 'b+')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.show()
def get_fc_model():

    model = Sequential([

        Lambda(standardize, input_shape=(28, 28, 1)),

        Flatten(),

        Dense(512, activation='relu'),

        Dense(10, activation='softmax')

    ])

    model.compile(optimizer='Adam',

                  loss='categorical_crossentropy',

                  metrics=['accuracy'])

    return model
fc = get_fc_model()

fc.optimizer.lr
history = fc.fit_generator(generator=batches,

                           steps_per_epoch=len(batches),

                           epochs=30,

                           validation_data=val_batches,

                           validation_steps=len(val_batches))
from keras.layers import Convolution2D, MaxPooling2D

from keras.optimizers import Adam



def get_cnn_model():

    model = Sequential([

        Lambda(standardize, input_shape=(28, 28, 1)),

        Convolution2D(32, (3, 3), activation='relu'),

        Convolution2D(32, (3, 3), activation='relu'),

        MaxPooling2D(),

        Convolution2D(64, (3, 3), activation='relu'),

        Convolution2D(64, (3, 3), activation='relu'),

        MaxPooling2D(),

        Flatten(),

        Dense(512, activation='relu'),

        Dense(10, activation='softmax')

    ])

    model.compile(Adam(),

                  loss='categorical_crossentropy',

                  metrics=['accuracy'])

    return model
model = get_cnn_model()
history = model.fit_generator(generator=batches,

                              steps_per_epoch=len(batches),

                              epochs=30,

                              validation_data=val_batches,

                              validation_steps=len(val_batches))
from keras.preprocessing.image import ImageDataGenerator



gen1 = ImageDataGenerator()

gen2 = ImageDataGenerator(

    rotation_range=8,

    width_shift_range=0.08,

    shear_range=0.3,

    height_shift_range=0.08,

    zoom_range=0.08

)

batches = gen2.flow(X_train, y_train, batch_size=64)

val_batches = gen1.flow(X_val, y_val, batch_size=64)
model = get_cnn_model()

history = model.fit_generator(generator=batches,

                              steps_per_epoch=len(batches),

                              epochs=30,

                              validation_data=val_batches,

                              validation_steps=len(val_batches))
from keras.layers.normalization import BatchNormalization



def get_bn_model():

    model = Sequential([

        Lambda(standardize, input_shape=(28, 28, 1)),

        Convolution2D(32, (3, 3), activation='relu'),

        # channel (axis=3) で正規化する

        BatchNormalization(axis=3),

        Convolution2D(32, (3, 3), activation='relu'),

        MaxPooling2D(),

        BatchNormalization(axis=3),

        Convolution2D(64, (3, 3), activation='relu'),

        BatchNormalization(axis=3),

        Convolution2D(64, (3, 3), activation='relu'),

        MaxPooling2D(),

        Flatten(),

        BatchNormalization(),

        Dense(512, activation='relu'),

        BatchNormalization(),

        Dense(10, activation='softmax')

    ])

    model.compile(Adam(),

                  loss='categorical_crossentropy',

                  metrics=['accuracy'])

    return model
model = get_bn_model()

history = model.fit_generator(generator=batches,

                              steps_per_epoch=len(batches),

                              epochs=30,

                              validation_data=val_batches,

                              validation_steps=len(val_batches))
predictions = model.predict_classes(X_test, verbose=0)

submissions = pd.DataFrame({'ImageID': list(range(1, len(predictions) + 1)),

                            'Label': predictions})

submissions.to_csv('cnn_batchnorm.csv', index=False, header=True)