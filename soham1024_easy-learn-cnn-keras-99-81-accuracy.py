import numpy as np # linear algebra

import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)



# Plot ad hoc mnist instances

from keras.datasets import mnist

import matplotlib.pyplot as plt

# load (downloaded if needed) the MNIST dataset

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# plot 4 images as gray scale

plt.subplot(221)

plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))

plt.subplot(222)

plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))

plt.subplot(223)

plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))

plt.subplot(224)

plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))

# show the plot

plt.show()
def visualize_input(img, ax):

    ax.imshow(img, cmap='gray')

    width, height = img.shape

    thresh = img.max()/2.5

    for x in range(width):

        for y in range(height):

            ax.annotate(str(round(img[x][y],2)), xy=(y,x),

                        horizontalalignment='center',

                        verticalalignment='center',

                        color='white' if img[x][y]<thresh else 'black')



fig = plt.figure(figsize = (12,12)) 

ax = fig.add_subplot(111)

visualize_input(X_train[1].reshape(28,28), ax)
import seaborn as sns



g = sns.countplot(y_train)
from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.utils import np_utils
#load data

(X_train, y_train), (X_test, y_test) = mnist.load_data()
#flatten 28*28 images to a 784 vector for each image

num_pixels = X_train.shape[1] * X_train.shape[2]

X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')

X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')
# normalize inputs from 0-255 to 0-1

X_train = X_train / 255

X_test = X_test / 255
# one hot encode outputs

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
# define baseline model

def baseline_model():

	# create model

	model = Sequential()

	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))

	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

	# Compile model

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model
# build the model

model = baseline_model()

model.summary()
# Fit the model

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

# Final evaluation of the model

scores = model.evaluate(X_test, y_test, verbose=0)

print("Baseline Error: %.2f%%" % (100-scores[1]*100))
# Simple CNN for the MNIST Dataset

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils



# load data

(X_train, y_train), (X_test, y_test) = mnist.load_data()



# reshape to be [samples][width][height][channels]

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')

X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')



# normalize inputs from 0-255 to 0-1

X_train = X_train / 255

X_test = X_test / 255



# one hot encode outputs

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]



# define a simple CNN model

def baseline_model():

	# create model

	model = Sequential()

	model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))

	model.add(MaxPooling2D())

	model.add(Dropout(0.2))

	model.add(Flatten())

	model.add(Dense(128, activation='relu'))

	model.add(Dense(num_classes, activation='softmax'))

	# Compile model

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model



# build the model

model_simple = baseline_model()

model_simple.summary()
# Fit the model

model_simple.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)



# Final evaluation of the model

scores = model_simple.evaluate(X_test, y_test, verbose=1)



print("CNN Error: %.2f%%" % (100-scores[1]*100))
# Large CNN for the MNIST Dataset

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils



# load data

(X_train, y_train), (X_test, y_test) = mnist.load_data()



# reshape to be [samples][width][height][channels]

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')

X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')



# normalize inputs from 0-255 to 0-1

X_train = X_train / 255

X_test = X_test / 255



# one hot encode outputs

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]



# define the larger model

def large_model():

	# create model

	model = Sequential()

	model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))

	model.add(MaxPooling2D())

	model.add(Conv2D(15, (3, 3), activation='relu'))

	model.add(MaxPooling2D())

	model.add(Dropout(0.2))

	model.add(Flatten())

	model.add(Dense(128, activation='relu'))

	model.add(Dense(50, activation='relu'))

	model.add(Dense(num_classes, activation='softmax'))

	# Compile model

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model



# build the model

model_large = large_model()

model_large.summary()
# Fit the model

model_large.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

# Final evaluation of the model



scores = model_large.evaluate(X_test, y_test, verbose=1)



print("Large CNN Error: %.2f%%" % (100-scores[1]*100))
# Larger CNN for the MNIST Dataset

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils



# load data

(X_train, y_train), (X_test, y_test) = mnist.load_data()



# reshape to be [samples][width][height][channels]

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')

X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')



# normalize inputs from 0-255 to 0-1

X_train = X_train / 255

X_test = X_test / 255



# one hot encode outputs

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]



# define the larger model

def larger_model():

    # create model

    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))

    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))

    model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=2))

    model.add(Dropout(0.1))

    model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))

    model.add(Conv2D(filters=192, kernel_size=3, padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=2))

    model.add(Dropout(0.1))

    model.add(Conv2D(filters=192, kernel_size=5, padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=2, padding='same'))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    # Compile model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



# build the model

model_larger = larger_model()

model_larger.summary()
# Fit the model

model_larger.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=100)

# Final evaluation of the model



scores = model_larger.evaluate(X_test, y_test, verbose=1)



print("Larger CNN Error: %.2f%%" % (100-scores[1]*100))
import pandas as pd

X_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv').values.astype('float32')

X_test = X_test.reshape(-1, 28, 28, 1)

X_test = X_test.astype('float32')/255

testY = model_larger.predict_classes(X_test, verbose=1)
sub = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

sub['Label'] = testY

sub.to_csv('submission.csv',index=False)
from IPython.display import YouTubeVideo

YouTubeVideo('3JQ3hYko51Y', width=800, height=450)