# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import h5py

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow

from keras.utils import plot_model

from IPython.display import SVG

import pydot

from keras.utils.vis_utils import model_to_dot

%matplotlib inline



from keras.models import Sequential

from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPool2D

from keras.optimizers import Adam, RMSprop

from keras.initializers import RandomNormal

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
trainFile = h5py.File('../input/train_happy.h5')

testFile = h5py.File('../input/test_happy.h5')



train_x = np.array(trainFile['train_set_x'][:])

train_y = np.array(trainFile['train_set_y'][:])



test_x = np.array(testFile['test_set_x'][:])

test_y = np.array(testFile['test_set_y'][:])



print ("number of training examples = " + str(train_x.shape[0]))

print ("number of test examples = " + str(test_x.shape[0]))

print ("X_train shape: " + str(train_x.shape))

print ("Y_train shape: " + str(train_y.shape))

print ("X_test shape: " + str(test_x.shape))

print ("Y_test shape: " + str(test_y.shape))
train_y = train_y.reshape((1, train_y.shape[0]))

test_y = test_y.reshape((1, test_y.shape[0]))



print(train_y.shape)

print(test_y.shape)
for i in range(0,5):

    plt.imshow(train_x[i])

    plt.show()
X_train = train_x / 255.0

X_test = test_x / 255.0



y_train = train_y.T

y_test = test_y.T

def HappyModel(input_shape):

   

    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes

    X = ZeroPadding2D((3, 3))(X_input)



    # CONV -> BN -> RELU Block applied to X

    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)

    X = BatchNormalization(axis=3, name='bn0')(X)

    X = Activation('relu')(X)



    # MAXPOOL

    X = MaxPooling2D((2, 2), name='max_pool')(X)



    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED

    X = Flatten()(X)

    X = Dense(1, activation='sigmoid', name='fc')(X)



    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.

    model = Model(inputs=X_input, outputs=X, name='HappyModel')



    return model

    
model = Sequential()



model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='Same', input_shape=(64,64,3)))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='Same'))

#dimension calculation

#(n+2*p/s)-1, where n=input size, p=padding size, s=stride steps

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='Same'))



model.add(MaxPool2D(pool_size=(2,2)))



model.add(Flatten())



model.add(Dense(128, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
epochs = 40

batch_size = 30
import keras
DESIRED_ACCURACY = 0.99

class myCallback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('acc')>DESIRED_ACCURACY):

            print(DESIRED_ACCURACY,'accuracy so cancelling training!')

            self.model.stop_training = True



callbacks = myCallback()

history = model.fit(x=X_train, y=y_train, epochs=epochs, verbose=2,batch_size=batch_size,callbacks=[callbacks])
test_score = model.evaluate(X_test, y_test, verbose=1)

print('loss and ACC',test_score)
training_accuracy = history.history['acc']

training_loss = history.history['loss']



E = range(len(training_accuracy))

plt.plot(E, training_accuracy, color='red', label='Training accuracy')

plt.title('epochs vs Training accuracy')

plt.legend()



plt.figure()

plt.plot(E, training_loss, color='red', label='Training Loss')

plt.title('epochs vs Training Loss')

plt.legend()
model.summary()


plot_model(model, to_file='HappyModel.png')

SVG(model_to_dot(model).create(prog='dot', format='svg'))