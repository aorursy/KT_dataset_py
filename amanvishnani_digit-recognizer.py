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

import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv("../input/train.csv");
df.info();
df.head()
df.shape
trainX = (df.iloc[:,1:].values).astype('float32')
trainY = (df.iloc[:,0].values).astype('float32')
trainY
trainX = trainX.reshape(trainX.shape[0], 28, 28)
for i in range(6, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
    plt.title(trainY[i]);
trainX= trainX.reshape(trainX.shape[0],28,28,1)
trainX.shape
mean_px = trainX.mean().astype(np.float32)
std_px = trainX.std().astype(np.float32)

def standardize(x): 
    return (x-mean_px)/std_px
mean_px
std_px

testX = pd.read_csv("../input/test.csv");
testX = testX.values.astype('float32')
testX = testX.reshape(testX.shape[0], 28, 28, 1);
testX.shape
from keras.utils.np_utils import to_categorical
trainY = to_categorical(trainY)
num_classes = trainY.shape[1]
num_classes
trainY
plt.title(trainY[9])
plt.plot(trainY[9])
plt.xticks(range(10));
# fix random seed for reproducibility
seed = 43
np.random.seed(seed)
from keras.models import  Sequential
from keras.layers.core import  Lambda , Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D
model = Sequential();
model.add(Lambda(standardize, input_shape=(28, 28, 1)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
print("input shape ",model.input_shape)
print("output shape ",model.output_shape)
from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
from keras.preprocessing import image
gen = image.ImageDataGenerator()
from sklearn.model_selection import train_test_split
X = trainX
Y = trainY
trainX, X_val, trainY, Y_val = train_test_split(trainX, trainY, test_size=0.1, random_state=42);
batches = gen.flow(trainX, trainY, batch_size=64)
val_batches = gen.flow(X_val, Y_val, batch_size=64);
history = model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3, validation_data=val_batches, validation_steps=val_batches.n);
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
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

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
from keras.optimizers import Adam

def get_cnn_model():
    model = Sequential([
        Lambda(standardize, input_shape=(28,28,1)),
        Convolution2D(32, (3,3), activation='relu'),
        Convolution2D(32, (3,3), activation='relu'),
        MaxPooling2D(),
        Convolution2D(64, (3,3), activation='relu'),
        Convolution2D(64, (3,3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
    ]);
    model.compile(Adam(), loss='categorical_crossentropy',
        metrics=['accuracy']);
    return model;
model = get_cnn_model();
model.optimizer.lr=0.01;
history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 
                    validation_data=val_batches, validation_steps=val_batches.n)
model.optimizer.lr=0.01
gen = image.ImageDataGenerator()
batches = gen.flow(trainX, trainY, batch_size=64)
history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3)
predictions = model.predict_classes(testX, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)