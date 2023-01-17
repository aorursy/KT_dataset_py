import numpy as np # linear algebra

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import seaborn as sns

import sys

import numpy

numpy.set_printoptions(threshold=10000,edgeitems = 10) #threshold=sys.maxsize



import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler
train_file = "../input/digit-recognizer/train.csv"

test_file = "../input/digit-recognizer/test.csv"

sample_submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
raw_data = np.loadtxt(train_file, skiprows=1, dtype='int', delimiter=',')

x_train, x_val, y_train, y_val = train_test_split(

    raw_data[:,1:], raw_data[:,0], test_size=0.1)
fig, ax = plt.subplots(2, 1, figsize=(12,6))

ax[0].plot(x_train[0])

ax[0].set_title('784x1 data')

ax[1].imshow(x_train[0].reshape(28,28), cmap='gray')

ax[1].set_title('28x28 data')
print(y_train[::]) # displays the Digits availabe corresponding to traning data.

y_train.shape
#Now we'll display the total count of each digits.

sns.countplot(raw_data[:,0]) #this will select the first column vales from the dataset
x_train = x_train.reshape(-1, 28, 28, 1)

x_val = x_val.reshape(-1, 28, 28, 1)
x_train = x_train.astype("float32")/255.

x_val = x_val.astype("float32")/255.
y_train = to_categorical(y_train)

y_val = to_categorical(y_val)

#example:

print(y_train[0])
model = Sequential()



model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',

                 input_shape = (28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

#model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))

#model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

#model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))

#model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
datagen = ImageDataGenerator(zoom_range = 0.1,

                            height_shift_range = 0.1,

                            width_shift_range = 0.1,

                            rotation_range = 10)
model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),

                           steps_per_epoch=500,

                           epochs=20, #Increase this when not on Kaggle kernel

                           verbose=2,  #1 for ETA, 0 for silent

                           validation_data=(x_val[:400,:], y_val[:400,:]), #For speed

                           callbacks=[annealer])
final_loss, final_acc = model.evaluate(x_val, y_val, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
#First graph is comparison of Training Loss and Validation loss

plt.plot(hist.history['loss'], color='b')

plt.plot(hist.history['val_loss'], color='r')

plt.show()

#Second graph is comparison of Training accuracy and Validation Accuracy.

plt.plot(hist.history['accuracy'], color='b')

plt.plot(hist.history['val_accuracy'], color='r')

plt.show()
y_hat = model.predict(x_val)

y_pred = np.argmax(y_hat, axis=1)

y_true = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_true, y_pred)

print(cm)
mnist_testset = np.loadtxt(test_file, skiprows=1, dtype='int', delimiter=',')

x_test = mnist_testset.astype("float32")

x_test = x_test.reshape(-1, 28, 28, 1)/255.
y_hat = model.predict(x_test, batch_size=64)
y_pred = np.argmax(y_hat,axis=1)
solution = pd.DataFrame({'ImageId': sample_submission['ImageId'], 'Label': y_pred})

solution[["ImageId","Label"]].to_csv("CNNPrediction.csv", index=False)

solution.head()