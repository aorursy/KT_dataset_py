# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.metrics import confusion_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler

from keras.losses import categorical_crossentropy
X = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

X_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

# Drop 'label' column

y = X.label

X.drop(['label'], axis=1, inplace=True)



g = sns.countplot(y)

y.value_counts()



print(X.shape)
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X, y,

                                                      train_size=0.8, test_size=0.2,

                                                      random_state=1)
X_train = np.array(X_train)

X_val = np.array(X_val)

y_train = np.array(y_train)

y_val = np.array(y_val)
fig = plt.figure(figsize= [3,3])

plt.imshow(X_train[3].reshape(28,28), cmap='gray')

ax.set_title(y_train[3])
import random

indices = range(len(y_train))

box = dict(facecolor='yellow', pad=5, alpha=1)



fig, ax = plt.subplots(10, 10, squeeze=True, figsize=(24,12))



for n in range(10):

    for m in range(10):

        d=random.choice(indices)

        ax[n][m].imshow(X_train[d].reshape(28,28), cmap='gray')

        ax[n][m].set_title(y_train[d],y=0.9,bbox=box)

X_train = X_train.reshape(-1, 28, 28, 1)

X_val = X_val.reshape(-1, 28, 28, 1)

print(X_train.shape, y_train.shape)
X_train = X_train.astype("float32")/255.

X_val = X_val.astype("float32")/255.
y_train = to_categorical(y_train)

y_val = to_categorical(y_val)



print(y_train[0])
model = Sequential()

model.add(Conv2D(20, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(28, 28, 1)))

# model.add(BatchNormalization())

model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', strides = 2))

model.add(Dropout(0.2))

model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))

# model.add(BatchNormalization())

# model.add(MaxPool2D(pool_size=(2,2)))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])
datagen = ImageDataGenerator(zoom_range = 0.1,

                            height_shift_range = 0.1,

                            width_shift_range = 0.1,

                            rotation_range = 10)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)



model.summary()
hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size=16),

                           steps_per_epoch=500,

                           epochs=30, #Increase this when not commiting the Kernel or testing few changes

                           verbose=2,  #1 for ETA, 0 for silent

                           validation_data=(X_val, y_val),

                           callbacks=[annealer])
final_loss, final_acc = model.evaluate(X_val, y_val, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
plt.plot(hist.history['loss'], color='b')

plt.plot(hist.history['val_loss'], color='r')

plt.show()

plt.plot(hist.history['acc'], color='b')

plt.plot(hist.history['val_acc'], color='r')

plt.show()
y_hat = model.predict(X_val)

y_pred = np.argmax(y_hat, axis=1)

y_true = np.argmax(y_val, axis=1)

confusion_matrix(y_true, y_pred)
X_test = np.array(X_test)

X_test = X_test.reshape(-1, 28, 28, 1)/255.
y_hat = model.predict(X_test, batch_size=64)
y_pred = np.argmax(y_hat,axis=1)
output_file = "submission.csv"

with open(output_file, 'w') as f :

    f.write('ImageId,Label\n')

    for i in range(len(y_pred)) :

        f.write("".join([str(i+1),',',str(y_pred[i]),'\n']))