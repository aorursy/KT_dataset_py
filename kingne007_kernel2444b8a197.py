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
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

print(train_df.shape,test_df.shape)
train_X = train_df.drop(['label'],axis=1).values

train_Y = train_df['label'].values

test_X = test_df.values

print(train_X.shape,train_Y.shape,test_X.shape)
import matplotlib.pyplot as plt



plt.figure(figsize=(15,6))

for i in range(40):

    plt.subplot(4,10,i+1)

    plt.imshow(train_X[i].reshape((28,28)),cmap=plt.cm.binary)

    plt.title('label=%d' % train_Y[i],y=0.9)

    plt.axis('off')

    

plt.subplots_adjust(wspace=0.3,hspace=-0.1)

plt.show()
n_x = 28

train_X_digit = train_X.reshape((-1,n_x,n_x,1))

test_X_digit = test_X.reshape((-1,n_x,n_x,1))

print(train_X_digit.shape,test_X_digit.shape)



train_X_digit = train_X_digit / 255

test_X_digit = test_X_digit / 255



from keras.utils.np_utils import to_categorical

onehot_labels = to_categorical(train_Y)

print(onehot_labels.shape)

print(train_Y[181],onehot_labels[181])

plt.figure(figsize=(3,2))

plt.imshow(train_X[181].reshape((28,28)),cmap=plt.cm.binary)

plt.show()
from keras_preprocessing.image import ImageDataGenerator

data_augment = ImageDataGenerator(rotation_range=10,

                                 width_shift_range=0.1,

                                 height_shift_range=0.1,

                                 zoom_range=0.1)
# build the CNN from keras

from keras import models

from keras import layers



model = models.Sequential()

model.add(layers.Conv2D(32, kernel_size=5, padding='same', activation='relu', input_shape=(28, 28, 1)))

model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Dropout(rate=0.4))

model.add(layers.Conv2D(64, kernel_size=5, activation='relu'))

model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Dropout(rate=0.4))

model.add(layers.Conv2D(128, kernel_size=3, activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dropout(rate=0.4))

model.add(layers.Dense(10, activation='softmax'))



model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', 

              metrics=['accuracy'])
# set up a dev set (5000 samples) to check the performance of the CNN

X_dev = train_X_digit[:5000]

rem_X_train = train_X_digit[5000:]

print(X_dev.shape, rem_X_train.shape)



Y_dev = onehot_labels[:5000]

rem_Y_train = onehot_labels[5000:]

print(Y_dev.shape, rem_Y_train.shape)
# Train and validate the model

epochs = 10

batch_size = 128

history = model.fit_generator(data_augment.flow(rem_X_train, rem_Y_train, batch_size=batch_size), 

                              epochs=epochs, steps_per_epoch=rem_X_train.shape[0]//batch_size, 

                              validation_data=(X_dev, Y_dev))
# plot and visualise the training and validation losses

loss = history.history['loss']

dev_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)



from matplotlib import pyplot as plt

plt.plot(epochs, loss, 'bo', label='training loss')

plt.plot(epochs, dev_loss, 'b', label='validation loss')

plt.title('Training and Validation Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
# predict on test set

predictions = model.predict(test_X_digit)

print(predictions.shape)
# set the predicted labels to be the one with the highest probability

predicted_labels = []

for i in range(28000):

    predicted_label = np.argmax(predictions[i])

    predicted_labels.append(predicted_label)
# look at some of the predictions for test_X

plt.figure(figsize=(15,6))

for i in range(40):  

    plt.subplot(4, 10, i+1)

    plt.imshow(test_X[i].reshape((28,28)),cmap=plt.cm.binary)

    plt.title("predict=%d" % predicted_labels[i],y=0.9)

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()
# create submission file

result = pd.read_csv('../input/sample_submission.csv')

result['Label'] = predicted_labels

# generate submission file in csv format

result.to_csv('rhodium_submission.csv', index=False)