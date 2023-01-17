import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

import numpy as np
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



train_data.head()
train_data.shape
test_data.shape
train_Y = train_data.iloc[:, 0]

train_X = train_data.iloc[:, 1:]
train_X_ = train_X.iloc[:int(0.8*28000), : ]

train_Y_ = train_Y.iloc[:int(0.8*28000)]

test_X_ = train_X.iloc[int(0.8*28000):, :]

test_Y_ = train_Y.iloc[int(0.8*28000):]
print("Training samples: %d\nTest samples: %d"%(len(train_X_), len(test_X_)))
train_X_ = [np.reshape(train_X_.iloc[i], (28, 28, 1)) for i in range(22400)]

test_X_ = [np.reshape(test_X_.iloc[i], (28, 28, 1)) for i in range(19600)]
train_X_ = np.array(train_X_)

train_Y_ = np.array(train_Y_)

test_X_ = np.array(test_X_)

test_Y_ = np.array(test_Y_)
import keras as ks

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

from keras.losses import categorical_crossentropy

from keras.optimizers import SGD
#10 = num. of classes

train_Y_ = ks.utils.to_categorical(train_Y_, 10)

test_Y_ = ks.utils.to_categorical(test_Y_, 10)
model = ks.Sequential()

model.add(Conv2D(filters=3, kernel_size=(4,4), input_shape=(28, 28, 1), activation='relu'))

model.add(MaxPooling2D())

model.add(Conv2D(filters=6, kernel_size=(2,2), activation='relu'))

model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(100, activation='relu'))

model.add(Dense(10, activation='softmax'))
model.compile(loss=ks.losses.categorical_crossentropy,

             optimizer=SGD(lr=0.01),

             metrics=['accuracy'])
history = model.fit(train_X_, train_Y_,

                    batch_size=128,

                    epochs=10,

                   validation_data=(test_X_, test_Y_))



# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='best')

plt.show()



# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='best')

plt.show()
test = [np.reshape(test_data.iloc[i], (28,28)) for i in range(3)]

for i in test:

    plt.imshow(i)

    plt.show()

    plt.clf()
test = [np.reshape(i, (28,28,1)) for i in test]

test = np.array(test)

model.predict_classes(test)