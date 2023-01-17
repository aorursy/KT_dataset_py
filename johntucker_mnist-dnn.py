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
import keras

keras.__version__

from keras.callbacks import LearningRateScheduler
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt



plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
train = pd.read_csv('../input/train.csv')
x_test = pd.read_csv('../input/test.csv')

Y_train = train['label']
x_train = train.drop(labels = ['label'], axis=1)

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.values.reshape(-1, 28, 28, 1)
x_test = x_test.values.reshape(-1, 28, 28, 1)

Y_train = to_categorical(Y_train, num_classes = 10)
x_train, x_val, Y_train, Y_val = train_test_split(x_train, 
                                                  Y_train, 
                                                  test_size=0.1)
plt.imshow(x_train[0][:,:,0], cmap='gray')
datagen = ImageDataGenerator(zoom_range = 0.15,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 15)
nets = 15
model = [0] * nets

for i in range(nets):
    input_img = keras.Input(shape=(28, 28, 1))
    x = Conv2D(32, 3, activation='relu')(input_img)
    x = BatchNormalization()(x)
    x = Conv2D(32, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, 3, strides=2, padding='same', activation='relu')(x)
    x = Dropout(0.4)(x)

    x = Conv2D(64, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, 5, strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(128, 4, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)

    prediction = Dense(10, activation='softmax')(x)


    model[i] = Model(input_img, prediction)
    
    model[i].compile(optimizer = 'adam',
              loss='categorical_crossentropy', 
              metrics=["accuracy"])
    model[i].summary()
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
history = [0] * nets
for i in range(nets):
    history[i] = model[i].fit_generator(datagen.flow(x_train, Y_train, batch_size=64),
                                  steps_per_epoch=500,
                                  epochs=60,
                                  verbose=2,
                                  validation_data=(x_val, Y_val),
                                  callbacks=[annealer])
for i in range(nets):
    acc = history[i].history['acc']
    val_acc = history[i].history['val_acc']
    loss = history[i].history['loss']
    val_loss = history[i].history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('CNN %d accuracy' % (i + 1))
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('CNN %d loss' % (i + 1))
    plt.legend()

    plt.show()
for i in range(nets):
    final_loss, final_acc = model[i].evaluate(x_val, Y_val, verbose=0)
    print("CNN {0:d} Final loss: {1:.4f}, final accuracy: {0:.4f}".format(i + 1, final_loss, final_acc))

results = np.zeros( (x_test.shape[0],10) )

for i in range(nets):
    results = results + model[i].predict(x_test)

results = np.argmax(results, axis=1)
results = pd.Series(results, name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submission.csv",index=False)
