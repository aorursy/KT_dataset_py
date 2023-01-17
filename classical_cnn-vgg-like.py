## This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Input, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
train_file='../input/train.csv'
test_file='../input/test.csv'
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
print(train.head())
inputs = Input(shape=(28,28, 1), name='inptuts')
net = Conv2D(16, kernel_size=(3,3), activation='relu', padding='same')(inputs)
net = Conv2D(16, kernel_size=(3,3), activation='relu', padding='same')(net)
net = MaxPooling2D((2,2))(net)
# 64*14*14
net = Conv2D(84, kernel_size=(3,3), activation='relu', padding='same')(net)
net = Conv2D(84, kernel_size=(3,3), activation='relu', padding='same')(net)
net = MaxPooling2D((2,2))(net)
# 128 * 7*7
net = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(net)
net = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(net)
net = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(net)
net = MaxPooling2D((2,2))(net)
# 256 * 4*4 
# FC
net = Flatten()(net)
net = Dropout(rate=0.5)(net)
net = Dense(1024, activation='relu',)(net)
net = Dropout(rate=0.5)(net)
net = Dense(2048, activation='relu',)(net)
net = Dropout(rate=0.5)(net)
net = Dense(1000, activation='relu',)(net)

outputs = Dense(10, activation='softmax')(net)


epochs = 50
learn_rate = 1e-4
batch_size = 128
print(train.shape)
onehot = train.join(pd.get_dummies(train['label'])) / 255

def get_samples():
    while True:
        sample = onehot.sample(n=16)
        y_train = np.array(sample[[i for i in range(10)]])
        x_train = np.array(sample.drop(['label',0, 1, 2, 3, 4, 5,6,7,8,9], axis=1)).reshape((-1,28,28,1))
        yield (x_train, y_train)

def get_validations():
    while True:
        sample = onehot.sample(n=16)
        y_test = np.array(sample[[i for i in range(10)]])
        x_test = np.array(sample.drop(['label',0, 1, 2, 3, 4, 5,6,7,8,9], axis=1)).reshape((-1, 28,28,1))
        yield (x_test, y_test)
y_train = np.array(onehot[[i for i in range(10)]])
x_train = np.array(onehot.drop(['label',0, 1, 2, 3, 4, 5,6,7,8,9], axis=1)).reshape((-1,28,28,1))
datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 10)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(lr=learn_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                           steps_per_epoch=train.shape[0]/batch_size,
                           nb_epoch=epochs,
                           validation_data=get_validations(), 
                           nb_val_samples=500)

final_loss, final_acc = model.evaluate(np.array(onehot.drop(['label',0, 1, 2, 3, 4, 5,6,7,8,9], axis=1)).reshape((-1,28,28,1)), 
                                       np.array(onehot[[i for i in range(10)]]) / 255, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
x_test = np.array(test).reshape((-1,28,28,1))/255
y_hat = model.predict(x_test, batch_size=batch_size)
y_pred = np.argmax(y_hat, 1)
y_pred = pd.Series(y_pred, name='Label', dtype=np.int32)
imgid = pd.Series(range(1, 28001), name='ImageId')
submission = pd.concat([imgid, y_pred], axis=1)
print(submission.head())
submission.to_csv('submission.csv', index=False)

import matplotlib.pyplot as plt
hist.history.keys()
plt.plot(hist.history['loss'], c='r')
plt.plot(hist.history['val_loss'], c='b')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['loss', 'val_loss'])
plt.show()

