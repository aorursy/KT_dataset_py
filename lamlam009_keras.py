# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from matplotlib import pyplot as plt
import random
data_train = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")
data_train.tail()
def show_im(id):
    image = data_train.loc[id]
    print(image["label"])
    image = image.drop(["label"])
    image = image.reshape(28,28)
    plt.imshow(image, cmap='gray')
show_im(random.randrange(0, data_train.shape[0]))
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
X_train = data_train.drop(["label"], axis = 1)
y_train = pd.get_dummies(data_train["label"])
y_train.head()
X_train = X_train.values.reshape(X_train.shape[0], 28, 28, 1)
X_train.shape
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()
lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=1, verbose=1)
model.fit(X_train, y_train, batch_size=128, epochs=20, verbose=1, validation_split=0.1, callbacks=[lr_reduce])
model.save('my_model.h5')
# from keras.models import load_model
# model = load_model('my_model.h5')
def show_im_test(id):
    image = data_test.loc[id]
    image = image.reshape(28,28)
    plt.imshow(image, cmap='gray')
show_im_test(random.randrange(0, data_test.shape[0]))
X_test = data_test.values.reshape(data_test.shape[0], 28, 28, 1)
X_test.shape
pred = model.predict(X_test).argmax(1)
rand = random.randrange(0, data_test.shape[0])
print(pred[rand])
show_im_test(rand)
Id = list(range(1,X_test.shape[0]+1))
my_submission = pd.DataFrame({'ImageId': Id, 'Label': pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
print(check_output(["ls", "../working/"]).decode("utf8"))
pred2 = model.predict(X_train).argmax(1)
data_wrong_pred = data_train[data_train["label"] != pred2]
rand = random.randrange(0, data_wrong_pred.shape[0])
show_im(data_wrong_pred.index[rand])
print("prediction: " + str(pred2[data_wrong_pred.index[rand]]))