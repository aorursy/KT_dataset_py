# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/Kannada-MNIST/train.csv")

train_data.shape
import matplotlib.pyplot as plt



show_exmpl = train_data.values[:8, :-1]

plt.figure(1, figsize=(14, 7))

for i in range(8):

    plt.subplot(2, 4, i + 1)

    plt.imshow(show_exmpl[i].reshape((28, 28)), cmap='gray')
X_train_test = train_data.values[:, 1:]

y_train_test = train_data.label.values



ind = np.random.permutation(X_train_test.shape[0])

X_train_test = X_train_test[ind]

y_train_test = y_train_test[ind]



validate_size = int(0.2 * X_train_test.shape[0])

X_train, X_test = X_train_test[validate_size:], X_train_test[:validate_size]

y_train, y_test = y_train_test[validate_size:], y_train_test[:validate_size]



print('Train shapes: ', X_train.shape, y_train.shape)

print('Test shapes: ', X_test.shape, y_test.shape)
print(np.min(X_train), np.max(X_train))

X_train_max = np.max(X_train)

X_train = X_train / (0.5 * X_train_max) - 1

print(np.min(X_train), np.max(X_train))



print(np.min(X_test), np.max(X_test))

X_test = X_test / (0.5 * X_train_max) - 1 

print(np.min(X_test), np.max(X_test))
from keras.layers import *

from keras.models import Sequential
model = Sequential()



model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1),padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=3, activation='relu',padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.7))



model.add(Conv2D(128, kernel_size=3, activation='relu',padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=3, activation='relu',padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.7))



model.add(Flatten())

model.add(Dense(128))

model.add(BatchNormalization())

model.add(Dropout(0.7))

model.add(Dense(10, activation='softmax'))



model.compile(optimizer ='sgd',

              loss = 'sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.summary()
from keras.callbacks import CSVLogger, ModelCheckpoint
X_train = X_train.reshape(X_train.shape[0],28,28,1)

X_test = X_test.reshape(X_test.shape[0],28,28,1)



model.fit(X_train, y_train,

          epochs=25,

          verbose=1,

          validation_data=(X_test, y_test),

          callbacks=[

              ModelCheckpoint('/kaggle/working/best_kannada_model.h5', save_best_only=True),

              CSVLogger('/kaggle/working/learning_log.csv'),

          ])
test_csv = pd.read_csv("../input/Kannada-MNIST/test.csv")

X_val = np.array(test_csv.drop("id",axis=1), dtype=np.float32)

X_val.shape
X_val_max = np.max(X_val)

X_val = X_val / (0.5 * X_val_max) - 1

X_val = np.reshape(X_val, (-1,28,28,1))



print(X_val.shape, np.min(X_val), np.max(X_val))
from keras.models import load_model



best_model = load_model('/kaggle/working/best_kannada_model.h5')

Y_val = best_model.predict(X_val)

Y_val = np.argmax(Y_val, axis = 1)
submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")

submission['label'] = Y_val

submission.to_csv("submission.csv",index=False)