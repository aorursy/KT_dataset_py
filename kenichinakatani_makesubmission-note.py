# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.

from sklearn.model_selection import train_test_split
import keras

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization
import os

print(os.listdir("../input"))
data_train = pd.read_csv('../input/train42000.csv')

data_test = pd.read_csv('../input/test28000.csv')
data_train.head()
data_test.head()
X = np.array(data_train.iloc[:, 1:])


y_tmp = np.array(data_train.iloc[:, 0])

y_tmp
y = to_categorical(y_tmp)

y
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)
X_test = np.array(data_test.iloc[:, 0:])
X_test
X_test.shape
X_test.shape[0]
img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)
X_test.shape
X_train.dtype
X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_val = X_val.astype('float32')
X_train /= 255

X_test /= 255

X_val /= 255
num_classes = 10

model = Sequential()

model.add(Conv2D(20, kernel_size=(5, 5),

                 activation='relu',

                 kernel_initializer='he_normal',

                 input_shape=input_shape))

model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(20, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adam(),

              metrics=['accuracy'])
batch_size = 256

epochs = 5
history = model.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(X_val, y_val))
predicted_classes = model.predict_classes(X_test)
predicted_classes
df = pd.DataFrame(predicted_classes,columns=['label'])
df.head()
df.to_csv(path_or_buf="Submission.csv",index_label="image_id")
history.history.keys()
history.history['val_acc']
import matplotlib.pyplot as plt

%matplotlib inline

 

plt.plot(range(1, epochs+1), history.history['acc'], label="training")

plt.plot(range(1, epochs+1), history.history['val_acc'], label="validation")

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()

plt.plot(range(1, epochs+1), history.history['loss'], label="training")

plt.plot(range(1, epochs+1), history.history['val_loss'], label="validation")

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()