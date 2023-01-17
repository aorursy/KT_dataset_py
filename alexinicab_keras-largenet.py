# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')  

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")



X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values

Y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits

x_test = test.values.astype('float32')



# Dividing in training and dev set

x_train = X_train[0:38000]

y_train = Y_train[0:38000]

x_dev = X_train[38000:42000]

y_dev = Y_train[38000:42000]



# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

x_train = x_train.reshape(-1,28,28,1)

x_dev = x_dev.reshape(-1,28,28,1)

x_test = x_test.reshape(-1,28,28,1)



# tocategorical

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

y_train = to_categorical(y_train, num_classes = 10)

y_dev = to_categorical(y_dev, num_classes = 10)
from keras import layers

from keras import models



model = models.Sequential()



model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu', input_shape=(28,28,1)))

model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))



model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))



model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))



model.add(layers.Flatten())



model.add(layers.Dense(units=512, activation='relu'))

model.add(layers.Dense(units=128, activation='relu'))

model.add(layers.Dense(units=10, activation = 'softmax'))
model.summary()
import time

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])



# Training

tic = time.time()

model.fit(x_train, y_train, epochs=10, batch_size=64)

toc = time.time()

print("The total training time is", toc-tic, "seconds")
dev_loss, dev_acc = model.evaluate(x_dev, y_dev)

dev_acc
# predict results

results = model.predict(x_test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)