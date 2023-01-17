# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten, Conv2D , MaxPool2D
import matplotlib.pyplot as plt
import random
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/fashion-mnist_train.csv')
df_test = pd.read_csv('../input/fashion-mnist_test.csv')
TYPE_MAP = {
    0:"T-shirt/top",
    1:"Trouser",
    2:"Pullover",
    3:"Dress",
    4:"Coat",
    5:"Sandal",
    6:"Shirt",
    7:"Sneaker",
    8:"Bag",
    9:"Ankle boot",
}
NUM_CLASSES=10
img_rows = img_cols = 28
input_shape = (img_rows, img_cols, 1)

X = np.array(df_train.iloc[:, 1:])
y = to_categorical(np.array(df_train.iloc[:, 0]))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)

X_test = np.array(df_test.iloc[:, 1:])
y_test = to_categorical(np.array(df_test.iloc[:, 0]))

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255
X_val = X_val.astype('float32')/255
model = Sequential()

model.add(Conv2D(32, kernel_size=(5, 5), activation='relu',input_shape=(img_rows,img_cols,1)))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.35))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, batch_size=256, epochs=10,verbose=1,validation_data=(X_test, y_test))
score=model.evaluate(X_test,y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
def show_digit(pixels):
  img= pixels.reshape(28,28)
  plt.axis('off')
  plt.imshow(img, cmap='gray_r')
sample = random.choice(X_test)
show_digit(sample)
sample = sample.reshape(1,28,28,1)

predictions = model.predict(sample)[0]

for i, v in enumerate(predictions):
  print(u"Picture %s Posbltyt : %.6f%%" % (TYPE_MAP[i] , v * 100))
