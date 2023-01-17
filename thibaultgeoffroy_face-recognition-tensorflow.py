###### This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python import keras



img_rows, img_cols = 56, 56
num_classes = 2
file_size = 111430
def prep_data(labels, raw, train_size, val_size):
    y = labels
    out_y = keras.utils.to_categorical(y, num_classes)
    
    x = raw[:,:]
    num_images = raw.shape[0]
    out_x = x.reshape(file_size, img_rows, img_cols, 3)
    out_x = out_x / 255
    return train_test_split(out_x, out_y, test_size=0.2)


faces_file = "../input/trainning-data-faces/db_train.raw"
fTrain = open(faces_file, 'rb')
faces_data = np.empty([file_size, 56*56*3], dtype=np.float32)
for i in range(file_size):
    faces_data[i,:] = np.fromfile(fTrain, dtype=np.uint8, count=56*56*3).astype(np.float32)
labels = open("../input/trainning-data-faces/label_train.txt").readlines()
X_train, X_test, y_train, y_test = prep_data(labels, faces_data, train_size=8000, val_size=3430)
import matplotlib.pyplot as plt
plt.imshow(X_train[5976])
plt.title(y_train[5976])
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(horizontal_flip=True)
datagen.fit(X_train)

valgen = ImageDataGenerator()

valgen.fit(X_test)


from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout

model = Sequential()
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu',strides=2, input_shape=(img_rows, img_cols, 3)))
model.add(Dropout(0.5))
model.add(Conv2D(256, kernel_size=(3, 3),strides=2, activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(256, kernel_size=(3, 3),strides=2, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.logcosh, optimizer='adam', metrics= ['mae', 'acc', 'logcosh'])
#x = model.fit(x, y, batch_size=100, epochs=4, validation_split=0.2)
class_weight = {0: 1.,
                1: 0.1}
model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                    steps_per_epoch=len(X_train)/32, epochs=20,
                   validation_data=valgen.flow(X_test, y_test),
                   validation_steps=len(X_test)/32,
                   class_weight=class_weight)
result_x = model.predict(X_test)
perfo = model.evaluate(X_test, y_test, batch_size = 32)
perfo
model.save('faces_model_fit.h5')
