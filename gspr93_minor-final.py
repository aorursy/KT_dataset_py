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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
from keras.utils import to_categorical

img_rows, img_cols = 160,480
num_classes = 4

def data_prep(raw):
    y=raw.values[:,0]
    out_y = to_categorical(y, num_classes)

    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y

train_file = "/kaggle/input/rice-leaf-dataset/final/dataset_final.csv"
raw_data = pd.read_csv(train_file)

x, y = data_prep(raw_data)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20)
model = Sequential()
model.add(Conv2D(20, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
          batch_size=9,
          epochs=9)
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.savefig('output2.png')
y_val=model.predict(x_test)
y_new=y_val

for i in range(y_new.shape[0]):
    maxi= max(y_new[i])
    for j in range(y_new.shape[1]):
        if(y_new[i,j]==maxi):
            y_new[i,j]=1
        else:
            y_new[i,j]=0
print(y_new)
print(y_test)
m=keras.metrics.Accuracy()
_=m.update_state(y_test,y_new)
print(m.result().numpy())
import matplotlib.pyplot as plt

plt.plot(y_val.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['validation'], loc='upper left')
plt.savefig('output.png')
from tensorflow.python.keras.layers import MaxPooling2D
model4 = Sequential()
model4.add(Conv2D(20, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model4.add(MaxPooling2D(pool_size=(2,2)))
model4.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
model4.add(MaxPooling2D(pool_size=(2,2)))
model4.add(Flatten())
model4.add(Dense(128, activation='relu'))
model4.add(Dense(num_classes, activation='softmax'))
model4.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

history4 = model4.fit(x_train, y_train, validation_data=[x_test,y_test], epochs=9, batch_size=9, verbose=0)
plt.plot(history4.history['accuracy'])
plt.plot(history4.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('output4.png')
