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
!pip install tensorflow keras sklearn matplotlib pandas pil
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import cv2

import tensorflow as tf

from PIL import Image

import os

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from keras.models import Sequential, load_model

from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout


data=[]

labels=[]



height = 30

width = 30

channels = 3

classes = 43

n_inputs = height * width*channels



for i in range(classes) :

    path = "/kaggle/input/gtsrb-german-traffic-sign/Train/{0}/".format(i)

    print(path)

    Class=os.listdir(path)

    for a in Class:

        try:

            image=cv2.imread(path+a)

            image_from_array = Image.fromarray(image, 'RGB')

            size_image = image_from_array.resize((height, width))

            data.append(np.array(size_image))

            labels.append(i)

        except AttributeError:

            print(" ")

            

Cells=np.array(data)

labels=np.array(labels)





#Randomize the order of the input images

s=np.arange(Cells.shape[0])

np.random.seed(42)

np.random.shuffle(s)

Cells=Cells[s]

labels=labels[s]


(X_train,X_val)=Cells[(int)(0.15*len(labels)):],Cells[:(int)(0.15*len(labels))]



X_train = X_train.astype('float32')/255 

X_val = X_val.astype('float32')/255



(y_train,y_val)=labels[(int)(0.15*len(labels)):],labels[:(int)(0.15*len(labels))]





from keras.utils import to_categorical

y_train = to_categorical(y_train, 43)

y_val = to_categorical(y_val, 43)
#Building the model

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))

model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.3))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(Conv2D(filters=128, kernel_size=(2, 2), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.3))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(rate=0.5))

model.add(Dense(43, activation='softmax'))


model.compile(

    loss='categorical_crossentropy', 

    optimizer='adam', 

    metrics=['accuracy']

)
X_test=X_val



y_test=y_val
epochs = 20

history = model.fit(X_train,

                    y_train,

                    batch_size=32,

                    epochs=epochs,

                    validation_data=(X_test, y_test))
model.save("model.h5")


scores = model.evaluate(X_test, y_test, verbose=0)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



plt.figure(0)

plt.plot(history.history['accuracy'], label='training accuracy')

plt.plot(history.history['val_accuracy'], label='val accuracy')

plt.title('Accuracy')

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.legend()

plt.show()
plt.figure(1)

plt.plot(history.history['loss'], label='training loss')

plt.plot(history.history['val_loss'], label='val loss')

plt.title('Loss')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()

plt.show()