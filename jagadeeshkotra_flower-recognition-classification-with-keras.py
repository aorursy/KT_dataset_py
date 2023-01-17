#define file paths.

import os

daisy_path = "../input/flowers/flowers/daisy/"

dandelion_path = "../input/flowers/flowers/dandelion/"

rose_path = "../input/flowers/flowers/rose/"

sunflower_path = "../input/flowers/flowers/sunflower/"

tulip_path = "../input/flowers/flowers/tulip/"
from os import listdir

import cv2







img_data = []

labels = []



size = 128,128

def iter_images(images,directory,size,label):

    try:

        for i in range(len(images)):

            img = cv2.imread(directory + images[i])

            img = cv2.resize(img,size)

            img_data.append(img)

            labels.append(label)

    except:

        pass



iter_images(listdir(daisy_path),daisy_path,size,0)

iter_images(listdir(dandelion_path),dandelion_path,size,1)

iter_images(listdir(rose_path),rose_path,size,2)

iter_images(listdir(sunflower_path),sunflower_path,size,3)

iter_images(listdir(tulip_path),tulip_path,size,4)
len(img_data),len(labels)
import numpy as np

data = np.asarray(img_data)



#div by 255

data = data / 255.0



labels = np.asarray(labels)
data.shape,labels.shape
from sklearn.model_selection import train_test_split



# Split the data

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.30, shuffle= True)
from keras.models import Sequential

from keras.layers import Conv2D,MaxPool2D,Flatten,Dense



model = Sequential()

model.add(Conv2D(16, (2,2),input_shape=(128, 128, 3),activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32, (2,2),activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dense(5,activation='softmax'))



model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



batch_size = 128

epochs = 30

model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          validation_data=(x_test, y_test))
pred = model.predict_classes(x_test[:10])



for i in range(len(pred)):

    print(pred[i],'==>',y_test[i])