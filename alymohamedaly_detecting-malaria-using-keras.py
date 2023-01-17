import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, Activation, MaxPooling2D

from keras.utils import np_utils

from sklearn.model_selection import train_test_split

from skimage.io import imshow, imread

from skimage.transform import resize, rotate

import os
StringLocation = "../input/cell_images/cell_images/"

Img_Size = 50



Positive=os.listdir(StringLocation + "Parasitized/")

Positive=[StringLocation + "Parasitized/" + i for i in Positive]

del Positive[3901]   #Removing Thumbs.db



Negative=os.listdir(StringLocation + "Uninfected/")

Negative=[StringLocation + "Uninfected/" + i for i in Negative]

del Negative[3913]   #Removing Thumbs.db



sz = len(Positive) + len(Negative)



labels = np.zeros(sz*2)

labels[::2]=1



data = np.zeros((sz*2,Img_Size,Img_Size,3))
for i in range(2):

    data[i*sz:(i+1)*sz:2] = [(rotate(resize(imread(ImgName), (Img_Size,Img_Size)), (i*180))) for ImgName in Positive]

    data[(i*sz)+1:(i+1)*sz:2] = [(rotate(resize(imread(ImgName), (Img_Size,Img_Size)), (i*180))) for ImgName in Negative]
np.save("data",data)

np.save("labels",labels)
data=np.load("data.npy")

labels=np.load("labels.npy")
plt.figure(figsize=(15,10))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.xticks([])

    plt.grid(False)

    plt.imshow(data[i])

    plt.xlabel('Infected' if labels[i] == 1 else 'Free')

plt.show()
data = data.reshape(-1, Img_Size, Img_Size, 3)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 37)



y_train = np_utils.to_categorical(y_train, num_classes = 2)

y_test = np_utils.to_categorical(y_test, num_classes = 2)
def BuildModel(Shape):

    model = Sequential()



    model.add(Conv2D(32, (3,3), padding='same', input_shape = Shape))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(2,2))



    model.add(Conv2D(64, (3,3), padding='same'))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(2,2))



    model.add(Conv2D(128, (3,3), padding='same'))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(2,2))

    

    model.add(Conv2D(128, (3,3), padding='same'))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(2,2))

    model.add(Dropout(0.5))



    model.add(Flatten())



    model.add(Dense(256))

    model.add(Activation('relu'))

    model.add(Dropout(0.5))



    model.add(Dense(2, activation = 'softmax'))

    

    return model
model = BuildModel(data.shape[1:])

model.summary()

model.compile(loss='binary_crossentropy',

                optimizer='adam',

                metrics=['accuracy'])
# Train

history = model.fit(x_train, y_train, epochs=25, validation_data=(x_test, y_test) , batch_size = 64, verbose=1)
print('Accuracy = {}%'.format((history.history['val_acc'][-1])*100))
from keras.models import load_model

model.save('Malaria.h5')
model = load_model('Malaria.h5')