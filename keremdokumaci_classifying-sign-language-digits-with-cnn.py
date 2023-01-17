import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import keras

import sklearn

import cv2

import seaborn as sns

import os

import skimage as sk

import random

#this function makes y_train and y_test correct. I wrote it because the dataset did not labeled correct

def correct_label(y):

    counter = 0

    for label in y:

        index = np.argmax(label)

        if(index == 0):

            y[counter][0] = 0

            y[counter][9] = 1

        elif(index == 1):

            y[counter][1] = 0

            y[counter][0] = 1

        elif(index == 2):

            y[counter][2] = 0

            y[counter][7] = 1

        elif(index == 3):

            y[counter][3] = 0

            y[counter][6] = 1

        elif(index == 4):

            y[counter][4] = 0

            y[counter][1] = 1

        elif(index == 5):

            y[counter][5] = 0

            y[counter][8] = 1

        elif(index == 6):

            y[counter][6] = 0

            y[counter][4] = 1

        elif(index == 7):

            y[counter][7] = 0

            y[counter][3] = 1

        elif(index == 8):

            y[counter][8] = 0

            y[counter][2] = 1

        else:

            y[counter][9] = 0

            y[counter][5] = 1

        

        counter += 1

        

    return y    



#load data set 

x = np.load('/kaggle/input/sign-language-digits-dataset/Sign-language-digits-dataset/X.npy')

y = np.load('/kaggle/input/sign-language-digits-dataset/Sign-language-digits-dataset/Y.npy')



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)



x_train = x_train.reshape(-1,64,64,1)

x_test = x_test.reshape(-1,64,64,1)

y_train = correct_label(y_train)

y_test = correct_label(y_test)

#Data Augmentation

class Augmentation:

    def __init__(self):

        pass

        

    def random_rotation(self,data,label):

        # pick a random degree of rotation between 25% on the left and 25% on the right

        augmented_images = []

        augmented_label = []

        random_degree = random.uniform(-25, 25)

        counter = 0

        for img in data:

            img = sk.transform.rotate(img, random_degree)

            augmented_images.append(img)

            augmented_label.append(label[counter])

            counter += 1

        return (augmented_images,augmented_label)

    

    # add random noise to the image

    def random_noise(self,data,label):

        augmented_images = []

        augmented_label = []

        counter = 0

        for img in data:

            img = sk.util.random_noise(img)

            augmented_images.append(img)

            augmented_label.append(label[counter])

            counter += 1

        

        return (augmented_images,augmented_label)



    def horizontal_flip(self,data,label):

        # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !

        counter = 0

        augmented_images = []

        augmented_label = []

        for img in data:

            img = img[:, ::-1]

            augmented_images.append(img)

            augmented_label.append(label[counter])

            counter += 1

        return (augmented_images,augmented_label)

    

    def vertical_flip(self,data,label):

        counter = 0

        augmented_images = []

        augmented_label = []

        for img in data:

            img = np.flip(img)

            augmented_images.append(img)

            augmented_label.append(label[counter])

            counter += 1

        return (augmented_images,augmented_label)

    

   



AUG = Augmentation()



(x_noise,y_noise) = AUG.random_noise(x_train,y_train)

(x_h_flipped,y_h_flipped) = AUG.horizontal_flip(x_train,y_train)

(x_v_flipped,y_v_flipped) = AUG.vertical_flip(x_train,y_train)

(x_rotated,y_rotated) = AUG.random_rotation(x_train,y_train)



f, axarr = plt.subplots(2,2)

axarr[0,0].imshow(x_noise[0].reshape(64,64),cmap='gray')

axarr[0,1].imshow(x_h_flipped[0].reshape(64,64),cmap='gray')

axarr[1,0].imshow(x_v_flipped[0].reshape(64,64),cmap='gray')

axarr[1,1].imshow(x_rotated[0].reshape(64,64),cmap='gray')

plt.show()

#concat data 



x_noise = np.asarray(x_noise)

x_h_flipped = np.asarray(x_h_flipped)

x_v_flipped = np.asarray(x_v_flipped)

x_rotated = np.asarray(x_rotated)



x_train = np.concatenate((x_train,x_noise,x_h_flipped,x_v_flipped,x_rotated),axis=0)



#----------------------------------------------------------------------------------------------------------------------------------------------------------------



y_noise = np.asarray(y_noise)

y_h_flipped = np.asarray(y_h_flipped)

y_v_flipped = np.asarray(y_v_flipped)

y_rotated = np.asarray(y_rotated)



y_train = np.concatenate((y_train,y_noise,y_h_flipped,y_v_flipped,y_rotated),axis=0)

#Keras Model

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten,BatchNormalization,Activation,MaxPooling2D,Dropout

from keras.utils import to_categorical



model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 1)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(10))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



hist = model.fit(x_train, y_train, validation_data=(x_test, y_test),shuffle=True, epochs=10)

        


print(hist.history.keys())



# summarize history for accuracy

plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
