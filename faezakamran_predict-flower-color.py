import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization

from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D

import h5py

import os

print(os.listdir("../input"))
filename = '../input/FlowerColorImages.h5'

f = h5py.File(filename, 'r')



# List all groups

print("Keys: %s" % f.keys())

a_group_key = list(f.keys())[0]





# Get the training data

h5_images = list(f['images'])

h5_labels = list(f['labels'])



plt.imshow(h5_images[0])  # graph it

plt.show()
h5_training_data = []

train_y = pd.Series(h5_labels)

IMG_SIZE = 50





def h5_create_training_data():

    for img in h5_images:

        try:

            new_img_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size

            h5_training_data.append(new_img_array)

        except Exception as e:  # in the interest in keeping the output clean...

            print(e)



        

h5_create_training_data()

print(len(h5_training_data))
h5_train_X = np.array(h5_training_data).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

h5_train_X.shape
flowers = pd.read_csv("../input/flower_images/flower_images/flower_labels.csv")



flowers.head()
imgs = flowers['file']

training_data = []

y = flowers['label'].values



def create_training_data():

    img_path = "../input/flower_images/flower_images/" 

    

    for im in imgs:

        try:

            img_array = cv2.imread(os.path.join(img_path,im))  # convert to array

            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size

            training_data.append(new_array)

        except Exception as e:  # in the interest in keeping the output clean...

            print(e)



create_training_data()

print(len(training_data))
flower_image = cv2.imread("../input/flower_images/flower_images/" + imgs[0])

rgb_flower_image = cv2.cvtColor(flower_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(3,3))

plt.imshow(rgb_flower_image);
Data = np.array(training_data).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

Data.shape
x_train, x_test, y_train, y_test = train_test_split(Data, y, 

                                                    test_size = 0.2, random_state = 1)

[x_train.shape, y_train.shape, x_test.shape, y_test.shape]
n = int(len(x_test)/2)

x_valid, y_valid = x_test[:n], y_test[:n]

x_test, y_test = x_test[n:], y_test[n:]

x_train.shape, x_test.shape, x_valid.shape, y_train.shape, y_test.shape, y_valid.shape
x_train = x_train.astype('float32')/255

x_test = x_test.astype('float32')/255

x_valid = x_valid.astype('float32')/255



c_y_train = to_categorical(y_train, 10)

c_y_test = to_categorical(y_test, 10)

c_y_valid = to_categorical(y_valid, 10)



[x_train.shape, c_y_train.shape, x_test.shape, c_y_test.shape, x_valid.shape, c_y_valid.shape]
# CNN

def cnn_mc_model():

    model = Sequential()



    model.add(Conv2D(32, (5, 5), padding='same', input_shape=x_train.shape[1:]))

    model.add(Activation('relu'))

    

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))



    model.add(Conv2D(96, (5, 5)))

    model.add(Activation('relu'))

    

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))



#     model.add(Flatten())

    model.add(GlobalAveragePooling2D())

    

    model.add(Dense(512, activation='tanh'))

    model.add(Dropout(0.25)) 

    

#    model.add(Dense(256, activation='tanh'))

#    model.add(Dropout(0.25)) 

    

    model.add(Dense(128, activation='tanh'))

    model.add(Dropout(0.25)) 



    model.add(Dense(10))

    model.add(Activation('softmax'))



    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

    

    return model



cnn_mc_model = cnn_mc_model()
cnn_mc_history = cnn_mc_model.fit(x_train, c_y_train, 

                                  epochs=50, batch_size=64, verbose=2,

                                  validation_data=(x_valid, c_y_valid))
cnn_mc_test_score = cnn_mc_model.evaluate(x_test, c_y_test)

cnn_mc_test_score