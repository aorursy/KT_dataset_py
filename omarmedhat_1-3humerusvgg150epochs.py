import numpy as np

import pandas as pd

import os

import cv2

import keras

from random import shuffle

from tqdm import tqdm

from keras.layers import Dense

from keras.optimizers import SGD, Adam

from keras.applications import vgg16

from joblib import dump, load

print("done")

# di = "../input/xr_elbow_train/XR_ELBOW_TRAIN"

# print(os.listdir(di))
TRAIN_DIR = "../input/xr_humerus_train/XR_HUMERUS_TRAIN"

print("ghhh")



postive = 1

negative = 0

IMG_SIZE = 224
def craete_label(class_name):

    label = np.zeros(2)

    label[class_name] = 1

    return label





def create_train_data():

    train_data = []

    m = 0

    take_part_of_data = int(len(os.listdir(TRAIN_DIR)) / 3)

    for item in tqdm(os.listdir(TRAIN_DIR)):

        m += 1

        if m == take_part_of_data:

           break

        patient_path = os.path.join(TRAIN_DIR, item)

        for patient_study in os.listdir(patient_path):

            type_of_study = patient_study.split('_')[1]

            if type_of_study == "positive":

                class_name = postive

            else:

                class_name = negative



            p_path = os.path.join(patient_path, patient_study)

            label = craete_label(class_name)



            for patient_image in os.listdir(p_path):

                image_path = os.path.join(p_path, patient_image)

                img = cv2.imread(image_path, 0)

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) 

                img = clahe.apply(img)

                img = np.divide(img, 255)

                #vert_img = cv2.flip(img, 1)

                train_data.append([np.array(img), label])  

                #train_data.append([np.array(vert_img), label]) 

    print("suffleing adta")

    shuffle(train_data)

    print("Data now save to disk")

    # dump(train_data, "train_data.joblib")

    return train_data



train_data = create_train_data()

print(len(train_data))
len(train_data)


X = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print("Train Image Load Succesfully")

print(X.shape)

y = np.array([i[1] for i in train_data])

print("Train Label Load Succeffully")

print(y.shape)
import keras

from keras.layers import Dense

from keras.models import Sequential

from keras.optimizers import Adam, SGD

from keras.layers import Conv2D, Activation, ZeroPadding2D, MaxPooling2D, Flatten, Dropout
model = Sequential()

model.add(ZeroPadding2D((1, 1), input_shape=X.shape[1:]))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(128, (3, 3)))

model.add(Activation('relu'))

model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(128, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(256, (3, 3)))

model.add(Activation('relu'))

model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(256, (3, 3)))

model.add(Activation('relu'))

model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(256, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(512, (3, 3)))

model.add(Activation('relu'))

model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(512, (3, 3)))

model.add(Activation('relu'))

model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(512, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(512, (3, 3)))

model.add(Activation('relu'))

model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(512, (3, 3)))

model.add(Activation('relu'))

model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(512, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



model.add(Flatten())

model.add(Dense(4096))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(4096))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(2))

model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, nesterov=True)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, validation_split=0.2, epochs=150)

model.save('VGG_HUMERUS150epoch.h5')
TEST_DIR = "../input/xr_humerus_valid/XR_HUMERUS_VALID"

def create_train_test_data():

    data = []

    take_part_of_data = int(len(os.listdir(TEST_DIR)) / 3)

    m=0

    for item in tqdm(os.listdir(TEST_DIR)):

        m += 1

        if m == take_part_of_data:

           break

        patient_path = os.path.join(TEST_DIR, item)

        for patient_study in os.listdir(patient_path):

            type_of_study = patient_study.split('_')[1]

            if type_of_study == "positive":

                class_name = postive

            else:

                class_name = negative



            p_path = os.path.join(patient_path, patient_study)

            label = craete_label(class_name)



            for patient_image in os.listdir(p_path):

                image_path = os.path.join(p_path, patient_image)

                img = cv2.imread(image_path, 0)



                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

               

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) 

                img = clahe.apply(img)

                img = np.divide(img, 255)



                data.append([np.array(img), label])

              

    print("suffleing adta")

    shuffle(data)

    print("Data now save to disk")

    return data

test_data = create_train_test_data()
x_test = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print("Train Image Load Succesfully")

print(x_test.shape)

y_test = np.array([i[1] for i in test_data])

print("Train Label Load Succeffully")

print(y_test.shape)
result = model.predict(x_test)
len(result)
result_v1 = model.predict_classes(x_test)
result_v1
y_test
res = np.zeros(len(y_test))

for i in range(len(y_test)):

    res[i] = np.argmax(y_test[i])

print(res[0])

print(result_v1[0])

print(len(res))

print(len(result_v1))
count = 0

for i in range(len(res)):

    if int(res[i]) == result_v1[i] :

        count += 1

        

print("Test Accuracy : ", count / len(res) * 100 , "%")