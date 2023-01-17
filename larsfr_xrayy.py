import numpy as np

import matplotlib.pyplot as plt

import os

import cv2

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_path= '../input/chest-xray-pneumonia/chest_xray/train/'

val_path = '../input/chest-xray-pneumonia/chest_xray/val/'

test_path = '../input/chest-xray-pneumonia/chest_xray/test/'



def count_files(path):

    num_normal = 0

    num_pneumonia_virus = 0

    num_pneumonia_bacteria = 0

    for folder_name in os.listdir(path):

        folder_path = path + folder_name

        for image_name in os.listdir(folder_path):

            if 'virus' in image_name:

                num_pneumonia_virus += 1

            elif 'bacteria' in image_name:

                num_pneumonia_bacteria += 1

            elif 'virus' not in image_name and 'bacteria' not in image_name:

                num_normal += 1

            else:

                raise Exception('Unhandled image!')

    num_pneumonia_combined = num_pneumonia_virus + num_pneumonia_bacteria

    return num_normal, num_pneumonia_combined, num_pneumonia_virus, num_pneumonia_bacteria





train_num_normal, train_num_combined, train_num_virus, train_num_bacteria = count_files(train_path)
TRAIN_DIR = "../input/chest-xray-pneumonia/chest_xray/train/"

VAL_DIR = "../input/chest-xray-pneumonia/chest_xray/val/"

TEST_DIR = "../input/chest-xray-pneumonia/chest_xray/test/"



DIRECTORIES = [TRAIN_DIR, VAL_DIR, TEST_DIR]

CATEGORIES = ["NORMAL","PNEUMONIA"]



for category in CATEGORIES:

    path = os.path.join(TRAIN_DIR, category)

    for img in os.listdir(path):

        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

        plt.imshow(img_array, cmap="gray")

        plt.show()

        break

    break
normal_data = []

bacteria_data = []

virus_data = []

IMG_SIZE = 200



def create_splitted_data():

    for directory in DIRECTORIES:

        for category in CATEGORIES:

            path = os.path.join(directory, category)

            #index: NORMAL --> 0, PNEUMONIA --> 1

            class_num = CATEGORIES.index(category)

            for img in os.listdir(path):

                try:

                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                    if directory == TRAIN_DIR:

                        if 'virus' in img:

                            virus_data.append([new_array, class_num])

                        elif 'bacteria' in img:

                            bacteria_data.append([new_array, class_num])

                        elif 'virus' not in img and 'bacteria' not in img:

                            normal_data.append([new_array, class_num])

                except Exception as e:

                    print(e)

                    

create_splitted_data()
import random



random.shuffle(normal_data)

random.shuffle(bacteria_data)

random.shuffle(virus_data)



X_normal = []

y_normal = []

X_bacteria = []

y_bacteria = []

X_virus = []

y_virus = []



for features, label in normal_data:

    X_normal.append(features)

    y_normal.append(label)

    

X_normal = np.array(X_normal).reshape(-1, IMG_SIZE, IMG_SIZE, 1)



for features, label in bacteria_data:

    X_bacteria.append(features)

    y_bacteria.append(label)

    

X_bacteria = np.array(X_bacteria).reshape(-1, IMG_SIZE, IMG_SIZE, 1)



for features, label in virus_data:

    X_virus.append(features)

    y_virus.append(label)

    

X_virus = np.array(X_virus).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

max_num_category = max(len(normal_data),len(bacteria_data),len(virus_data))

AUG_PATH='aug_images/'

num_images_per_category=2500

os.makedirs(AUG_PATH+'NORMAL')

os.makedirs(AUG_PATH+'PNEUMONIA')
from matplotlib import pyplot

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator



def augment_smaller_classes(X_data, y_data, num_images_per_category, name, prefix):

    # fit parameters from data

    data_generator_with_aug.fit(X_data)

    i = 0

    for batch in data_generator_with_aug.flow(X_data, y_data, batch_size=10, save_to_dir=(AUG_PATH+name), save_prefix=prefix, save_format='jpeg'):

        i += 1

        if i > int(num_images_per_category/10):

            break 

    

        



num_of_most_occuring_class = max(len(normal_data), len(bacteria_data), len(virus_data))



data_generator_with_aug = ImageDataGenerator(

                                   width_shift_range = 0.2,

                                   height_shift_range = 0.2,

                                   zoom_range=0.2,

                                   fill_mode='nearest')



augment_smaller_classes(X_normal, y_normal, (num_of_most_occuring_class-train_num_normal), 'NORMAL', 'normal')

augment_smaller_classes(X_bacteria, y_bacteria, (num_of_most_occuring_class-train_num_bacteria), 'PNEUMONIA', 'bacteria')

augment_smaller_classes(X_virus, y_virus, (num_of_most_occuring_class-train_num_virus), 'PNEUMONIA', 'virus')





AUG_DIR = "../working/aug_images/"

aug_data = []

def reload_created_data():

        for category in CATEGORIES:

            path = os.path.join(AUG_DIR, category)

            #index: NORMAL --> 0, PNEUMONIA --> 1

            class_num = CATEGORIES.index(category)

            for img in os.listdir(path):

                try:

                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                    aug_data.append([new_array, class_num])

                except Exception as e:

                    print(e)

                    

reload_created_data()
aug_num_normal, aug_num_combined, aug_num_virus, aug_num_bacteria = count_files(AUG_DIR)

train_num_normal, train_num_combined, train_num_virus, train_num_bacteria = count_files(train_path)

print(aug_num_normal)

print(aug_num_virus)

print(aug_num_bacteria)
training_data = []

validation_data = []

test_data = []



def create_data():

    for directory in DIRECTORIES:

        for category in CATEGORIES:

            path = os.path.join(directory, category)

            #index: NORMAL --> 0, PNEUMONIA --> 1

            class_num = CATEGORIES.index(category)

            for img in os.listdir(path):

                try:

                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                    if directory == TRAIN_DIR:

                        training_data.append([new_array, class_num])

                    elif directory == VAL_DIR:

                        validation_data.append([new_array, class_num])

                    elif directory == TEST_DIR:

                        test_data.append([new_array, class_num])

                except Exception as e:

                    print("faulty image")

                

create_data()
training_data = training_data + aug_data
print(len(training_data))

print(len(validation_data))

print(len(test_data))
#shuffle data to improve learning behavior

import random



random.shuffle(training_data)

#random.shuffle(validation_data)

random.shuffle(test_data)
X_train = []

y_train = []



#X_val = []

#y_val = []



X_test = []

y_test = []
for features, label in training_data:

    X_train.append(features)

    y_train.append(label)

    

X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#for features, label in validation_data:

#    X_val.append(features)

#    y_val.append(label)

    

#X_val = np.array(X_val).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
for features, label in test_data:

    X_test.append(features)

    y_test.append(label)

    

X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#save data set

import pickle



pickle_out = open("X_train.pickle", "wb")

pickle.dump(X_train, pickle_out)

pickle_out.close()



pickle_out = open("y_train.pickle", "wb")

pickle.dump(y_train, pickle_out)

pickle_out.close()



#pickle_out = open("X_val.pickle", "wb")

#pickle.dump(X_val, pickle_out)

#pickle_out.close()



#pickle_out = open("y_val.pickle", "wb")

#pickle.dump(y_val, pickle_out)

#pickle_out.close()



pickle_out = open("X_test.pickle", "wb")

pickle.dump(X_test, pickle_out)

pickle_out.close()



pickle_out = open("y_test.pickle", "wb")

pickle.dump(y_test, pickle_out)

pickle_out.close()
#load dataset

import pickle



X_train = pickle.load(open("X_train.pickle", "rb"))

y_train = pickle.load(open("y_train.pickle", "rb"))



#X_val = pickle.load(open("X_val.pickle", "rb"))

#y_val = pickle.load(open("y_val.pickle", "rb"))



X_test = pickle.load(open("X_test.pickle", "rb"))

y_test = pickle.load(open("y_test.pickle", "rb"))
import numpy as np

X_train = np.asarray(X_train)

y_train = np.asarray(y_train)

#X_val = np.asarray(X_val)

#y_val = np.asarray(y_val)



X_train = X_train/255.0

#X_val = X_val/255.0
import tensorflow as tf

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential 

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from tensorflow.keras.callbacks import TensorBoard

import time



NAME = "pneumonia-cnn-64x2-3ep-{}".format(time.strftime("%Y%m%d-%H%M%S"))



tensorboard = TensorBoard(log_dir = "logs/{}".format(NAME))



model = Sequential()



model.add(Conv2D(64, (3,3), input_shape = X_train.shape[1:]))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(64, (3,3)))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())



model.add(Dense(64))

model.add(Activation("relu"))



model.add(Dense(1))

model.add(Activation("sigmoid"))



model.compile(loss="binary_crossentropy", 

              optimizer="adam", 

              metrics=["accuracy"])



model.fit(X_train, y_train, batch_size=32, epochs=3, validation_split=0.4, callbacks = [tensorboard])