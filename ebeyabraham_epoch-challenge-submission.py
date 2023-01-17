import os

import cv2

import random

import numpy as np

from imutils import paths

from keras.models import Sequential, Model

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dense,Dropout

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam,RMSprop

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

from keras.preprocessing.image import img_to_array

from keras.utils import to_categorical

import csv
#get all the input data paths and randomly shuffle them

data_paths = sorted(list(paths.list_images("../input/hestia-epoch/train data/Train data")))

#more_data_paths = sorted(list(paths.list_images("../input/hestia-epoch/train data/Train data/non-leukocoria")))

#more_data_paths = random.sample(more_data_paths,len(data_paths))

#data_paths += more_data_paths



random.seed(1203)

random.shuffle(data_paths)



data = []

labels = []

encode = {"leukocoria":1,"non-leukocoria":0}



#add img array to data

for path in data_paths:

    #read image from image path

    img = cv2.imread(path)

    img = cv2.resize(img,(100,100))

    img = img_to_array(img)

    #cv2.normalize(img,img, 0, 255, cv2.NORM_MINMAX)

    #add images to data list

    data.append(img)



    #add encoded lable to label list

    label = path.split(os.path.sep)[-2]

    labels.append(encode[label])

    

    
#scaling data

data = np.array(data, dtype = float) / 255.0

labels = np.array(labels)



#train-test split

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0, random_state = 1203)



#one hot encoding of labels

trainY = to_categorical(trainY, num_classes = 2)

testY = to_categorical(testY, num_classes = 2)



#data augmentation for image generator

#aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,horizontal_flip=True)

'''

#some parameters for the model

height = 28

width = 28

depth = 3

classes = 2



model = Sequential()

input_shape = (height, width, depth)

model.add(Conv2D(20,(5,5), padding = "same", input_shape = input_shape))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))

model.add(Conv2D(50, (5,5), padding = "same"))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))

model.add(Flatten())

model.add(Dense(500))

model.add(Activation("relu"))



#softmax

model.add(Dense(classes))

model.add(Activation("softmax"))

'''

from keras.applications.xception import Xception, preprocess_input



height = 100

width = 100

depth = 3

classes = 2

base_model = Xception(weights='imagenet',include_top=False,input_shape=(height,width,depth))



train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input, rotation_range = 90, horizontal_flip = 30)



def build_finetune_model(base_model, dropout, fc_layers, num_classes):

    for layer in base_model.layers:

        layer.trainable = False



    x = base_model.output

    x = Flatten()(x)

    for fc in fc_layers:

        # New FC layer, random init

        x = Dense(fc, activation='relu')(x) 

        x = Dropout(dropout)(x)



    # New softmax layer

    predictions = Dense(num_classes, activation='softmax')(x) 

    

    finetune_model = Model(inputs=base_model.input, outputs=predictions)



    return finetune_model



FC_LAYERS = [500]

dropout = 0.5



finetune_model = build_finetune_model(base_model, dropout=dropout, fc_layers=FC_LAYERS, num_classes = classes) 
from math import ceil

EPOCHS = 25

INIT_LR = 1e-3

BS = 32



print("[Compiling the Model]")

opt = Adam(lr = INIT_LR, decay=INIT_LR/EPOCHS)

#opt = RMSprop(lr = INIT_LR, decay=INIT_LR/EPOCHS)

finetune_model.compile(loss="binary_crossentropy", optimizer=opt, metrics = ["accuracy"])



print("[Training]")

H = finetune_model.fit_generator(train_datagen.flow(trainX, trainY, batch_size = BS), steps_per_epoch = ceil(len(trainX) / BS), epochs = EPOCHS, verbose = 1)

data_paths = sorted(list(paths.list_images("../input/hestia-epoch/evaluation data/Evaluation data")))

a = 0.4

b = 0.6



#variable for writing to the csv file

with open("submission2.csv",'a') as f:

    writer = csv.writer(f)

    writer.writerow(["Id","Category"])

    for path in data_paths:

        img = cv2.imread(path)

        img = cv2.resize(img,(100,100))

        img = img.astype("float")/255.0

        img = img_to_array(img)

        img = np.expand_dims(img, axis = 0)



        (neg, pos) = finetune_model.predict(img)[0]

        neg = (neg + a)/2

        pos = (pos + b)/2

        label = 0 if neg > pos else 1

        i = path.split(os.path.sep)[-1]

        i = i[:-4]

        

        row = [i, label]

        print(row)

        #print(i,neg, pos)

        writer.writerow(row)

        

f.close()

        

        

    

    

    