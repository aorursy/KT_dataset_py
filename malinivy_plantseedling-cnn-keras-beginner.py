# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np



import os

import sys

import cv2

from keras.utils import to_categorical

import matplotlib

from keras import backend as k

k.clear_session()
import random

random.seed(10)

allLabels =  os.listdir("../input/plant-seedlings-classification/train/")  # list of subdirectories and files

trainDir='/kaggle/working/../input/plant-seedlings-classification/train/'



from keras.preprocessing.image import  img_to_array, load_img

WIDTH = 128

HEIGHT = 128

DEPTH = 3



data = []

labels = []



# loop over the input images

dirs = os.listdir(trainDir) 

for dir in dirs:

    absDirPath = os.path.join(os.path.sep,trainDir, dir)

    images = os.listdir(absDirPath)

    for imageFileName in images:

        

        # load the image, pre-process it, and store it in the data list

        imageFullPath = os.path.join(trainDir, dir, imageFileName)

        #print(imageFullPath)

        img = load_img(imageFullPath)

        arr = img_to_array(img)  # Numpy array with shape (233,233,3)

        arr = cv2.resize(arr, (HEIGHT,WIDTH)) #Numpy array with shape (HEIGHT, WIDTH,3)

        #print(arr.shape)

        data.append(arr)

        #label = classes_to_int(dir)

        label=str(imageFullPath.split('/')[-2])

        #print(label)

        labels.append(label)



  
len(images)

print('Number of images :-',len(data))

print('Numbe of Labels',len(labels))
data[0]
%matplotlib inline

import os

import matplotlib

import matplotlib.pyplot as plt

#to view the images



for i in range(1,10):

    #print(i)

    new_image = tf.keras.preprocessing.image.array_to_img(data[i])

    #Show image

    #fig, axs = plt.subplots(1, j, figsize=(20, 20))

    plt.imshow(new_image)

    plt.show()

    



from sklearn.preprocessing import LabelEncoder



# scale the raw pixel intensities to the range [0, 1]

TrainX = np.array(data, dtype="float") / 255.0

Y_labels = np.array(labels)

# convert the labels from integers to vectors

#Y =  to_categorical(Y, num_classes=12)

labelEncoder = LabelEncoder()

labelEncoder.fit(Y_labels)

train_labels_encoded = labelEncoder.transform(Y_labels)

trainY = tf.keras.utils.to_categorical(train_labels_encoded, num_classes=12)

        
print(TrainX.shape)

print(trainY.shape)
from sklearn.model_selection import train_test_split

print("Train Validation Split into 80:20...")

sys.stdout.flush()

# partition the data into training and validation splits 

(x_train, valX, y_train, valY) = train_test_split(TrainX,trainY,test_size=0.20, random_state=10)

from keras.preprocessing.image import ImageDataGenerator



aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, 

    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,

   horizontal_flip=True, fill_mode="nearest")
from keras import backend as k

from keras.models import Sequential

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers.core import Activation, Flatten, Dense

from keras.optimizers import Adam

# initialize the model



sys.stdout.flush()

k.clear_session()



inputShape = (WIDTH, HEIGHT, DEPTH)

EPOCHS = 40

INIT_LR = 1e-3

BS = 32



model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape)) 

model.add(Activation("relu"))

#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (5, 5), padding="same"))

model.add(Activation("relu"))

# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3, 3), padding="same"))

model.add(Activation("relu"))

#  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(units=500))

model.add(Activation("relu"))



# softmax classifier

model.add(Dense(units=12))

model.add(Activation("softmax"))

   

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

model.summary()

# train the network



sys.stdout.flush()



H = model.fit_generator(aug.flow(x_train, y_train, batch_size=BS), 

                        validation_data=(valX, valY), 

                        steps_per_epoch=len(x_train) // BS, 

                        epochs=EPOCHS, verbose=1)



from matplotlib import pyplot

sys.stdout.flush()



pyplot.style.use("ggplot")

pyplot.figure()

N = EPOCHS



pyplot.plot(np.arange(0, N), H.history["acc"], label="train_acc")

pyplot.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")

pyplot.title("Training /Validation and Accuracy on  crop classification")

pyplot.xlabel("Epoch #")

pyplot.ylabel("Accuracy")

pyplot.legend(loc="lower left")



from matplotlib import pyplot

sys.stdout.flush()



pyplot.style.use("ggplot")

pyplot.figure()

N = EPOCHS

pyplot.plot(np.arange(0, N), H.history["loss"], label="train_loss")

pyplot.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")



pyplot.title("Training /Validation Loss on  crop classification")

pyplot.xlabel("Epoch #")

pyplot.ylabel("Loss")

pyplot.legend(loc="lower left")

#allLabels =  os.listdir("../input/test/")  # list of subdirectories and files

testDir='/kaggle/working/../input/plant-seedlings-classification/test/'



from keras.preprocessing.image import  img_to_array, load_img

WIDTH = 128

HEIGHT = 128

DEPTH = 3

Testdata = []

filenames = []

    # loop over the input images

images = os.listdir(testDir)

for imageFileName in images:

 # load the image, pre-process it, and store it in the data list

    imageFullPath = os.path.join(testDir, imageFileName)

    #print(imageFullPath)

    img = load_img(imageFullPath)

    arr = img_to_array(img)  # Numpy array with shape (...,..,3)

    arr = cv2.resize(arr, (HEIGHT,WIDTH)) 

    Testdata.append(arr)

    filenames.append(imageFileName)

   



# scale the raw pixel intensities to the range [0, 1]

testX = np.array(Testdata, dtype="float") / 255.0

testX.shape
len(filenames)
print(filenames[0])
predicted_probs = model.predict(testX, batch_size=10, verbose=1)



#predicted_probs = model.predict_generator(test_generator(test_files), steps=len(test_files))

#[f.split('/')[3] for f in test_files]

predicted_classes = np.argmax(predicted_probs, axis=1)

out_df = pd.DataFrame({'file':filenames , 

                       'species': labelEncoder.inverse_transform(predicted_classes)})

out_df.to_csv('submission_conv.csv', index=False)
#tf.keras.applications

from tensorflow.python.keras.applications import ResNet50

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D





#RESNET_WEIGHTS_PATH = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

#RESNET_WEIGHTS_PATH = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



model2 = Sequential()

#model2 = Sequential()

model2.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))



#model.add(ResNet50(include_top = False, pooling = RESNET50_POOLING_AVERAGE, weights = resnet_weights_path))





#model2.add(ResNet50(include_top=False, pooling='max', weights= 'imagenet'))

#model2.layers[0].trainable = True



# Say not to train first layer (ResNet) model as it is already trained

model2.layers[0].trainable = False



from tensorflow.keras.layers import Dropout

#get Output layer of Pre0trained model

x = model2.output



#Flatten the output to feed to Dense layer

x = tf.keras.layers.Flatten()(x)



x = Dense(1024, activation="relu")(x)

x = Dropout(0.5)(x)



#Add one Dense layer

x = tf.keras.layers.Dense(1024, activation='relu')(x)



#Add output layer

prediction = tf.keras.layers.Dense(12,activation='softmax')(x)



#Using Keras Model class

final_model = tf.keras.models.Model(inputs=model2.input, #Pre-trained model input as input layer

                                    outputs=prediction) #Output layer added



#optimizer = optimizers.SGD(lr=0.0001, momentum=0.9)

final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



from keras.callbacks import ModelCheckpoint, EarlyStopping

early = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1, mode='auto')

H = final_model.fit_generator(aug.flow(x_train, y_train, batch_size=BS), 

                                validation_data=(valX, valY), 

                                steps_per_epoch=len(x_train) // BS, 

                                epochs=EPOCHS, verbose=1,

                                callbacks = [early])
predicted_probs2 = final_model.predict(testX, batch_size=10, verbose=1)



#predicted_probs = model.predict_generator(test_generator(test_files), steps=len(test_files))

#[f.split('/')[3] for f in test_files]

predicted_classes2 = np.argmax(predicted_probs2, axis=1)

out_df = pd.DataFrame({'file':filenames , 

                       'species': labelEncoder.inverse_transform(predicted_classes2)})

out_df.to_csv('submissionTL.csv', index=False)
from matplotlib import pyplot

sys.stdout.flush()



pyplot.style.use("ggplot")

pyplot.figure()

N = EPOCHS



pyplot.plot(np.arange(0, N), H.history["acc"], label="train_acc")

pyplot.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")

pyplot.title("Training /Validation and Accuracy on  crop classification with Transfer Learning")

pyplot.xlabel("Epoch #")

pyplot.ylabel("Accuracy")

pyplot.legend(loc="lower left")