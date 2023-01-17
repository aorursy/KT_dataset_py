import cv2                                         # working with, mainly resizing, images

import numpy as np                                 # dealing with arrays

import os                                          # dealing with directories

from random import shuffle                         # mixing up or currently ordered data that might lead our network astray in training.

from keras.models import Sequential                # creating sequential model of CNN

from keras.layers import Convolution2D             # creating convolution layer

from keras.layers import MaxPooling2D              # creating maxpool layer

from keras.layers import Flatten                   # creating input vector for dense layer

from keras.layers import Dense                     # create dense layer or fully connected layer

from keras.layers import Dropout                   # use to avoid overfitting by droping some parameters

from keras.preprocessing import image              # generate image

import matplotlib.pyplot as plt                    # use for visualization

import warnings#

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))

TRAIN_DIR = '../input/training_set/training_set'

TEST_DIR = '../input/test_set/test_set'

IMG_SIZE = 64,64
image_names = []

data_labels = []

data_images = []
def  create_data(DIR):

     for folder in os.listdir(TRAIN_DIR):

        for file in os.listdir(os.path.join(TRAIN_DIR,folder)):

            if file.endswith("jpg"):

                image_names.append(os.path.join(TRAIN_DIR,folder,file))

                data_labels.append(folder)

                img = cv2.imread(os.path.join(TRAIN_DIR,folder,file))

                im = cv2.resize(img,IMG_SIZE)

                data_images.append(im)

            else:

                continue
#calling functions to create data

create_data(TRAIN_DIR)

create_data(TEST_DIR)
data = np.array(data_images)
len(data_images)
data.shape
from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils



le = LabelEncoder()

label = le.fit_transform(data_labels)

from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val=train_test_split(data,label,test_size=0.20,random_state=42)



print("X_train shape",X_train.shape)

print("X_test shape",X_val.shape)

print("y_train shape",y_train.shape)

print("y_test shape",y_val.shape)
classifier=Sequential()

classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.5))

classifier.add(Flatten())

classifier.add(Dense(output_dim= 128, activation='relu'))

classifier.add(Dense(output_dim= 1, activation='sigmoid'))

classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

classifier.summary()
from keras.preprocessing.image import ImageDataGenerator



train_datagen=ImageDataGenerator(

    rescale=1./255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



train_datagen.fit(X_train)
batch_size = 32

steps_per_epoch=len(X_train)

validation_steps=len(y_val)



history=classifier.fit_generator(

    train_datagen.flow(X_train,y_train, batch_size=batch_size),

    steps_per_epoch = steps_per_epoch,

    epochs = 4,

    verbose = 2,

    validation_data = (X_val,y_val),

    validation_steps = validation_steps)
classifier.save_weights('model.h5')
from sklearn.metrics import confusion_matrix

import seaborn as sns



pred = classifier.predict_classes(X_val)

cm = confusion_matrix(y_val,pred)



f,ax = plt.subplots(figsize=(4, 4))

sns.heatmap(cm, annot=True, linewidths=0.01,cmap="Purples",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()
accuracy=(cm[0][0]+cm[1][1])/len(y_val)

print(accuracy)
import numpy as np

from keras.preprocessing import image





test_image=image.load_img('../input/test_set/test_set/dogs/dog.4042.jpg',target_size=(64,64))

test_image=image.img_to_array(test_image)

test_image=np.expand_dims(test_image,axis=0)

result=classifier.predict_classes(test_image)



if result[0][0] >=0.5:

    prediction='dog'

else:

    prediction='cat'

print(prediction)