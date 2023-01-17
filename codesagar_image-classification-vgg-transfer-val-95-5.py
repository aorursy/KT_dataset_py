import numpy as np

import os

from sklearn.metrics import confusion_matrix

import seaborn as sn; sn.set(font_scale=1.4)

from sklearn.utils import shuffle           

import matplotlib.pyplot as plt             

import cv2                                 

import tensorflow as tf    
class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']

class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

IMAGE_SIZE = (150, 150)
## Thanks to Vincent for this function - https://www.kaggle.com/vincee/intel-image-classification-cnn-keras



## Function definition

def load_data():

    """

        Load the data:

            - 14,034 images to train the network.

            - 3,000 images to evaluate how accurately the network learned to classify images.

    """

    

    datasets = ['/kaggle/input/intel-image-classification/seg_train/seg_train', '/kaggle/input/intel-image-classification/seg_test/seg_test']

    output = []

    

    # Iterate through training and test sets

    for dataset in datasets:

        

        images = []

        labels = []

        

        print("Loading {}".format(dataset))

        

        # Iterate through each folder corresponding to a category

        for folder in os.listdir(dataset):

            curr_label = class_names_label[folder]

            

            # Iterate through each image in our folder

            for file in os.listdir(os.path.join(dataset, folder)):

                

                # Get the path name of the image

                img_path = os.path.join(os.path.join(dataset, folder), file)

                

                # Open and resize the img

                curr_img = cv2.imread(img_path)

                curr_img = cv2.resize(curr_img, IMAGE_SIZE) 

                

                # Append the image and its corresponding label to the output

                images.append(curr_img)

                labels.append(curr_label)

                

        images = np.array(images, dtype = 'float32')

        labels = np.array(labels, dtype = 'int32')   

        

        output.append((images, labels))



    return output



## Function Call

(train_images, train_labels), (test_images, test_labels) = load_data()

train_images = train_images / 255.0 

test_images = test_images / 255.0



train_images, train_labels = shuffle(train_images, train_labels)
def display_examples(class_names, images, labels):

    """

        Display 25 images from the images array with its corresponding labels

    """

    

    fig = plt.figure(figsize=(20,20))

    fig.suptitle("Some examples of images of the dataset", fontsize=16)

    for i in range(25):

        plt.subplot(5,5,i+1)

        plt.xticks([])

        plt.yticks([])

        plt.grid(False)

        plt.imshow(images[i])

        plt.xlabel(class_names[labels[i]])

    plt.show()



    ## Function call

display_examples(class_names, train_images, train_labels)

from keras.utils import np_utils

y_train = np_utils.to_categorical(train_labels, 6)

y_test = np_utils.to_categorical(test_labels, 6)
from keras.applications import VGG16

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Activation, Conv2D, Flatten, BatchNormalization, Dropout, MaxPool2D

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, EarlyStopping



vgg_conv = VGG16(weights='imagenet',

                  include_top=False,input_shape=(150,150,3))

## Disabling training from VGG layers

vgg_conv.trainable=False



## Instantiating model

vgg_conv.trainable=False

transfer_model = Sequential()

transfer_model.add(vgg_conv)

transfer_model.add(Flatten())

transfer_model.add(Dropout(0.25))

transfer_model.add(Dense(64, activation='relu'))

transfer_model.add(Dropout(0.25))

transfer_model.add(Dense(6, activation='softmax'))





## Model summary

transfer_model.summary()
optimizer = Adam(lr=0.2e-3, beta_1=0.9, beta_2=0.999, amsgrad=False)



transfer_model.compile(optimizer,loss='binary_crossentropy',metrics=["accuracy"])



## I know binary_crossentropy is not recommended for multiclass problems.

## But I saw somebody use it and surprisingly it gave better results.
train_gen = ImageDataGenerator(

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        fill_mode='nearest')



# fits the model on batches with real-time data augmentation:

history = transfer_model.fit_generator(train_gen.flow(train_images, y_train, batch_size=32),

                              steps_per_epoch=len(train_images) / 32, epochs=3,

                              validation_data = (test_images, y_test))