# Importing Libraries

import os,re,random

import cv2

import numpy as np

import pandas as pd

from keras import layers, models, optimizers

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K

from keras.preprocessing.image import img_to_array, load_img
# Reducing the size of img_width and img_height to 150

img_width = 96

img_height = 96

TRAIN_DIR = '../input/train/'

TEST_DIR = '../input/test/'



train_images_dogs_cats = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset

test_images_dogs_cats = [TEST_DIR+i for i in os.listdir(TEST_DIR)]

# print(train_images_dogs_cats)

# print(test_images_dogs_cats)

# We have to sort the values as the list is random

def atoi(text):

    return int(text) if text.isdigit() else text



def natural_keys(text):

    return [ atoi(c) for c in re.split('(\d+)', text) ]



train_images_dogs_cats.sort(key=natural_keys)

# print(train_images_dogs_cats)

# as First all starting images have cats and then all the images are of dogs 

# we take significant amount of data and train and test the model on it
# train_images_dogs_cats = train_images_dogs_cats[0:1000] + train_images_dogs_cats[12800:13800] 

test_images_dogs_cats.sort(key=natural_keys)

# print(train_images_dogs_cats)
print(len(train_images_dogs_cats))

print(len(test_images_dogs_cats))
# preparing data to linear data to feed to neural network

def prepare_data(list_of_images):

    """

    Returns two arrays: 

        x is an array of resized images

        y is an array of labels

    """

    x = [] # images as arrays

    y = [] # labels

    # there is a test name file in test folder which is not a image

    for image in list_of_images:

        x.append(cv2.resize(cv2.imread(image), (img_width,img_height), interpolation=cv2.INTER_CUBIC))

    

    for i in list_of_images:

        if 'dog' in i:

            y.append(1)

        elif 'cat' in i:

            y.append(0)

        #else:

            #print('neither cat nor dog name present in images')

            

    return x, y



for im in train_images_dogs_cats:

    try:

        cv2.resize(cv2.imread(im), (img_width,img_height), interpolation=cv2.INTER_CUBIC)

    except Exception as e:

        print(str(e))

        print(str(im))
train_images_dogs_cats.remove('../input/train/train')

X , Y = prepare_data(train_images_dogs_cats)

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2 , random_state = 101)
nb_train_samples = len(X_train)

nb_test_samples = len(X_test)

batch_size = 16
# for convulaitional 2d network

# initailise CNN

model = models.Sequential()

# defining 2D Network with 32 features and of shape =3*3 and input size of image

model.add(layers.Conv2D(32  ,(3 , 3), input_shape= (img_width, img_height , 3)))

# adding the activation as Rectified Linear Unit.

model.add(layers.Activation('relu'))

# max pooling with window size of 2*2

model.add(layers.MaxPooling2D(pool_size=(2,2)))



model.add(layers.Conv2D(32 ,(3,3)))

model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size= (2,2)))



model.add(layers.Conv2D(64 , (3,3)))

model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size= (2,2)))

# Flattening the CNN

model.add(layers.Flatten())

# Output of activation as relu of 64 output

model.add(layers.Dense(64))

model.add(layers.Activation('relu'))   

# Using dropout as 0.5

model.add(layers.Dropout(0.5))

# Using output as 1

model.add(layers.Dense(1))

# As using Sigmoid Activation to get probability output

model.add(layers.Activation('sigmoid'))



# Compiling options

model.compile(loss = 'binary_crossentropy' , optimizer = 'rmsprop' , metrics =['accuracy'])

model.summary()
# ImageDataGenerator class

# Generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches).



train_datagen = ImageDataGenerator(

    rescale=1./255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)

val_datagen  = ImageDataGenerator(

    rescale = 1./255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)

train_generated = train_datagen.flow(np.array(X_train), Y_train  , batch_size = batch_size)

val_generated =  val_datagen.flow(np.array(X_test) , Y_test , batch_size = batch_size)

history = model.fit_generator(

    train_generated, 

    steps_per_epoch=nb_train_samples // batch_size,

    epochs=50,

    validation_data=val_generated,

    validation_steps=nb_test_samples // batch_size

)
for im in test_images_dogs_cats:

    try:

        cv2.resize(cv2.imread(im), (img_width,img_height), interpolation=cv2.INTER_CUBIC)

    except Exception as e:

        print(str(e))

        print(str(im))
model.save_weights('model_wieghts.h5')

model.save('model_keras.h5')
test_images_dogs_cats.remove('../input/test/test')
X_t, Y_t = prepare_data(test_images_dogs_cats)

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_g = val_datagen.flow(np.array(X_t), batch_size=batch_size)

print(test_g)
prediction_probabilities = model.predict_generator(test_g , verbose=1 , steps = len(test_g))
print(prediction_probabilities.shape)
counter = range(1, len(test_images_dogs_cats) + 1)

solution = pd.DataFrame({"id": counter, "label":list(prediction_probabilities)})

cols = ['label']



for col in cols:

    solution[col] = solution[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)



solution.to_csv("dogsVScats.csv", index = False)