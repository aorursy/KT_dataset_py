# for numerical things

import numpy as np



# opencv & matplotlib to deal with images

import cv2

import matplotlib.pyplot as plt



# os for file system related tasks

import os



# random to fix seeds

import random

import tensorflow as tf

import torch



# import keras to build CNN model

from keras.models import Model

from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import plot_model
CATS_PATH = "../input/cat-and-dog/training_set/training_set/cats/"



cats_images_paths = os.listdir(CATS_PATH)

len(cats_images_paths)
# randomly select 5 images.

cats_5_images = random.sample(cats_images_paths, k=5)



for img in cats_5_images:

    cv2_img = cv2.imread(CATS_PATH + img)

    plt.figure()

    plt.imshow(cv2_img)
DOGS_PATH = "../input/cat-and-dog/training_set/training_set/dogs/"



dogs_images_paths = os.listdir(DOGS_PATH)

len(dogs_images_paths)
# randomly select 5 images.

dogs_5_images = random.sample(dogs_images_paths, k=5)



for img in dogs_5_images:

    cv2_img = cv2.imread(DOGS_PATH + img)

    plt.figure()

    plt.imshow(cv2_img)
def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    tf.random.set_seed(seed)



# We fix all the random seed so that, we can reproduce the results.

seed_everything(2020)
# images shape

IMAGE_SHAPE = 128



input_layer = Input(shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))



convolution_layer_1 = Conv2D(16, kernel_size=(3,3), activation = 'relu')(input_layer)

convolution_layer_2 = Conv2D(16, kernel_size=(3,3), activation = 'relu')(convolution_layer_1)

max_pool_1 = MaxPooling2D(pool_size=(2,2))(convolution_layer_2)

batch_norm_1 = BatchNormalization()(max_pool_1)

dropout_1 = Dropout(0.2)(batch_norm_1)



convolution_layer_3 = Conv2D(32, kernel_size=(3,3), activation = 'relu')(dropout_1)

convolution_layer_4 = Conv2D(32, kernel_size=(3,3), activation = 'relu')(convolution_layer_3)

max_pool_2 = MaxPooling2D(pool_size=(2,2))(convolution_layer_4)

batch_norm_2 = BatchNormalization()(max_pool_2)

dropout_2 = Dropout(0.2)(batch_norm_2)



flattened = Flatten()(dropout_2)

dense_layer_1 = Dense(128, activation='relu')(flattened)

dense_layer_2 = Dense(64, activation='relu')(dense_layer_1)

output_layer = Dense(1, activation='sigmoid')(dense_layer_2)



model = Model(input=input_layer, output=output_layer)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
plot_model(model, show_shapes=True)
train_data_generator = ImageDataGenerator(

    # We divide each pixel value(0, 255) with 255 to make them in range [0, 1]

    rescale = 1./255, 

    

    # We randomly shear & zoom our image while training to make our training robust

    shear_range = 0.2, 

    zoom_range = 0.2, 

    

    # We also flip our images by 180 degree horizontally to make our training robust

    horizontal_flip = True

)



test_data_generator = ImageDataGenerator(

    rescale = 1./255

)



training_set_data = train_data_generator.flow_from_directory(

    "../input/cat-and-dog/training_set/training_set/", 

    target_size = (IMAGE_SHAPE, IMAGE_SHAPE), 

    batch_size = 64,

    class_mode = 'binary'

)



test_set_data = test_data_generator.flow_from_directory(

    '../input/cat-and-dog/test_set/test_set/',

    target_size = (IMAGE_SHAPE, IMAGE_SHAPE),

    batch_size = 64, 

    class_mode = 'binary'

)
model.fit_generator(

    training_set_data, 

    epochs = 10, 

    validation_data = test_set_data, 

)
TEST_PATH = "../input/cat-and-dog/test_set/test_set/"

test_dogs_images = os.listdir(TEST_PATH + "dogs/")



test_dog_img = test_dogs_images[10]

test_dog_img = cv2.imread(TEST_PATH + "dogs/" + test_dog_img)



plt.figure()

plt.imshow(test_dog_img)



test_dog_img = cv2.resize(test_dog_img / 255, (IMAGE_SHAPE, IMAGE_SHAPE))

test_dog_img = test_dog_img.reshape(1, IMAGE_SHAPE, IMAGE_SHAPE, 3)



prediction = model.predict(test_dog_img)



if prediction[0][0] <= 0.5:

    print("Model : It's a CAT")

else:

    print("Model : It's a DOG")
TEST_PATH = "../input/cat-and-dog/test_set/test_set/"

test_cats_images = os.listdir(TEST_PATH + "cats/")



test_cat_img = test_cats_images[10]

test_cat_img = cv2.imread(TEST_PATH + "cats/" + test_cat_img)



plt.figure()

plt.imshow(test_cat_img)



test_cat_img = cv2.resize(test_cat_img / 255, (IMAGE_SHAPE, IMAGE_SHAPE))

test_cat_img = test_cat_img.reshape(1, IMAGE_SHAPE, IMAGE_SHAPE, 3)



prediction = model.predict(test_cat_img)



if prediction[0][0] <= 0.5:

    print("Model : It's a CAT")

else:

    print("Model : It's a DOG")