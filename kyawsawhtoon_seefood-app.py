import os

import numpy as np

from PIL import Image

from skimage.io import imread

import tensorflow as tf

import tensorflow_datasets as tfds

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from keras.applications.vgg16 import VGG16, preprocess_input

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Dense

from keras import backend as K
test_path = "../input/hot-dog-not-hot-dog/test"

train_path = "../input/hot-dog-not-hot-dog/train"
# Creating a list of labels

labels = os.listdir(train_path)



# Viewing the labels

labels
# get all the data in the directory split/train and reshape them

train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(

        train_path, 

        target_size=(227, 227), batch_size=500)



# get all the data in the directory split/test and reshape them

test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(

        test_path, 

        target_size=(227, 227), batch_size = 500) 



# Assigning image and labels to the variables



train_images, train_labels = next(train_generator)

test_images, test_labels = next(test_generator)
# Splitting a validation dataset from the train dataset



train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25, 

                                                                      random_state=42)
# Checking a sample image and its label



display(plt.imshow(train_images[5]))

print(train_labels[5])
# Setting up the Augmentation

datagen = ImageDataGenerator(

        rotation_range=30,

        horizontal_flip=True,

        fill_mode='nearest')



# fit parameters from train_images

datagen.fit(train_images, augment=True)
base_model = VGG16(include_top=False, input_shape = (227, 227, 3), weights = 'imagenet')



model = Sequential()

model.add(base_model)

model.add(GlobalAveragePooling2D())

model.add(Dropout(0.5))

model.add(Dense(2,activation='sigmoid'))

model.summary()
# Compiling the model

model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])



# Setting the learning rate as 0.0015

K.set_value(model.optimizer.learning_rate, 0.0015)
results = model.fit(datagen.flow(train_images, train_labels, batch_size=32), epochs = 100, 

                        validation_data =(val_images, val_labels), steps_per_epoch=len(train_images) / 32)
# Checking model's performance with training data

results_train = model.evaluate(train_images, train_labels)
# Checking model's performance with testing data

results_test = model.evaluate(test_images, test_labels)
def download_and_predict(url, filename):

    

    # download the image from the url and save

    os.system("curl -s {} -o {}".format(url, filename))

    

    # open the image

    img = Image.open(filename)

    

    # save the image

    img.save(filename)

    

    # convert it to RGB

    img = img.convert('RGB')

    

    # show image

    plt.imshow(img)

    plt.axis('off')

    

    # resize the image for VGG16 model

    img = img.resize((227, 227))

        

    # calculate probabilities of breeds

    img = imread(filename)

    img = preprocess_input(img)

    probs = model.predict(np.expand_dims(img, axis=0))

    pred = probs.argsort()[0][::-1][0]

    

    if pred == 1.:

        print("It's a Hot Dog!")

    else:

        print("It's not a hot dog :(")   
download_and_predict("https://i.ytimg.com/vi/tbiUujBVkq4/maxresdefault.jpg", "test_1.jpg")
download_and_predict("https://thumbor.thedailymeal.com/CSZ4VcpQmQ6lOvgkxnfG3LDg9GY=/870x565/https://www.thedailymeal.com/sites/default/files/recipe/2020/mainshutterstock_boiledhotd.jpg",

                     "test_2.jpg")
download_and_predict("https://www.vvsupremo.com/wp-content/uploads/2018/05/Pepperoni-Pizza-1.jpg", "test_3.jpg")
download_and_predict("https://www.simplyhappyfoodie.com/wp-content/uploads/2018/04/instant-pot-hamburgers-3.jpg", "test_4.jpg")
download_and_predict("https://inst-1.cdn.shockers.de/hs_cdn/out/pictures/master/product/1/hot-dog-hundekostuem--hot-dog-pet-costume--27513.jpg", "test_5.jpg")