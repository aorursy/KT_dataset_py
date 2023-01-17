# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print('test')

# Any results you write to the current directory are saved as output.
#images directory

imgs_dir = '../input/flowers-recognition/flowers/flowers'

test_imgs = '../input/testsunflower/testdata/testdata'
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout



num_classes = 5

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



my_new_model = Sequential()

my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))





my_new_model.add(Dense(1200, activation='relu'))

my_new_model.add(Dense(2400, activation='relu'))

my_new_model.add(Dense(1000, activation='relu'))

my_new_model.add(Dense(1200, activation='relu'))

my_new_model.add(Dense(200, activation='relu'))

my_new_model.add(Dense(100, activation='relu'))

my_new_model.add(Dense(1200, activation='relu'))

my_new_model.add(Dense(1200, activation='relu'))

my_new_model.add(Dense(1200, activation='relu'))

my_new_model.add(Dense(1200, activation='relu'))

my_new_model.add(Dense(1200, activation='relu'))

my_new_model.add(Dense(1200, activation='relu'))

my_new_model.add(Dense(1200, activation='relu'))

my_new_model.add(Dense(num_classes, activation='softmax'))



# Indicate whether the first layer should be trained/changed or not.

my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='sgd', 

                     loss='categorical_crossentropy', 

                     metrics=['accuracy'])
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator





data_generator = ImageDataGenerator(preprocess_input)







image_size = 244

data_generator = ImageDataGenerator(preprocess_input)



train_generator = data_generator.flow_from_directory(

                                        directory=imgs_dir,

                                        target_size=(image_size, image_size),

                                        batch_size=10,

                                        class_mode='categorical')

#testing an epochs value of 42

fit_stats = my_new_model.fit_generator(train_generator,

                                       steps_per_epoch=339,

                                       validation_steps=1,

                                       epochs=8

                                      )
image_size = 500

data_generator = ImageDataGenerator(preprocess_input)



train_generator = data_generator.flow_from_directory(

                                        directory=imgs_dir,

                                        target_size=(image_size, image_size),

                                        batch_size=10,

                                        class_mode='categorical')

#testing an epochs value of 42

fit_stats = my_new_model.fit_generator(train_generator,

                                       steps_per_epoch=339,

                                       validation_steps=1,

                                       epochs=8

                                      )
from IPython.display import Image, display



import os, random

img_locations = []

#for d in os.listdir("../input/flowers-recognition/flowers/flowers/"):

    #directory = "../input/flowers-recognition/flowers/flowers/" + d

    #sample = [directory + '/' + s for s in random.sample(

        #os.listdir(directory), int(random.random()*10))]

    #img_locations += sample

for d in os.listdir("../input/testsunflower2/testdata/"):

    directory = "../input/testsunflower2/testdata/" + d

    sample = [directory + '/' + s for s in random.sample(

        os.listdir(directory),12)]

    img_locations += sample



from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):

    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]

    img_array = np.array([img_to_array(img) for img in imgs])

    return preprocess_input(img_array)



random.shuffle(img_locations)

imgs = read_and_prep_images(img_locations)

predictions = my_new_model.predict_classes(imgs)

#preds = my_new_model.predict(imgs)

#most_likely_labels = decode_predictions(preds, top=3)

#for i, imgs in enumerate(imgs):

   # display(Image(img_path))

    #print(most_likely_labels[i])



classes = dict((v,k) for k,v in train_generator.class_indices.items())

score = 0

total = 0

for pred in predictions:

    total += 1

    if classes[pred] == 'sunflower':

        score+=1

print('score=',score/total)


for img, prediction in zip(img_locations, predictions):

    display(Image(img))

    print(classes[prediction])