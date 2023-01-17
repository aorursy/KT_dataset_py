import os, cv2

import numpy as np

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.optimizers import SGD

from keras.preprocessing.image import img_to_array, array_to_img, load_img

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle



np.random.seed(7)



# fix dimension ordering issue

# https://stackoverflow.com/questions/39547279/loading-weights-in-th-format-when-keras-is-set-to-tf-format

from keras import backend as K

# K.set_image_dim_ordering('th')



# https://www.tensorflow.org/api_docs/python/tf/keras/backend/set_image_data_format

K.set_image_data_format('channels_last')

# print(K.image_data_format())  
import zipfile

with zipfile.ZipFile('../input/dogs-vs-cats-redux-kernels-edition/train.zip',"r") as z:

    z.extractall('input/dogs-vs-cats-redux-kernels-edition/')

with zipfile.ZipFile('../input/dogs-vs-cats-redux-kernels-edition/test.zip',"r") as z:

    z.extractall('input/dogs-vs-cats-redux-kernels-edition/')    
img_width = 150

img_height = 150

TRAIN_DIR = 'input/dogs-vs-cats-redux-kernels-edition/train/'

TEST_DIR = '.input/dogs-vs-cats-redux-kernels-edition/test/'

train_images_dogs_cats = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
# Sort the traning set. Use 5000 images of cats and dogs instead of all 25000 to speed up the learning process.

# If we sort them, the top part will be cats, bottom part will be dogs.

train_images_dogs_cats.sort()

train_images_dogs_cats = train_images_dogs_cats[:5000] + train_images_dogs_cats[-5000:] 
# Now the images have to be represented in numbers. For this, using the openCV library read and resize the image.

# Generate labels for the supervised learning set.

# Below is the helper function to do so.



def prepare_data(list_of_images):

    """

    Returns two arrays: 

        x is an array of resized images

        y is an array of labels

    """

    x = [] # images as arrays

    y = [] # labels

    

    for image in list_of_images:

        x.append(cv2.resize(cv2.imread(image), (img_width,img_height), interpolation=cv2.INTER_CUBIC))

    

    for image in list_of_images:

        image = image.replace("dogs-vs-cats-redux-kernels-edition/", "")

        if 'dog' in image:

            y.append(1)

        elif 'cat' in image:

            y.append(0)

            

    return shuffle(np.array(x), np.array(y))
X, Y = prepare_data(train_images_dogs_cats)
print (X.shape)

print (Y.shape)

print (Y.sum())
batch_size = 16



some_entry = array_to_img(X[-1])

some_entry
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(img_width, img_height, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Dropout(0.5))



model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Dropout(0.5))



model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())



model.add(Dense(256, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



print (model.summary())
model.fit(X, Y, batch_size=batch_size, epochs=5, validation_split=0.3)
# https://faroit.github.io/keras-docs/1.2.2/backend/

# For 2D data (e.g. image), "tf" assumes (rows, cols, channels) while "th" assumes (channels, rows, cols).



# from keras.applications.inception_v3 import InceptionV3

# weights = '../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# model = InceptionV3(include_top=False, weights=weights)



from keras.applications.vgg16 import VGG16

weights = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

model = VGG16(include_top=False, weights=weights)
from os import makedirs

from os.path import join, exists, expanduser



cache_dir = expanduser(join('~', '.keras'))

if not exists(cache_dir):

    makedirs(cache_dir)

models_dir = join(cache_dir, 'models')

if not exists(models_dir):

    makedirs(models_dir)

!cp  ../input/inceptionv3/* ~/.keras/models/

!cp  ../input/vgg16/* ~/.keras/models/
X.shape
bottleneck_features = model.predict(X)
model = Sequential()

model.add(Flatten(input_shape=bottleneck_features.shape[1:]))

model.add(Dense(256, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))



model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])



print (model.summary())



model.fit(bottleneck_features, 

          Y,

          epochs=10,

          batch_size=batch_size,

          validation_split=0.3)
# Fine Tuning

base_model = VGG16(include_top=False, weights=weights, input_shape=(150,150,3))



full_model = Model(inputs=base_model.input, outputs=model(base_model.output))



# set the first 25 layers (up to the last conv block)

# to non-trainable (weights will not be updated)

# for layer in full_model.layers[:25]:

#     layer.trainable = False

    

# compile the model with a SGD/momentum optimizer

# and a very slow learning rate.

full_model.compile(loss='binary_crossentropy',

                   optimizer=SGD(lr=1e-4, momentum=0.9),

                   metrics=['accuracy'])



print (full_model.summary())
full_model.fit(X, 

               Y,

               epochs=15,

               batch_size=batch_size,

               validation_split=0.3)