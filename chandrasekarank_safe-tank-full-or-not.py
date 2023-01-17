# For displaying the image

from IPython.display import Image, display 



# Preprocessing the image for fit_generator

import numpy as np

from tensorflow.python.keras.applications.resnet import preprocess_input 

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
# Transfer learing 

from tensorflow.python.keras.applications.resnet import ResNet50

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Take this comment out



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



test_img_path = "../input/safetankfullornot/safe/train/Safe tank full/DEBUG_IMG_20200616_124457.jpg"

display(Image(test_img_path))
image_size = 224

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, 

                                    width_shift_range = 0.2,

                                   height_shift_range = 0.2)





train_generator = data_generator.flow_from_directory(

        '../input/safetankfullornot/safe/train/',

        target_size=(image_size, image_size),

        batch_size=1,

        class_mode='categorical')



validation_generator = data_generator.flow_from_directory(

        '../input/safetankfullornot/safe/valid/',

        target_size=(image_size, image_size),

        class_mode='categorical')
num_classes = 2

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



my_new_model = Sequential()

my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))

my_new_model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

my_new_model.layers[0].trainable = False
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
my_new_model.fit_generator(

        train_generator,

        steps_per_epoch=3,

        validation_data=validation_generator,

        validation_steps=1)
from os.path import join



image_dir = "../input/safetankfullornot/safe/valid/Safe tank not full/"

img_paths = [join(image_dir, filename) for filename in 

                           ["DEBUG_IMG_20200616_124447.jpg",  

                           "DEBUG_IMG_20200616_125309.jpg"]]

import numpy as np

from tensorflow.python.keras.applications.resnet import preprocess_input

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array



image_size = 224



def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):

    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]

    img_array = np.array([img_to_array(img) for img in imgs])

    output = preprocess_input(img_array)

    return(output)



test_data = read_and_prep_images(img_paths)

preds = my_new_model.predict(test_data)

print(preds)
Image(filename = test_img_path, width = 224, height = 224)
from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
import os

def img_path(path):

    for dirname, _, filenames in os.walk(path):

        l = []

        y = []

        for filename in filenames:

            l.append(os.path.join(dirname, filename))

            y.append(os.path.join(dirname, filename).split("/")[-2])

        return (l, y)
from tensorflow.python import keras

image_size = 224

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):

    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]

    img_array = np.array([img_to_array(img) for img in imgs])

    output = preprocess_input(img_array)

    return(output)
l_img_path, out_y = img_path("../input/safetankfullornot/safe/train/Safe tank full/")

x, y = img_path("../input/safetankfullornot/safe/train/Safe tank not full/")

l_img_path = l_img_path + x 

out_y = out_y + y 

x, y = img_path("../input/safetankfullornot/safe/valid/Safe tank full/")

l_img_path = l_img_path + x 

out_y = out_y + y 

x, y = img_path("../input/safetankfullornot/safe/valid/Safe tank not full/")

l_img_path = l_img_path + x 

out_y = out_y + y 

#print(l_img_path)

#print(out_y)

out_x = read_and_prep_images(l_img_path)



for i in range(len(out_y)):

    if out_y[i] == "Safe tank full":

        out_y[i] = 1

    else:

        out_y[i] = 0



print(out_y)

from tensorflow import keras

out_y = keras.utils.to_categorical(out_y, 2)

num_classes = 2

image_size = 224

img_rows = image_size

img_cols = image_size



scratch_model = Sequential()

scratch_model.add(Conv2D(10, kernel_size=(1, 1),

                 activation='relu',

                 input_shape=(img_rows, img_cols, 3)))

scratch_model.add(Conv2D(10, kernel_size=(1, 1), activation='relu'))

scratch_model.add(Flatten())

scratch_model.add(Dense(100, activation='relu'))

scratch_model.add(Dense(num_classes, activation='softmax'))

scratch_model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])

scratch_model.fit(out_x, out_y,

          batch_size=1,

          epochs=10)
img_paths = ['../input/safetankfullornot/safe/valid/Safe tank full/DEBUG_IMG_20200616_124450.jpg', '../input/safetankfullornot/safe/valid/Safe tank not full/DEBUG_IMG_20200616_125231.jpg', '../input/safetankfullornot/safe/train/Safe tank full/IMG_20200616_124440_1.jpg' ] 

test_img = read_and_prep_images(img_paths)

preds = scratch_model.predict(test_img)

print(preds)
img_paths = ['../input/safetankfullornot/DEBUG_IMG_20200622_092725.jpg', '../input/safetankfullornot/DEBUG_IMG_20200622_092728.jpg'] 

test_img = read_and_prep_images(img_paths)

preds = scratch_model.predict(test_img)

print(preds)