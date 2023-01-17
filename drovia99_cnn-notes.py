#Escogemos con que imágenes trabajar



from os.path import join



image_dir = '../input/dog-breed-identification/train/'

img_paths = [join(image_dir, filename) for filename in 

                           ['0287b3374c33346e2b41f73af3a36261.jpg',

                            '042ecd9a978c2ee48d17f7f781621ac9.jpg',

                            '04fb4d719e9fe2b6ffe32d9ae7be8a22.jpg',

                            '0c19867277e6c96ad8f487b4fe343ff9.jpg']]



from IPython.display import Image, display

size=100

for i in img_paths:

    display(Image(i,width=size, height=size))
#Función para leer y preparar las imágenes

import numpy as np

from tensorflow.python.keras.applications.resnet import preprocess_input

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array



image_size = 224



def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):

    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]

    img_array = np.array([img_to_array(img) for img in imgs]) 

    output = preprocess_input(img_array) #[-1,1]

    return(output)
from tensorflow.python.keras.applications.resnet import ResNet50



my_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')

test_data = read_and_prep_images(img_paths)

preds = my_model.predict(test_data)
from learntools.deep_learning.decode_predictions import decode_predictions

from IPython.display import Image, display



most_likely_labels = decode_predictions(preds, top=3, class_list_path='../input/resnet50/imagenet_class_index.json')



size = 150

for i, img_path in enumerate(img_paths):

    display(Image(img_path,width=size, height=size))

    print(most_likely_labels[i])
import os

from os.path import join





hot_dog_image_dir = '../input/hot-dog-not-hot-dog/seefood/train/hot_dog'



hot_dog_paths = [join(hot_dog_image_dir,filename) for filename in 

                            ['1000288.jpg',

                             '127117.jpg']]



not_hot_dog_image_dir = '../input/hot-dog-not-hot-dog/seefood/train/not_hot_dog'

not_hot_dog_paths = [join(not_hot_dog_image_dir, filename) for filename in

                            ['823536.jpg',

                             '99890.jpg']]



img_paths = hot_dog_paths + not_hot_dog_paths



size = 150

for i in img_paths:

    display(Image(i,width=size, height=size))
from IPython.display import Image, display

from learntools.deep_learning.decode_predictions import decode_predictions

import numpy as np

from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.applications import ResNet50

from tensorflow.keras.preprocessing.image import load_img, img_to_array



image_size = 224



def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):

    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]

    img_array = np.array([img_to_array(img) for img in imgs])

    output = preprocess_input(img_array) #input to [-1,1]

    return(output)

    

my_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')

test_data = read_and_prep_images(img_paths)

preds = my_model.predict(test_data)



most_likely_labels = decode_predictions(preds, top=3)
size = 150

for i, img_path in enumerate(img_paths):

    display(Image(img_path, width=size, height=size))

    print(most_likely_labels[i])
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.deep_learning.exercise_3 import *

print("Setup Complete")
def is_hot_dog(preds):

    decoded = decode_predictions(preds, top=1)

    out = []

    for d in decoded:

        label = d[0][1]

        if label == 'hotdog':

            out.append(True)

        else: 

            out.append(False)

    return out

    

is_hot_dog(preds)



q_1.check()
def get_corrects(hot_dog, paths):

    test_data = read_and_prep_images(paths)

    preds = my_model.predict(test_data)

    out = is_hot_dog(preds)

    if (hot_dog):

        return sum(out)

    else:

        return out.count(False)



def calc_accuracy(model, paths_to_hotdog_images, paths_to_other_images):

    total_hot_dogs = len(paths_to_hotdog_images)

    total_others = len(paths_to_other_images)

    

    total_corrects = get_corrects(True, paths_to_hotdog_images) + get_corrects(False, paths_to_other_images)

    total_imgs = total_hot_dogs + total_others

    

    acc = total_corrects / total_imgs

    return acc



# Code to call calc_accuracy.  my_model, hot_dog_paths and not_hot_dog_paths were created in the setup code

my_model_accuracy = calc_accuracy(my_model, hot_dog_paths, not_hot_dog_paths)

print("Fraction correct in small test set: {}".format(my_model_accuracy))



# Check your answer

q_2.check()
# import the model

from tensorflow.keras.applications import VGG16



vgg16_model = VGG16(weights='../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5')

vgg16_accuracy = calc_accuracy(vgg16_model, hot_dog_paths, not_hot_dog_paths)





print("Fraction correct in small dataset: {}".format(vgg16_accuracy))

q_3.check()
from tensorflow.python.keras.applications.resnet import ResNet50

from tensorflow.python.keras.models import Sequential # El modelo será una secuencia de capas

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D



num_classes = 2 # Urbal, Rural

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



my_new_model = Sequential()

my_new_model.add(ResNet50(include_top=False, # Excluir la capa que realiza las predicciones

                          pooling='avg', 

                          weights=resnet_weights_path))

my_new_model.add(Dense(num_classes, activation='softmax')) # softmax => probabilidades



# Say not to train first layer (ResNet) model. It is already trained

my_new_model.layers[0].trainable = False
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# sgd = stochastic gradient descent
from tensorflow.python.keras.applications.resnet import preprocess_input

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator 



image_size = 224

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input) 

                # preprocess_input = usar esta función cada vez que se lea una imagen



train_generator = data_generator.flow_from_directory(

        '../input/urban-and-rural-photos/rural_and_urban_photos/train',

        target_size=(image_size, image_size),

        batch_size=24,

        class_mode='categorical')



validation_generator = data_generator.flow_from_directory(

        '../input/urban-and-rural-photos/rural_and_urban_photos/val',

        target_size=(image_size, image_size),

        class_mode='categorical')



my_new_model.fit_generator(

        train_generator,

        steps_per_epoch=3,

        validation_data=validation_generator,

        validation_steps=1)
from tensorflow.python.keras.applications.resnet import preprocess_input

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator



image_size = 224



data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,

                                   horizontal_flip=True, # Espejo horizontal

                                   width_shift_range = 0.2, # Cortar y rotar

                                   height_shift_range = 0.2,

                                            zoom_range= 0.2)



train_generator = data_generator_with_aug.flow_from_directory(

        '../input/urban-and-rural-photos/train',

        target_size=(image_size, image_size),

        batch_size=24,

        class_mode='categorical')



data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

validation_generator = data_generator_no_aug.flow_from_directory(

        '../input/urban-and-rural-photos/val',

        target_size=(image_size, image_size),

        class_mode='categorical')



my_new_model.fit_generator(

        train_generator,

        steps_per_epoch=3,

        epochs=3, # Va por cada imagen e veces

        validation_data=validation_generator,

        validation_steps=1)
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.deep_learning.exercise_4 import *

print("Setup Complete")
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D



num_classes = 2

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



my_new_model = Sequential()

my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))

my_new_model.add(Dense(num_classes, activation='softmax'))



# Indicate whether the first layer should be trained/changed or not.

my_new_model.layers[0].trainable = False



# Check your answer

step_1.check()
my_new_model.compile(optimizer='sgd', 

                     loss='categorical_crossentropy', 

                     metrics=['accuracy'])
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_size = 224

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)



train_generator = data_generator.flow_from_directory(

                                        directory='../input/dogs-gone-sideways/train',

                                        target_size=(image_size, image_size),

                                        batch_size=10,

                                        class_mode='categorical')



validation_generator = data_generator.flow_from_directory(

                                        directory='../input/dogs-gone-sideways/val',

                                        target_size=(image_size, image_size),

                                        class_mode='categorical')



'''

    batch_size = 10

    total = 220

    step = 22

'''

# fit_stats below saves some statistics describing how model fitting went

# the key role of the following line is how it changes my_new_model by fitting to data



fit_stats = my_new_model.fit_generator(train_generator,

                                       steps_per_epoch=22,

                                       validation_data=validation_generator,

                                       validation_steps=1)

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout



from keras.utils import np_utils



img_rows, img_cols = 28, 28

num_classes = 10



def data_prep(raw):

    out_y = np_utils.to_categorical(raw.label, num_classes) # target = [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]



    num_images = raw.shape[0] # 42000

    x_as_array = raw.values[:,1:] # np.array = csv - label

    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1) # 1 = grayscale

    out_x = x_shaped_array / 255

    return out_x, out_y



train_file = "../input/digit-recognizer/train.csv"

raw_data = pd.read_csv(train_file)



x, y = data_prep(raw_data)



model = Sequential()

model.add(Conv2D(20, kernel_size=(3, 3), # 20 = # filters

                 activation='relu',

                 input_shape=(img_rows, img_cols, 1))) # solo en la primera

model.add(Conv2D(20, kernel_size=(3, 3), activation='relu')) 

    # pueden haber más Conv2D

model.add(Flatten()) # output -> 1D representation para cada imagen

model.add(Dense(128, activation='relu')) # usual

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam', # determina el learning-rate solo

              metrics=['accuracy'])

model.fit(x, y,

          batch_size=128,

          epochs=2,

          validation_split = 0.2) # 20% para validation



# Se puede mejorar: # filters, # Conv2D

# t_epoch = 22s
model = Sequential()

model.add(Conv2D(30, kernel_size=(3, 3),

                 strides=2,

                 activation='relu',

                 input_shape=(img_rows, img_cols, 1)))

model.add(Dropout(0.5)) #*

model.add(Conv2D(30, kernel_size=(3, 3), strides=2, activation='relu')) # strides = 2

model.add(Dropout(0.5)) #*

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])

model.fit(x, y,

          batch_size=128,

          epochs=2,

          validation_split = 0.2)



# t_epoch = 7s
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow import keras



img_rows, img_cols = 28, 28

num_classes = 10



def prep_data(raw):

    y = raw[:, 0]

    out_y = keras.utils.to_categorical(y, num_classes)

    

    x = raw[:,1:]

    num_images = raw.shape[0]

    out_x = x.reshape(num_images, img_rows, img_cols, 1)

    out_x = out_x / 255

    return out_x, out_y



fashion_file = "../input/fashionmnist/fashion-mnist_train.csv"

fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')

x, y = prep_data(fashion_data)



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.deep_learning.exercise_7 import *

print("Setup Complete")
from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D



fashion_model = Sequential()

fashion_model.add(Conv2D(12, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(img_rows, img_cols, 1)))

fashion_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu')) 

fashion_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu')) 

fashion_model.add(Flatten())

fashion_model.add(Dense(100, activation='relu'))

fashion_model.add(Dense(num_classes, activation='softmax'))

fashion_model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam', # determina el learning-rate solo

              metrics=['accuracy'])

fashion_model.fit(x, y,

          batch_size=100,

          epochs=4,

          validation_split = 0.2)
second_fashion_model = Sequential()

second_fashion_model.add(Conv2D(12, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(img_rows, img_cols, 1)))

second_fashion_model.add(Conv2D(25, kernel_size=2, activation='relu')) # 20 - 3,3

second_fashion_model.add(Conv2D(25, kernel_size=2, activation='relu')) 

second_fashion_model.add(Flatten())

second_fashion_model.add(Dense(100, activation='relu'))

second_fashion_model.add(Dense(num_classes, activation='softmax')) # Prediction layer

second_fashion_model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam', # determina el learning-rate solo

              metrics=['accuracy'])

second_fashion_model.fit(x, y,

          batch_size=100,

          epochs=4, 

          validation_split = 0.2)
from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D



fashion_model = Sequential()

fashion_model.add(Conv2D(12, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(img_rows, img_cols, 1)))

model.add(Dropout(0.5)) #*

fashion_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', strides=2)) 

model.add(Dropout(0.5)) #*

fashion_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', strides=2)) 

model.add(Dropout(0.5)) #*

fashion_model.add(Flatten())

fashion_model.add(Dense(100, activation='relu'))

fashion_model.add(Dense(num_classes, activation='softmax'))

fashion_model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam', # determina el learning-rate solo

              metrics=['accuracy'])

fashion_model.fit(x, y,

          batch_size=100,

          epochs=4,

          validation_split = 0.2)