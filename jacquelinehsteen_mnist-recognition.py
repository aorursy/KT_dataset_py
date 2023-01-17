import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D



# set image resolution to 28x28 pixels

img_rows, img_cols = 28, 28



# number of classes = number of possible digits

num_classes = 10



def prep_data(raw):

    #target variable is in first column (0-9)

    y = raw[:,0]

    # OHE from numerical to categorical values

    out_y = keras.utils.to_categorical(y, num_classes)

    # extract all feature variables (as matrix X)

    X = raw[:, 1:]

    num_images = raw.shape[0]

    #all images stacked on top of each other with 2 dim (rows and cols). 1 stands for 1 channel/filer (greyscale). Color images has 3 instead.

    out_x = X.reshape(num_images, img_rows, img_cols, 1)

    out_x = out_x / 255

    return out_x, out_y



mnist_file = '../input/digit-recognizer/train.csv'

mnist_data = np.loadtxt(mnist_file, skiprows=1, delimiter=',')



x, y = prep_data(mnist_data)
# create model

model = Sequential()



# First layer

model.add(Conv2D(12,

                activation='relu',

                kernel_size=3,

                input_shape=(img_rows, img_cols, 1)))



# 2nd & 3rd layer

model.add(Conv2D(20,

                activation='relu',

                kernel_size=3))

model.add(Conv2D(20,

                activation='relu',

                kernel_size=3))



# Flatten out into 1D vector

model.add(Flatten())



# Dense layers, turn values into probabilities with softmax function

model.add(Dense(128, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))



# compile model, adam stands for AdaGrad and RMSProp, generally faster than working with a fixed learning rate (like SGD)

model.compile(loss='categorical_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])
# fit model

model.fit(x, y,

         batch_size=100,

         epochs=4,

         validation_split=0.2)
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.layers import GlobalAveragePooling2D





num_classes = 2

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



# create model

res_model = Sequential()



# first layer is the transfered model ResNet50

# include top=False, because top layer (last layer) is the prediction layer

res_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))



# predicition layer

res_model.add(Dense(num_classes, activation='softmax'))



# we exclude the first layer (i.e. Resnet50 ) because it has already been trained

res_model.layers[0].trainable=False



# compile model

res_model.compile(optimizer='sgd',

             loss='categorical_crossentropy',

             metrics=['accuracy'])



# compile model
import os

from os.path import join



print(os.listdir("../input"))



pokemon_image_dir = '../input/pokemon-images-and-types/images/images'



pokemon_paths = [join(pokemon_image_dir,filename) for filename in 

                            ['araquanid.jpg',

                             'ambipom.png',

                             'arbok.png',

                             'alomomola.png']]



cat_image_dir = '../input/animals10/raw-img/gatto'

cat_paths = [join(cat_image_dir, filename) for filename in

                            ['1007.jpeg',

                             '10.jpeg',

                             '1017.jpeg',

                             '1001.jpeg']]



img_paths = pokemon_paths + cat_paths



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

    output = preprocess_input(img_array)

    return(output)





my_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')

test_data = read_and_prep_images(img_paths)

preds = my_model.predict(test_data)



most_likely_labels = decode_predictions(preds, top=3)



for i, img_path in enumerate(img_paths):

    display(Image(img_path))

    print(most_likely_labels[i])
import shutil



shutil.rmtree('/kaggle/working/poke')

os.mkdir('/kaggle/working/poke/')
catpath = '../input/animals10/raw-img/gatto/'

pokepath = '../input/pokemon-images-and-types/images/images/'

new_path = '/kaggle/working/poke'



pokefileList = os.listdir(pokepath)



animaldict={}

for f in pokefileList:

    shutil.copy(pokepath + str(f), new_path)

    animaldict[f] = 'poke'



catfileList = os.listdir(catpath)



for f in catfileList[0:len(pokefileList)]:

    shutil.copy(catpath + str(f), new_path)

    animaldict[f] = 'cat'



# print(list(animaldict.items()))

train = pd.DataFrame(list(animaldict.items()))

train.columns = ['title', 'target']

print(train.shape)
import tensorflow as tf

import pandas as pd

print(tf.__version__)
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_dataGen = ImageDataGenerator(rescale = 1.0/255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True,

                                   fill_mode='nearest')



validation_datagen = ImageDataGenerator(rescale = 1.0/255)



train_generator = train_dataGen.flow_from_dataframe(dataframe = training_set,

                                                    directory = '..')





# set image resolution to 28x28 pixels

img_rows, img_cols = 28, 28



# number of classes = number of possible digits

num_classes = 10



def prep_data(raw):

    #target variable is in first column (0-9)

    y = raw[:,0]

    # OHE from numerical to categorical values

    out_y = keras.utils.to_categorical(y, num_classes)

    # extract all feature variables (as matrix X)

    X = raw[:, 1:]

    num_images = raw.shape[0]

    #all images stacked on top of each other with 2 dim (rows and cols). 1 stands for 1 channel/filer (greyscale). Color images has 3 instead.

    out_x = X.reshape(num_images, img_rows, img_cols, 1)

    out_x = out_x / 255

    return out_x, out_y



mnist_file = '../input/digit-recognizer/train.csv'

mnist_data = np.loadtxt(mnist_file, skiprows=1, delimiter=',')



x, y = prep_data(mnist_data)