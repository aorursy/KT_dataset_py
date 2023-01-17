# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.preprocessing import image

from zipfile import ZipFile

import sys

from tqdm import tqdm

import os





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Deep learning libraries

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

import cv2



def Dataset_loader(DIR,RESIZE):

    IMG = []

    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))

    for IMAGE_NAME in tqdm(os.listdir(DIR)):

        PATH = os.path.join(DIR,IMAGE_NAME)

        _, ftype = os.path.splitext(PATH)

        if ftype == ".jpg":

            img = read(PATH)

            img = cv2.resize(img, (RESIZE,RESIZE))

            IMG.append(np.array(img)/255.)

    return IMG





cats_train = np.array(Dataset_loader('../input/cats_and_dogs_small/train/cats',224))

dogs_train = np.array(Dataset_loader('../input/cats_and_dogs_small/train/dogs',224))
# Cats vs. Dogs

# Create labels

cats_train_label = np.zeros(len(cats_train))

dogs_train_label = np.ones(len(dogs_train))





# Merge data 

x_train = np.concatenate((cats_train, dogs_train), axis = 0)

y_train = np.concatenate((cats_train_label, dogs_train_label), axis = 0)



# Shuffle train data

s = np.arange(x_train.shape[0])

np.random.shuffle(s)

x_train = x_train[s]

y_train = y_train[s]



# # Display first 15 images of animals, and how they are classified

w=60

h=40

fig=plt.figure(figsize=(15, 15))

columns = 4

rows = 3



for i in range(1, columns*rows +1):

    ax = fig.add_subplot(rows, columns, i)

    if y_train[i] == 0:

        ax.title.set_text('Cat')

    else:

        ax.title.set_text('Dog')

    plt.imshow(x_train[i], interpolation='nearest')

plt.show()
from tensorflow.python.keras import backend as K

K.clear_session()

del x_train

del y_train

del cats_train

del dogs_train
directory = "../input/cats_and_dogs_small"



train_dir = directory + '/train'

test_dir = directory + '/validation'



train_dir_cats = directory + '/train/cats'

train_dir_dogs = directory + '/train/dogs'

test_dir_cats = directory + '/validation/cats'

test_dir_dogs = directory + '/validation/dogs'
print('number of cats training images - ',len(os.listdir(train_dir_cats)))

print('number of dogs training images - ',len(os.listdir(train_dir_dogs)))

print('number of cats testing images - ',len(os.listdir(test_dir_cats)))

print('number of dogs testing images - ',len(os.listdir(test_dir_dogs)))
data_generator = ImageDataGenerator(rescale = 1.0/255.0,

                                    rotation_range=20,

                                    width_shift_range=0.2,

                                    height_shift_range=0.2,

                                    horizontal_flip=True,

                                    zoom_range=0.2)
batch_size = 64

training_data = data_generator.flow_from_directory(directory = train_dir,

                                                   shuffle=True,

                                                   target_size = (400, 400),

                                                   batch_size = batch_size,

                                                   class_mode = 'binary')

testing_data = data_generator.flow_from_directory(directory = test_dir,

                                                  shuffle=True,

                                                  target_size = (400, 400),

                                                  batch_size = batch_size,

                                                  class_mode = 'binary')
import sys

from matplotlib import pyplot

from keras.utils import to_categorical

from keras.applications.vgg16 import VGG16

from keras.models import Model

from keras.layers import Dense

from keras.layers import Flatten

from keras.optimizers import SGD

from keras.preprocessing.image import ImageDataGenerator#Se preparan las capas para la CNN

model = VGG16(include_top=False, input_shape=(400, 400, 3))

# mark loaded layers as not trainable

#Se marcan las capas cargadas como no entrenables

for layer in model.layers:

    layer.trainable = False

# Se añaden las nuevas capas del clasificador

flat1 = Flatten()(model.layers[-1].output)

class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)

output = Dense(1, activation='sigmoid')(class1)

# define new model

model = Model(inputs=model.inputs, outputs=output)

# compile model

opt = SGD(lr=0.001, momentum=0.9)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
fitted_model = model.fit_generator(training_data,

                        steps_per_epoch = len(training_data),

                        epochs = 5,

                        validation_data = testing_data,

                        validation_steps = len(testing_data),

                        use_multiprocessing = False)
# list all data in history

print(fitted_model.history .keys())

# summarize history for accuracy

plt.plot(fitted_model.history ['accuracy'])

plt.plot(fitted_model.history ['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(fitted_model.history ['loss'])

plt.plot(fitted_model.history ['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
def testing_image(image_directory):

    test_image = image.load_img(image_directory, target_size = (400, 400))

    test_image = image.img_to_array(test_image)

    test_image = np.expand_dims(test_image, axis = 0)

    result = model.predict(x = test_image)

    if result[0][0]  >= 1:

        #Dog

        prediction = 1

    else:

        #Cat

        prediction = 0

    return prediction
#print(testing_image(directory + '/test/'))



test = directory + '/test/'

lista = []

for images in os.listdir(test):

    lista.append(testing_image(test + images))

#Nos quedamos con los números para hacer el submission

import re

listaID = []

for item in os.listdir(test):

    listaID.append(re.sub("[^0-9]", "", str(item)))

    
def submission_generation(dataframe, name):

    """

    Esta función genera un csv a partir de un dataframe de pandas. 

    Con FileLink se genera un enlace desde el que poder descargar el fichero csv

    

    dataframe: DataFrame de pandas

    name: nombre del fichero csv

    """

    import os

    from IPython.display import FileLink

    os.chdir(r'/kaggle/working')

    dataframe.to_csv(name, index = False)

    return  FileLink(name)
print(len(lista))

print(len(listaID))



Predicting = pd.DataFrame({"id": listaID, 

                                      "label": lista})



submission_generation(Predicting , "TryPredict.csv")