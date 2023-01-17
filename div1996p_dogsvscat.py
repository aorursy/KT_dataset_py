# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os,cv2,re,random

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array , load_img

from keras import layers,models,optimizers

from keras import backend as K

from sklearn.model_selection import train_test_split

    

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


train_dir = "../input/train/"

test_dir = "../input/test/"

train_img_dogs_cat = [train_dir + i for i in os.listdir(train_dir)]#use this f or use whole dataset

test_img_dogs_cat = [test_dir + i for i in os.listdir(test_dir)]

def atoi(text):

    return int(text) if text.isdigit() else text

def natural_keys(text):

    return [atoi(c) for c in re.split('(\d+)', text) ]
train_img_dogs_cat.sort(key = natural_keys)

train_img_dogs_cat = train_img_dogs_cat[0:1300] + train_img_dogs_cat[12500 :13800]

test_img_dogs_cat.sort(key = natural_keys)

img_height = 150

img_width = 150





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

    

    for i in list_of_images:

        if 'dog' in i:

            y.append(1)

        elif 'cat' in i:

            y.append(0)

        #else:

            #print('neither cat nor dog name present in images')

            

    return x, y

    

    
X,Y = prepare_data(train_img_dogs_cat)

print(K.image_data_format())
X_train, X_val ,y_train, y_val = train_test_split(X,Y,test_size = 0.2 ,random_state = 1)
nb_train_samples = len(X_train)

nb_validation_samples = len(X_val)

batch_size = 16
model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),input_shape=(img_width,img_height,3)))

model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size = (2,2)))



model.add(layers.Conv2D(32,(3,3)))

model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size = (2,2)))



model.add(layers.Conv2D(64,(3,3)))

model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size = (2,2)))



model.add(layers.Flatten())

model.add(layers.Dense(64))

model.add(layers.Activation('relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1))

model.add(layers.Activation('sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])



model.summary()
train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



val_datagen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)
train_generator = train_datagen.flow(np.array(X_train),y_train,batch_size = batch_size)

validation_generator = train_datagen.flow(np.array(X_val),y_val,batch_size = batch_size)
hystory = model.fit_generator(

                train_generator,

                steps_per_epoch = nb_train_samples // batch_size,

                epochs = 30,

                validation_data = validation_generator,

                validation_steps=nb_validation_samples // batch_size

                 )
X_test, Y_test = prepare_data(test_img_dogs_cat) #Y_test in this case will be []
test_datagen = ImageDataGenerator(rescale=1. / 255)
steps = nb_train_samples // batch_size

test_generator = val_datagen.flow(np.array(X_test), batch_size=batch_size)



prediction_probabilities = model.predict_generator(test_generator, verbose=1,steps=782)
counter = range(1, len(test_img_dogs_cat) + 1)

solution = pd.DataFrame({"id": counter, "label":list(prediction_probabilities)})

cols = ['label']



for col in cols:

    solution[col] = solution[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)



solution.to_csv("dogsVScats.csv", index = False)