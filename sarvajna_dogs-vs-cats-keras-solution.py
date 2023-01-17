import os, cv2, re, random

import numpy as np

import pandas as pd

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array, load_img

from keras import layers, models, optimizers

from keras import backend as K

from sklearn.model_selection import train_test_split
img_width = 150

img_height = 150

TRAIN_DIR = '../input/train/'

TEST_DIR = '../input/test/'

train_images_dogs_cats = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset

test_images_dogs_cats = [TEST_DIR+i for i in os.listdir(TEST_DIR)]
def atoi(text):

    return int(text) if text.isdigit() else text



def natural_keys(text):

    return [ atoi(c) for c in re.split('(\d+)', text) ]
train_images_dogs_cats.sort(key=natural_keys)

train_images_dogs_cats = train_images_dogs_cats[0:1300] + train_images_dogs_cats[12500:13800] 



test_images_dogs_cats.sort(key=natural_keys)
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
X, Y = prepare_data(train_images_dogs_cats)

print(K.image_data_format())
# First split the data in two sets, 80% for training, 20% for Val/Test)

X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=0.2, random_state=1)
nb_train_samples = len(X_train)

nb_validation_samples = len(X_val)

batch_size = 16
model = models.Sequential()



model.add(layers.Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))

model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))



model.add(layers.Conv2D(32, (3, 3)))

model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))



model.add(layers.Conv2D(64, (3, 3)))

model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))



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
train_generator = train_datagen.flow(np.array(X_train), Y_train, batch_size=batch_size)

validation_generator = val_datagen.flow(np.array(X_val), Y_val, batch_size=batch_size)
history = model.fit_generator(

    train_generator, 

    steps_per_epoch=nb_train_samples // batch_size,

    epochs=30,

    validation_data=validation_generator,

    validation_steps=nb_validation_samples // batch_size

)

model.save_weights('model_wieghts.h5')

model.save('model_keras.h5')
X_test, Y_test = prepare_data(test_images_dogs_cats) #Y_test in this case will be []
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = val_datagen.flow(np.array(X_test), batch_size=batch_size)

prediction_probabilities = model.predict_generator(test_generator, verbose=1)
counter = range(1, len(test_images_dogs_cats) + 1)

solution = pd.DataFrame({"id": counter, "label":list(prediction_probabilities)})

cols = ['label']



for col in cols:

    solution[col] = solution[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)



solution.to_csv("dogsVScats.csv", index = False)