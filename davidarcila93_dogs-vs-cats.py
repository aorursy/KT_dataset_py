import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import random

import shutil



TRAIN_DIR = '../input/dogs-vs-cats-redux-kernels-edition/train'

TEST_DIR = '../input/dogs-vs-cats-redux-kernels-edition/test'



def make_dir(path):

    if not os.path.exists(path):

        os.mkdir(path)



        

make_dir('train')

make_dir('train/cat')

make_dir('train/dog')

make_dir('validation')

make_dir('validation/cat')

make_dir('validation/dog')



train_validation_split = 0.9



for base, _, filenames in os.walk(TRAIN_DIR):

    cats = [name for name in filenames if 'cat' in name ]

    dogs = [name for name in filenames if 'dog' in name ]

    random.seed(42)

    random.shuffle(cats)

    random.shuffle(dogs)

    

    cats_number = len(cats)

    dogs_number = len(dogs)

    

    train_cats = cats[:int(cats_number*train_validation_split)]

    validation_cats = cats[int(cats_number*train_validation_split):]

    

    train_dogs = dogs[:int(dogs_number*train_validation_split)]

    validation_dogs = dogs[int(dogs_number*train_validation_split):]

    

    for name in train_cats:

        shutil.copy(base + '/' + name, './train' + '/cat/' + name)

        

    for name in validation_cats:

        shutil.copy(base + '/' + name, './validation' + '/cat/' + name)

        

    for name in train_dogs:

        shutil.copy(base + '/' + name, './train' + '/dog/' + name)

        

    for name in validation_dogs:

        shutil.copy(base + '/' + name, './validation' + '/dog/' + name)
print('Train cats:', len(os.listdir('./train/cat')))

print('Train dogs:', len(os.listdir('./train/dog')))

print('Validation cats:', len(os.listdir('./validation/cat')))

print('Validation dogs:', len(os.listdir('./validation/dog')))
import tensorflow as tf

from tensorflow import keras

print(tf.__version__)
model = keras.models.Sequential([

    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),

    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(32, (3,3), activation='relu'),

    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(64, (3,3), activation='relu'),

    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(64, (3,3), activation='relu'),

    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Flatten(),

    keras.layers.Dense(1024, activation='relu'),

    keras.layers.Dropout(0.1),

    keras.layers.Dense(1, activation='sigmoid')

])

model.summary()
model.compile(loss='binary_crossentropy',

              optimizers=keras.optimizers.RMSprop(lr=0.001),

              metrics=['acc']

             )
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=30,

    width_shift_range=0.1,

    height_shift_range=0.1,

    shear_range=0.1,

    zoom_range=0.1,

    horizontal_flip=True,

    fill_mode='nearest'

)

train_generator = train_datagen.flow_from_directory(

    './train',

    target_size=(150, 150),

    batch_size=50,

    class_mode='binary'

)

validation_datagen = ImageDataGenerator(

    rescale=1./255

)

validation_generator = validation_datagen.flow_from_directory(

    './validation',

    target_size=(150, 150),

    batch_size=50,

    class_mode='binary'

)
history = model.fit_generator(

    train_generator,

    epochs=10,

    validation_data=validation_generator

)
def plot_graph(history, metric):

    plt.plot(history.history[metric])

    plt.plot(history.history['val_' + metric])

    plt.xlabel('Epoch')

    plt.ylabel(metric)

    plt.legend([metric, 'val_' + metric])

    plt.show()



plot_graph(history, 'acc')

plot_graph(history, 'loss')
shutil.rmtree('./train')

shutil.rmtree('./validation')
from keras.preprocessing.image import array_to_img, img_to_array, load_img



def get_number(s):

    ar = s.split('.')

    first = ar[0]

    return int(first)



filenames = [ name for name in os.listdir(TEST_DIR + '/test')]

filenames.sort(key=get_number)



X_test = [ img_to_array(load_img(TEST_DIR + '/test/' + name, target_size=(150, 150))) for name in filenames]



test_generator = validation_datagen.flow(np.array(X_test), batch_size=50)



predictions = model.predict_generator(test_generator)
df = pd.DataFrame({'id':range(1, len(predictions)+1), 'label': predictions.reshape(len(predictions))})

df.to_csv('prediction.csv', index=None)

df.head()