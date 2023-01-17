%matplotlib inline

import os

import cv2

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam

from random import randrange

from shutil import copytree, move, rmtree
INPUT_DATA_DIR = '../input/cell_images/cell_images'

ROOT_DATA_DIR = '../cell_images'

copytree(INPUT_DATA_DIR, ROOT_DATA_DIR)
INF_DIR = os.path.join(ROOT_DATA_DIR, 'Parasitized')

UNINF_DIR = os.path.join(ROOT_DATA_DIR, 'Uninfected')



inf_fnames = os.listdir(INF_DIR)

uninf_fnames = os.listdir(UNINF_DIR)



print(f'Amount of parasitized images: {len(inf_fnames)}')

print(f'Amount of uninfected images: {len(uninf_fnames)}')

print(f'Total Images: {len(inf_fnames) + len(uninf_fnames)}')
nrows, ncols = 4, 4

fig_size = 3 



fig = plt.gcf()

fig.set_size_inches(ncols * fig_size, nrows * fig_size)



inf_pic_paths = [os.path.join(INF_DIR, inf_fnames[randrange(len(inf_fnames))]) 

                for _ in range(4) 

                ]



uninf_pic_paths = [os.path.join(UNINF_DIR, uninf_fnames[randrange(len(uninf_fnames))]) 

                for _ in range(4) 

                ]



for i, img_path in enumerate(inf_pic_paths + uninf_pic_paths):  

    sp = plt.subplot(nrows, ncols, i + 1)

    sp.axis('Off') # Don't show axes (or gridlines)



    img = mpimg.imread(img_path)

    plt.imshow(img)



plt.show()
def split_data(sourse, split_size):

    #Create root folder for valid&test data

    root_folder_name = ROOT_DATA_DIR.strip('/').split('/')[-1]

    valid_test_folder = ROOT_DATA_DIR.replace(root_folder_name,

                                            'valid_test_' + root_folder_name)

    try:

        os.mkdir(valid_test_folder)

    except:

        pass

    

    folders = os.listdir(sourse)

    for folder in folders:

        try:

            os.mkdir(os.path.join(valid_test_folder, folder))

        except:

            pass

        fnames = os.listdir(os.path.join(sourse, folder))

        start_split = len(fnames) - int(len(fnames) * split_size)

        splited_fnames = fnames[start_split:]

        for fname in splited_fnames:

            s_dir = os.path.join(sourse, folder, fname)

            d_dir = os.path.join(valid_test_folder, folder, fname)

            move(s_dir, d_dir)

        print(f'Moved {len(splited_fnames)} files')

split_data(ROOT_DATA_DIR, 0.2)
img_size = 64

dim = img_size, img_size



img_path = os.path.join(INF_DIR, inf_fnames[0])

img = cv2.imread(img_path)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = cv2.resize(img, dim)



plt.imshow(img)

plt.show()
train_datagen = ImageDataGenerator(

        rescale=1./255,

        rotation_range=45,

        shear_range=0.2,

        vertical_flip=True,

        horizontal_flip=True,

        )



train_generator = train_datagen.flow_from_directory(

        '../cell_images',  

        target_size=dim, 

        batch_size=32,

        class_mode='binary',

        )
valid_test_datagen = ImageDataGenerator(

        rescale=1./255,

        validation_split=0.5

        )



valid_generator = valid_test_datagen.flow_from_directory(

        '../valid_test_cell_images',  

        target_size=dim, 

        batch_size=32,

        class_mode='binary',

        subset='training',

        )



test_generator = valid_test_datagen.flow_from_directory(

        '../valid_test_cell_images',  

        target_size=dim, 

        batch_size=32,

        class_mode='binary',

        subset='validation',

        )
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', input_shape=(64, 64, 3)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2, 2),

    

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2, 2),

    

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2, 2),

    

    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2, 2), 

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.5),

    

    tf.keras.layers.Dense(1024, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    

    tf.keras.layers.Dense(1, activation='sigmoid')

])





model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['acc'])
model.summary()
history = model.fit_generator(

        train_generator,

        epochs=20,

        validation_data=valid_generator,

        )
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
model.evaluate_generator(test_generator, verbose=1)