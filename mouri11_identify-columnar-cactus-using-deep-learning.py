import os

from os.path import join

from tensorflow.python import keras

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import numpy as np

import pandas as pd



train_img_dir = '../input/train/train'



test_img_dir = '../input/test/test'

test_img_paths = [join(test_img_dir, img) for img in os.listdir(test_img_dir)]



train_data = pd.read_csv('../input/train.csv')

train_img_paths = [join(train_img_dir, img) for img in train_data['id']]



img_size = 32



def prep_imgs(img_paths, img_height = img_size, img_width = img_size): # loading images and converting to numpy array

    imgs = [load_img(img, target_size = (img_height, img_width)) for img in img_paths]

    img_arr = np.array([img_to_array(img) for img in imgs]) / 255

    return img_arr



X = prep_imgs(train_img_paths)

y = train_data['has_cactus'].values # not using one-hot as sparse doesn't support it



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 1)



# adding layers to CNN

model = Sequential()

model.add(Conv2D(25, kernel_size = 2, strides = 2, activation = 'relu', input_shape = (img_size, img_size, 3)))

model.add(MaxPooling2D(pool_size = (2,2), strides = 2)) #applying max pooling to convolution layer

# model.add(Dropout(0.5))

model.add(Conv2D(25, kernel_size = 2, strides = 2, activation = 'relu'))

# model.add(Dropout(0.5))

model.add(MaxPooling2D(pool_size = (2,2), strides = 2))



model.add(Flatten())



model.add(Dense(250, activation = 'relu'))

model.add(Dense(2, activation = 'softmax'))



model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])



batch_size = 32



train_datagen = ImageDataGenerator(featurewise_center = True,

                                   rotation_range=20,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   brightness_range = [0.2, 0.5],

                                   horizontal_flip=True)



# train_generator = train_datagen.flow_from_directory(

#     '../input/train/',

#     target_size = (img_size, img_size), 

#     batch_size = batch_size, 

#     subset = 'training')



# validation_generator = train_datagen.flow_from_directory(

#     '../input/train/',

#     target_size = (img_size, img_size), 

#     batch_size = batch_size, 

#     subset = 'validation')



train_datagen.fit(X_train)



model.summary()
train_generator = train_datagen.flow(X_train, y_train, batch_size = batch_size)



history = model.fit_generator(

    train_generator, 

    steps_per_epoch = len(X_train) // batch_size, 

    validation_data = (X_val, y_val), 

    validation_steps = len(X_val) // batch_size,

    epochs = 10)

# model.fit(X_train, y_train, batch_size = 100, epochs = 4, validation_split = 0.2)
# X_test = prep_imgs(test_img_paths)

test_datagen = ImageDataGenerator(rescale = 1. / 255)



test_generator = test_datagen.flow_from_directory(

    '../input/test/',

    target_size = (img_size, img_size), 

    color_mode = 'rgb', 

    shuffle = False, 

    batch_size = 1)



nb_samples = len(test_generator.filenames)



# X_train = prep_imgs(test_img_dir)

preds_temp = model.predict_generator(test_generator, steps = nb_samples)

preds = preds_temp.argmax(axis = -1)



output = pd.DataFrame({'id': os.listdir(test_img_dir), 'has_cactus': preds})

output.to_csv('submission.csv', index = False)