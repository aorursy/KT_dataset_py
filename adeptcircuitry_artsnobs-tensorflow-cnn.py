from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import keras

board = keras.callbacks.TensorBoard(log_dir='./logs/artsnobs', histogram_freq=0, batch_size=100, write_graph=True)

datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode = 'nearest')
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense



model = Sequential()

# model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))

model.add(Conv2D(32, (3, 3), input_shape=(175, 175, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(5))

model.add(Activation('softmax'))
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

# model.summary()
batch_size = 100 #change this from 16

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory('/Users/ethan/Documents/MLData/ArtSnobs/dataset/dataset_updated/training_set/', target_size = (175, 175), batch_size = batch_size, class_mode = 'categorical', color_mode='rgb')

v_generator = test_datagen.flow_from_directory('/Users/ethan/Documents/MLData/ArtSnobs/dataset/dataset_updated/validation_set/', target_size = (175,175), batch_size = batch_size, class_mode = 'categorical', color_mode='rgb')

# train_generator = train_datagen.flow_from_directory('/Users/ethan/Documents/MLData/ArtSnobs/dataset/dataset_updated/training_set/', target_size = (150, 150), batch_size = batch_size, class_mode = 'categorical', color_mode='rgb')

# v_generator = test_datagen.flow_from_directory('/Users/ethan/Documents/MLData/ArtSnobs/dataset/dataset_updated/validation_set/', target_size = (150,150), batch_size = batch_size, class_mode = 'categorical', color_mode='rgb')
model.fit_generator(train_generator, steps_per_epoch = 2500 // batch_size, epochs = 100, validation_data =v_generator, validation_steps = 1250 // batch_size, callbacks = [board])

model.save_weights('first_try.h5')