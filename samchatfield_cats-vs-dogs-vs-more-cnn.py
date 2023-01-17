import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras as ks # neural network models



# For working with images

import cv2

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import tqdm



# Potentially useful tools - you do not have to use these

from keras.models import Sequential, Model, Input

from keras.layers import Activation, Flatten, Dense, Dropout, ZeroPadding2D, Conv2D, MaxPool2D, BatchNormalization, GlobalAveragePooling2D, Average

from keras.optimizers import SGD, RMSprop, Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical



import os

import math



# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.
# CONSTANTS

# You may not need all of these, and you may find it useful to set some extras



CATEGORIES = ['airplane','car','cat','dog','flower','fruit','motorbike','person']



IMG_WIDTH = 100

IMG_HEIGHT = 100

TRAIN_PATH = '../input/natural_images/natural_images/'

TEST_PATH = '../input/evaluate/evaluate/'
# To find data:

folders = os.listdir(TRAIN_PATH)



images = []



for folder in folders:

    files = os.listdir(TRAIN_PATH + folder)

    images += [(folder, file, folder + '/' + file) for file in files]



image_locs = pd.DataFrame(images, columns=('class','filename','file_loc'))



# data structure is three-column table

# first column is class, second column is filename, third column is image address relative to TRAIN_PATH

display(image_locs.head(10))

display(image_locs.shape)
# Shuffle the image_locs dataframe

image_locs_shuffled = image_locs.sample(frac=1)

display(image_locs_shuffled.head())
row_count = len(image_locs_shuffled.index)

val_split = 0.1

train_split = 1 - val_split



# Split the shuffled image_locs into training and validation dataframes by the proportion given by val_split

train_image_locs = image_locs_shuffled[:math.floor(train_split * row_count)]

val_image_locs = image_locs_shuffled[-math.ceil(val_split * row_count):]

display(train_image_locs.shape)

display(val_image_locs.shape)
# Training data generator with scaling and data augmentation

train_datagen = ImageDataGenerator(

    rescale=1./255,

#     rotation_range=10,

#     width_shift_range=0.1,

#     height_shift_range=0.1,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True

)



# Validation/Test data generator with only scaling

test_datagen = ImageDataGenerator(

    rescale=1./255

)
# train_generator = train_datagen.flow_from_directory(

#     directory=TRAIN_PATH,

#     target_size=(IMG_WIDTH, IMG_HEIGHT)

# )



train_generator = train_datagen.flow_from_dataframe(

    train_image_locs,

    directory=TRAIN_PATH,

    x_col='file_loc',

    target_size=(IMG_WIDTH, IMG_HEIGHT)

)



# val_generator = test_datagen.flow_from_directory(

#     directory=TRAIN_PATH,

#     target_size=(IMG_WIDTH, IMG_HEIGHT)

# )



val_generator = test_datagen.flow_from_dataframe(

    val_image_locs,

    directory=TRAIN_PATH,

    x_col='file_loc',

    target_size=(IMG_WIDTH, IMG_HEIGHT),

    shuffle=False

)



test_generator = test_datagen.flow_from_directory(

    directory='../input/evaluate/',

    target_size=(IMG_WIDTH, IMG_HEIGHT),

    class_mode=None,

    batch_size=1,

    shuffle=False

)
def build_model(weights_path=None):

    model = Sequential()

    

    # First Convolution layer

    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

    model.add(Conv2D(32, (3,3), activation ='relu', padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2), strides=2))

#     model.add(Dropout(0.1))



    # Second Convolution layer

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))

    model.add(Conv2D(64, (3,3), activation ='relu', padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2), strides=2))

#     model.add(Dropout(0.1))



    # Fully-connected layers

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    

    # Output layer

    model.add(Dense(len(CATEGORIES), activation='softmax'))

    

    # Load weights if provided (used in final model)

    if weights_path is not None:

        model.load_weights(weights_path)



    # Compile using Adam optimizer

    model.compile(Adam(), loss='categorical_crossentropy', metrics=['categorical_accuracy'])



    return model
def build_model_2(weights_path=None):

    model = Sequential()

    

    # First Convolution layer

    model.add(Conv2D(16, (3,3), activation='relu', padding='same', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

    model.add(Conv2D(16, (3,3), activation='relu', padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2), strides=2))

    model.add(Dropout(0.1))

    

    # Second Convolution layer

    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))

    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2), strides=2))

    model.add(Dropout(0.1))

    

    # Third Convolution layer

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2), strides=2))

    model.add(Dropout(0.1))

    

    # Output layers

    model.add(Conv2D(len(CATEGORIES), (1, 1)))

    model.add(GlobalAveragePooling2D())

    model.add(Activation('softmax'))

    

    # Load weights if provided (used in final model)

    if weights_path is not None:

        model.load_weights(weights_path)



    # Compile using Adam optimizer

    model.compile(Adam(), loss='categorical_crossentropy', metrics=['categorical_accuracy'])



    return model
def build_model_3(weights_path=None):

    model = Sequential()

    

    # First Convolution layer

    model.add(Conv2D(8, (3,3), activation='relu', padding='same', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2), strides=2))

    model.add(Dropout(0.1))

    

    # Second Convolution layer

    model.add(Conv2D(16, (3,3), activation='relu', padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2), strides=2))

    model.add(Dropout(0.1))

    

    # Third Convolution layer

    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))

    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2), strides=2))

    model.add(Dropout(0.1))

    

    # Fully-connected layers

    model.add(Dense(64, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    

    # Output layer

    model.add(Dense(len(CATEGORIES), activation='softmax'))

    

    # Load weights if provided (used in final model)

    if weights_path is not None:

        model.load_weights(weights_path)



    # Compile using Adam optimizer

    model.compile(Adam(), loss='categorical_crossentropy', metrics=['categorical_accuracy'])



    return model
ensemble_builders = [

    build_model,

    build_model_2,

    build_model_3

]



ensemble_models = []

for i, (build) in enumerate(ensemble_builders):

    m = build()

    display(m.summary())

    ensemble_models.append((i, m))

display(ensemble_models)
model = build_model()

model.summary()
val_split = 0.2

train_steps = train_generator.n // train_generator.batch_size

val_steps = train_steps * val_split

# val_steps = val_generator.n // val_generator.batch_size



# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)



weights_path = 'best_weights.h5'

mc = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')



# history = model.fit_generator(

#     train_generator,

#     steps_per_epoch=train_steps,

#     validation_data=val_generator,

#     validation_steps=val_steps,

#     epochs=25,

# #     callbacks=[es, mc]

#     callbacks=[mc]

# )



ensemble_histories = []



for (i, m) in ensemble_models:

    mc = mc = ModelCheckpoint('{}_{}'.format(i, weights_path), monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    h = m.fit_generator(

        train_generator,

        steps_per_epoch=train_steps,

        validation_data=val_generator,

        validation_steps=val_steps,

        epochs=20,

        callbacks=[mc]

    )

    ensemble_histories.append((i, h))
# # Plot the training and validation losses over time during model training

# plt.plot(history.history['loss'], label='train')

# plt.plot(history.history['val_loss'], label='validation')

# plt.legend()

# plt.show()



for (i, h) in ensemble_histories:

    plt.plot(h.history['loss'], label='train')

    plt.plot(h.history['val_loss'], label='validation')

    plt.legend()

    plt.show()
# # Load the checkpointed "best" weights into new model

# final_model = build_model(weights_path)



ensemble_final_models = []

for i, (build) in enumerate(ensemble_builders):

    m = build('{}_{}'.format(i, weights_path))

    ensemble_final_models.append((i, m))
# display(final_model.evaluate_generator(generator=val_generator, steps=val_steps))



for (i, final_model) in ensemble_final_models:

    display(final_model.evaluate_generator(generator=val_generator, steps=val_steps))
# val_generator.reset()



# val_predictions = final_model.predict_generator(

#     val_generator,

#     steps=val_steps,

#     verbose=1

# )

# display(val_predictions.shape)



# val_predictions_labels = np.argmax(val_predictions, axis=1)



# val_true_labels = val_generator.classes[:val_predictions.shape[0]]



# display(confusion_matrix(val_true_labels, val_predictions_labels))



for (i, final_model) in ensemble_final_models:

    val_generator.reset()



    val_predictions = final_model.predict_generator(

        val_generator,

        steps=val_steps,

        verbose=1

    )

    display(val_predictions.shape)



    val_predictions_labels = np.argmax(val_predictions, axis=1)



    val_true_labels = val_generator.classes[:val_predictions.shape[0]]



    display(confusion_matrix(val_true_labels, val_predictions_labels))
model_input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))

# ensemble_outputs = [m.outputs[0] for i, m in ensemble_final_models]

yModels=[m(model_input) for i, m in ensemble_final_models] 

overall_model = Model(

    model_input,

    Average()(yModels),

    name='ensemble'

)
# test_generator.reset()



# predictions = final_model.predict_generator(

#     test_generator,

#     steps=test_generator.n,

#     verbose=1

# )



# predicted_class_indices = np.argmax(predictions, axis=1)

# display(predicted_class_indices)



test_generator.reset()



predictions = overall_model.predict_generator(

    test_generator,

    steps=test_generator.n,

    verbose=1

)



predicted_class_indices = np.argmax(predictions, axis=1)

display(predicted_class_indices)
labels = train_generator.class_indices

labels = dict((v,k) for k,v in labels.items())

display(labels)
filenames = [fn.split('/')[1] for fn in test_generator.filenames]

predictions = [labels[k] for k in predicted_class_indices]



display(filenames[:10])

display(predictions[:10])
# Save results



# results go in dataframe: first column is image filename, second column is category name

# category names are: airplane, car, cat, dog, flower, fruit, motorbike, person

df = pd.DataFrame()

df['filename'] = filenames

df['label'] = predictions

df = df.sort_values(by='filename')



df.to_csv('results.csv', header=True, index=False)