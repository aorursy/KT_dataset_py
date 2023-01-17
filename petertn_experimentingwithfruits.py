import cv2

import matplotlib.pyplot as plt

import numpy as np

import random

from pathlib import Path

from keras.preprocessing.image import ImageDataGenerator



# List all fruit classes

training_data_dir = Path("../input/fruits-360_dataset/fruits-360/Training")

test_data_dir = Path("../input/fruits-360_dataset/fruits-360/Test")

fruit_classes = [str(d.name) for d in training_data_dir.iterdir()]

print("Fruit types: {}".format(", ".join(fruit_classes)))

print("Number of fruits: {}".format(len(fruit_classes)))



# Reading one example image to get the size of the data

image_path = training_data_dir / 'Cocos' / 'r_33_100.jpg'

test_img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

print("Image shape: {}".format(test_img.shape))



# Set up the ImageDataGenerator and read data using flow_from_directory

selected_fruits = fruit_classes[0:4]

batch_size=128

example_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(

        str(training_data_dir),

        classes=selected_fruits,

        class_mode='categorical',

        color_mode='rgb',

        batch_size=batch_size,

        target_size=test_img.shape[0:2])



# Show a sample from the training data by extracting a batch

batch_data = next(example_generator)

images = batch_data[0]

classes = batch_data[1]

print("Batch images shape: {}".format(images.shape))

print("Batch classes shape: {}".format(classes.shape))

index = random.randint(0, batch_size-1)

image = images[index, :, :, :]

class_index = np.where(classes[index, :] == 1)[0][0]

class_str = selected_fruits[class_index]

print("Below fruit class: {}".format(class_str))

plt.imshow(image)
import keras



# Useful constants

INPUT_SHAPE = test_img.shape

TRAINING_DIR = training_data_dir

TEST_DIR = test_data_dir

FRUIT_CLASSES = fruit_classes

EXAMPLES_PER_CLASS = 500 # This is an approximate number

VALIDATION_RATIO = 0.25

NUM_CLASSES = len(FRUIT_CLASSES)



DEFAULT_DATAGEN_ARGS = dict(rescale=1./255)



def get_data_generators(data_gen_args=DEFAULT_DATAGEN_ARGS,

                        selected_fruits=None,

                        color_mode='rgb',

                        target_size=INPUT_SHAPE[0:2],

                        batch_size=32):

    training_generator = ImageDataGenerator(**data_gen_args).flow_from_directory(

        str(TRAINING_DIR),

        classes=selected_fruits,

        class_mode='categorical',

        color_mode=color_mode,

        batch_size=batch_size,

        target_size=target_size)

    # No augmentation applied for the validation data

    validation_generator = ImageDataGenerator(**DEFAULT_DATAGEN_ARGS).flow_from_directory(

        str(TEST_DIR),

        classes=selected_fruits,

        class_mode='categorical',

        color_mode=color_mode,

        batch_size=batch_size,

        target_size=target_size)

    return training_generator, validation_generator





def plot_history(history):

    # Plot training and validation accuracy values

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.title('Model accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()



    # Plot training and validation loss values

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()



def train_fruit_classifier(model, data_gen_args=DEFAULT_DATAGEN_ARGS, epochs=10, batch_size=128, selected_fruits=FRUIT_CLASSES):

    num_fruits = len(selected_fruits)



    # Approximately the size of the training set

    steps_per_epoch = (EXAMPLES_PER_CLASS * num_fruits) // batch_size



    train_generator, val_generator = get_data_generators(

        data_gen_args=data_gen_args, selected_fruits=selected_fruits, batch_size=batch_size)



    training_history = model.fit_generator(

            train_generator,

            steps_per_epoch=steps_per_epoch,

            validation_data=val_generator,

            validation_steps=int(steps_per_epoch * VALIDATION_RATIO),

            epochs=epochs)



    return training_history

# Define a simple model

from keras.models import Sequential

from keras.layers import Activation

from keras.layers import Conv2D

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers import MaxPooling2D



def create_simple_model(num_classes=NUM_CLASSES):

    model = Sequential()



    # Convolutional/downsampling layers

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=INPUT_SHAPE))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))  # Output shape (50, 50, 32)



    model.add(Conv2D(32, (3, 3), padding='same'))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))  # Output shape (25, 25, 32)



    model.add(Conv2D(64, (3, 3), padding='same'))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))  # Output shape (12, 12, 64)



    # Flatten data to be able to apply dense layers

    model.add(Flatten()) # Output size = 12 * 12 * 64 = 9216

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))



    return model

# Select some fruits

num_fruits = 4

selected_fruits = FRUIT_CLASSES[0:num_fruits]

print("Selected fruits: {}".format(", ".join(selected_fruits)))



# Create the model

model = create_simple_model(num_fruits)

model.compile(optimizer='adam',

              loss=keras.losses.categorical_crossentropy,

              metrics=['accuracy'])

model.summary()



# Train

training_history = train_fruit_classifier(

    model, epochs=10, batch_size=128, selected_fruits=selected_fruits)



# Display result

plot_history(training_history)



model.save_weights('simple_model_v1_fruits_4.h5')
# Use all fruit classes (default)

model = create_simple_model()

model.compile(optimizer='adam',

              loss=keras.losses.categorical_crossentropy,

              metrics=['accuracy'])

model.summary()



# Train

training_history = train_fruit_classifier(model, epochs=20, batch_size=128)



# Display results

plot_history(training_history)



model.save_weights('simple_model_v1_fruits_95.h5')
from keras.models import Sequential

from keras.layers import Activation

from keras.layers import BatchNormalization

from keras.layers import Conv2D

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers import MaxPooling2D





def add_relu_bottleneck_layer(model, num_filters, filter_size, 

                              dropout_rate=0.0, use_batch_norm=False, first_layer=True):

    if first_layer:

        model.add(Conv2D(num_filters, (filter_size, filter_size), padding='same', input_shape=INPUT_SHAPE))

    else:

        model.add(Conv2D(num_filters, (filter_size, filter_size), padding='same'))

    if use_batch_norm:

        model.add(BatchNormalization())

    model.add(Activation('relu')) # Uncertain if this should be before or after batch norm

    if dropout_rate > 0.0:

        model.add(Dropout(dropout_rate))

    model.add(MaxPooling2D(pool_size=(2, 2)))





def create_extended_model(num_classes=NUM_CLASSES,

                          fc_units=128,

                          use_batch_norm=False,

                          conv_dropout_rate=0.0,

                          fc_dropout_rate=0.5):

    model = Sequential()



    # Bottleneck layers

    add_relu_bottleneck_layer(model, 32, 3, dropout_rate=conv_dropout_rate, use_batch_norm=use_batch_norm,

                              first_layer=True)

    add_relu_bottleneck_layer(model, 64, 3, dropout_rate=conv_dropout_rate, use_batch_norm=use_batch_norm)

    add_relu_bottleneck_layer(model, 64, 3, dropout_rate=conv_dropout_rate, use_batch_norm=use_batch_norm)



    # Flatten data to be able to apply dense layers

    model.add(Flatten())

    model.add(Dense(fc_units))

    if use_batch_norm:

        model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(fc_dropout_rate))

    model.add(Dense(num_classes, activation='softmax'))



    return model

# Train extended model

# Use batch normalization and dropout on convolutional layers

model = create_extended_model(num_classes=NUM_CLASSES,

                              fc_units=128,

                              use_batch_norm=True,

                              conv_dropout_rate=0.2,

                              fc_dropout_rate=0.5)

model.compile(optimizer='adam',

              loss=keras.losses.categorical_crossentropy,

              metrics=['accuracy'])

model.summary()



# Train

training_history = train_fruit_classifier(model, epochs=20, batch_size=128)



# Display results

plot_history(training_history)



model.save_weights('extended_model_v1_fruits_95.h5')