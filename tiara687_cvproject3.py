import tensorflow as tf

import numpy as np



# set the random seeds to make sure your results are reproducible

from numpy.random import seed

seed(1)

from tensorflow import set_random_seed

set_random_seed(1)



# specify path to training data and testing data



folderbig = "big"

foldersmall = "small"



train_x_location = "../input" + "/" + "x_train.csv"

train_y_location = "../input" + "/" + "y_train.csv"

test_x_location = "../input" + "/" + "x_test.csv"

test_y_location = "../input" + "/" + "y_test.csv"

print("Reading training data")

x_train_2d = np.loadtxt(train_x_location, dtype="uint8", delimiter=",")

x_train_3d = x_train_2d.reshape(-1,28,28,1)

x_train = x_train_3d

y_train = np.loadtxt(train_y_location, dtype="uint8", delimiter=",")



print("Pre processing x of training data")

x_train = x_train / 255.0

# define the training model

model = tf.keras.models.Sequential([

    tf.keras.layers.MaxPool2D(4, 4, input_shape=(28,28,1)),

#     tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.leaky_relu),

#     tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.leaky_relu),

#     tf.keras.layers.MaxPool2D(1, 1),

#     tf.keras.layers.Dropout(0.25),        

    

    tf.keras.layers.Conv2D(126, (3, 3), padding='same', activation=tf.nn.leaky_relu),

    tf.keras.layers.Conv2D(126, (3, 3), padding='same', activation=tf.nn.leaky_relu),

    tf.keras.layers.MaxPool2D(1, 1),

    tf.keras.layers.Dropout(0.25),    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(10, activation=tf.nn.softmax)

])

# Define the optimizer

optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model

model.compile(optimizer = optimizer , loss = "sparse_categorical_crossentropy", metrics=["accuracy"])

# Set a learning rate annealer  monitor = loss, lr, acc.

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.000000001)

epochs = 30 # Turn epochs to 30 to get 0.9967 accuracy

batch_size = 56

datagen = tf.keras.preprocessing.image.ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.05, # Randomly zoom image 

        width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



datagen.fit(x_train)

print("train")

# model.fit(x_train, y_train, epochs=26)

# Fit the model

%time model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size), epochs = epochs, verbose = 2, steps_per_epoch=x_train.shape[0], callbacks=[learning_rate_reduction])

print("Reading testing data")

x_test_2d = np.loadtxt(test_x_location, dtype="uint8", delimiter=",")

x_test_3d = x_test_2d.reshape(-1,28,28,1)

x_test = x_test_3d

y_test = np.loadtxt(test_y_location, dtype="uint8", delimiter=",")

print("Pre processing testing data")

x_test = x_test / 255.0

print("evaluate")

%time model.evaluate(x_test, y_test)