# Please turn the 'Internet' toggle On in the Settings panel to your left, in order to make changes to this kernel.

!pip install wandb

import sys

!{sys.executable} -m pip install --upgrade git+git://github.com/wandb/client.git@master#egg=wandb 
# Essentials

import numpy as np

import pandas as pd



# Models

import tensorflow as tf



# Ignore useless warnings

import warnings

warnings.filterwarnings(action="ignore")



# Set random state for numpy

np.random.seed(42)



# WandB

import wandb

from wandb.keras import WandbCallback

# You can change your project name here. For more config options, see https://docs.wandb.com/docs/init.html

wandb.init(anonymous='allow', project="neural-nets-kaggle", name="basic_neural_network")



# Go to https://app.wandb.ai/authorize to get your WandB key
config = wandb.config # Config is a variable that holds and saves hyperparameters and inputs

config.dropout = 0.2

config.conv_layer_1_size  = 32

config.conv_layer_2_size = 64

config.conv_layer_3_size = 128

config.hidden_layer_size = 512

config.learn_rate = 0.01

config.learn_rate_low = 0.001

config.kernel_size = 3

config.pool_size = 2

config.decay = 1e-6

config.momentum = 0.9

config.n_epochs = 25



config.img_width=28

config.img_height=28

config.num_classes = 10

config.batch_size = 128

config.validation_size = 5000
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

labels=["T-shirt/top","Trouser","Pullover","Dress","Coat",

        "Sandal","Shirt","Sneaker","Bag","Ankle boot"]



# Normalize pixel values

X_train_full, X_test = X_train_full / 255.0, X_test / 255.0



#reshape input data

X_train_full = X_train_full.reshape(X_train_full.shape[0], config.img_width, config.img_height, 1)

X_test = X_test.reshape(X_test.shape[0], config.img_width, config.img_height, 1)



# one hot encode outputs

y_train_full = tf.keras.utils.to_categorical(y_train_full)

y_test = tf.keras.utils.to_categorical(y_test)

num_classes = y_test.shape[1]



# Split into validation, and training sets

X_valid, X_train = X_train_full[:config.validation_size], X_train_full[config.validation_size:]

y_valid, y_train = y_train_full[:config.validation_size], y_train_full[config.validation_size:]

X_train_full.shape, X_train.shape, X_valid.shape, X_test.shape
wandb.run.get_url()
%%wandb

# build model

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(config.conv_layer_1_size, kernel_size=(config.kernel_size, config.kernel_size), activation='relu', input_shape=(config.img_width, config.img_height, 1)))

model.add(tf.keras.layers.Conv2D(config.conv_layer_1_size, kernel_size=(config.kernel_size, config.kernel_size), activation='relu'))

model.add(tf.keras.layers.MaxPooling2D((config.pool_size, config.pool_size)))

model.add(tf.keras.layers.Conv2D(config.conv_layer_2_size, kernel_size=(config.kernel_size, config.kernel_size), activation='relu'))

model.add(tf.keras.layers.Conv2D(config.conv_layer_2_size, kernel_size=(config.kernel_size, config.kernel_size), activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(config.pool_size, config.pool_size)))

model.add(tf.keras.layers.Conv2D(config.conv_layer_3_size, (config.kernel_size, config.kernel_size), activation='relu'))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(config.hidden_layer_size, activation='relu'))

model.add(tf.keras.layers.Dense(config.num_classes, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=config.learn_rate), metrics=['accuracy'])

model.fit(X_train, y_train,  validation_data=(X_valid, y_valid), epochs=config.n_epochs,

        callbacks=[WandbCallback(data_type="image", validation_data=(X_valid, y_valid), labels=labels)])
wandb.init(project="building-neural-nets", name="lower_learning_rate")
%%wandb

# build model

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(config.conv_layer_1_size, kernel_size=(config.kernel_size, config.kernel_size), activation='relu', input_shape=(config.img_width, config.img_height, 1)))

model.add(tf.keras.layers.Conv2D(config.conv_layer_1_size, kernel_size=(config.kernel_size, config.kernel_size), activation='relu'))

model.add(tf.keras.layers.MaxPooling2D((config.pool_size, config.pool_size)))

model.add(tf.keras.layers.Conv2D(config.conv_layer_2_size, kernel_size=(config.kernel_size, config.kernel_size), activation='relu'))

model.add(tf.keras.layers.Conv2D(config.conv_layer_2_size, kernel_size=(config.kernel_size, config.kernel_size), activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(config.pool_size, config.pool_size)))

model.add(tf.keras.layers.Conv2D(config.conv_layer_3_size, (config.kernel_size, config.kernel_size), activation='relu'))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(config.hidden_layer_size, activation='relu'))

model.add(tf.keras.layers.Dense(config.num_classes, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=config.learn_rate_low), metrics=['accuracy'])

model.fit(X_train, y_train,  validation_data=(X_valid, y_valid), epochs=config.n_epochs,

        callbacks=[WandbCallback(data_type="image", validation_data=(X_valid, y_valid), labels=labels)])
wandb.init(project="building-neural-nets", name="momentum")
%%wandb

# build model

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(config.conv_layer_1_size, kernel_size=(config.kernel_size, config.kernel_size), activation='relu', input_shape=(config.img_width, config.img_height, 1)))

model.add(tf.keras.layers.Conv2D(config.conv_layer_1_size, kernel_size=(config.kernel_size, config.kernel_size), activation='relu'))

model.add(tf.keras.layers.MaxPooling2D((config.pool_size, config.pool_size)))

model.add(tf.keras.layers.Conv2D(config.conv_layer_2_size, kernel_size=(config.kernel_size, config.kernel_size), activation='relu'))

model.add(tf.keras.layers.Conv2D(config.conv_layer_2_size, kernel_size=(config.kernel_size, config.kernel_size), activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(config.pool_size, config.pool_size)))

model.add(tf.keras.layers.Conv2D(config.conv_layer_3_size, (config.kernel_size, config.kernel_size), activation='relu'))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(config.hidden_layer_size, activation='relu'))

model.add(tf.keras.layers.Dense(config.num_classes, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=config.learn_rate_low, decay=config.decay, momentum=config.momentum,

                            nesterov=True), metrics=['accuracy'])

model.fit(X_train, y_train,  validation_data=(X_valid, y_valid), epochs=config.n_epochs,

        callbacks=[WandbCallback(data_type="image", validation_data=(X_valid, y_valid), labels=labels)])
wandb.init(project="building-neural-nets", name="vanishing_gradients")
%%wandb

# build model

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(config.conv_layer_1_size, kernel_size=(config.kernel_size, config.kernel_size), activation='selu', kernel_initializer='lecun_normal',

                 input_shape=(config.img_width, config.img_height, 1)))

tf.keras.layers.BatchNormalization(),

model.add(tf.keras.layers.Conv2D(config.conv_layer_1_size, kernel_size=(config.kernel_size, config.kernel_size), activation='selu', kernel_initializer='lecun_normal'))

tf.keras.layers.BatchNormalization(),

model.add(tf.keras.layers.MaxPooling2D((config.pool_size, config.pool_size)))

model.add(tf.keras.layers.Conv2D(config.conv_layer_2_size, kernel_size=(config.kernel_size, config.kernel_size), activation='selu', kernel_initializer='lecun_normal'))

tf.keras.layers.BatchNormalization(),

model.add(tf.keras.layers.Conv2D(config.conv_layer_2_size, kernel_size=(config.kernel_size, config.kernel_size), activation='selu', kernel_initializer='lecun_normal'))

tf.keras.layers.BatchNormalization(),

model.add(tf.keras.layers.MaxPooling2D(pool_size=(config.pool_size, config.pool_size)))

model.add(tf.keras.layers.Conv2D(config.conv_layer_3_size, (config.kernel_size, config.kernel_size), activation='selu', kernel_initializer='lecun_normal'))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(config.hidden_layer_size, activation='selu', kernel_initializer='lecun_normal'))

tf.keras.layers.BatchNormalization(),

model.add(tf.keras.layers.Dense(config.num_classes, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=config.learn_rate, clipnorm=1.0),

              metrics=['accuracy'])

model.fit(X_train, y_train,  validation_data=(X_test, y_test), epochs=config.n_epochs,

          callbacks=[WandbCallback(data_type="image", validation_data=(X_valid, y_valid), labels=labels), tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
wandb.init(project="building-neural-nets", name="dropout")
%%wandb

# build model

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(config.conv_layer_1_size, kernel_size=(config.kernel_size, config.kernel_size), activation='selu', kernel_initializer='lecun_normal',

                 input_shape=(config.img_width, config.img_height, 1)))

tf.keras.layers.BatchNormalization(),

model.add(tf.keras.layers.Conv2D(config.conv_layer_1_size, kernel_size=(config.kernel_size, config.kernel_size), activation='selu', kernel_initializer='lecun_normal'))

tf.keras.layers.BatchNormalization(),

model.add(tf.keras.layers.MaxPooling2D((config.pool_size, config.pool_size)))

tf.keras.layers.AlphaDropout(rate=config.dropout),

model.add(tf.keras.layers.Conv2D(config.conv_layer_2_size, kernel_size=(config.kernel_size, config.kernel_size), activation='selu', kernel_initializer='lecun_normal'))

tf.keras.layers.BatchNormalization(),

model.add(tf.keras.layers.Conv2D(config.conv_layer_2_size, kernel_size=(config.kernel_size, config.kernel_size), activation='selu', kernel_initializer='lecun_normal'))

tf.keras.layers.BatchNormalization(),

model.add(tf.keras.layers.MaxPooling2D(pool_size=(config.pool_size, config.pool_size)))

tf.keras.layers.AlphaDropout(rate=config.dropout),

model.add(tf.keras.layers.Conv2D(config.conv_layer_3_size, (config.kernel_size, config.kernel_size), activation='selu', kernel_initializer='lecun_normal'))

tf.keras.layers.AlphaDropout(rate=config.dropout),

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(config.hidden_layer_size, activation='selu', kernel_initializer='lecun_normal'))

tf.keras.layers.BatchNormalization(),

tf.keras.layers.AlphaDropout(rate=config.dropout),

model.add(tf.keras.layers.Dense(config.num_classes, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=config.learn_rate, clipnorm=1.0),

              metrics=['accuracy'])

model.fit(X_train, y_train,  validation_data=(X_test, y_test), epochs=config.n_epochs,

          callbacks=[WandbCallback(data_type="image", validation_data=(X_valid, y_valid), labels=labels), tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
wandb.init(project="building-neural-nets", name="nadam_optimizer")
%%wandb

# build model

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(config.conv_layer_1_size, kernel_size=(config.kernel_size, config.kernel_size), activation='selu', kernel_initializer='lecun_normal',

         input_shape=(config.img_width, config.img_height, 1)))

tf.keras.layers.BatchNormalization(),

model.add(tf.keras.layers.Conv2D(config.conv_layer_1_size, kernel_size=(config.kernel_size, config.kernel_size), activation='selu', kernel_initializer='lecun_normal'))

tf.keras.layers.BatchNormalization(),

model.add(tf.keras.layers.MaxPooling2D((config.pool_size, config.pool_size)))

tf.keras.layers.AlphaDropout(rate=config.dropout),

model.add(tf.keras.layers.Conv2D(config.conv_layer_2_size, kernel_size=(config.kernel_size, config.kernel_size), activation='selu', kernel_initializer='lecun_normal'))

tf.keras.layers.BatchNormalization(),

model.add(tf.keras.layers.Conv2D(config.conv_layer_2_size, kernel_size=(config.kernel_size, config.kernel_size), activation='selu', kernel_initializer='lecun_normal'))

tf.keras.layers.BatchNormalization(),

model.add(tf.keras.layers.MaxPooling2D(pool_size=(config.pool_size, config.pool_size)))

tf.keras.layers.AlphaDropout(rate=config.dropout),

model.add(tf.keras.layers.Conv2D(config.conv_layer_3_size, (config.kernel_size, config.kernel_size), activation='selu', kernel_initializer='lecun_normal'))

tf.keras.layers.AlphaDropout(rate=config.dropout),

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(config.hidden_layer_size, activation='selu', kernel_initializer='lecun_normal'))

tf.keras.layers.BatchNormalization(),

tf.keras.layers.AlphaDropout(rate=config.dropout),

model.add(tf.keras.layers.Dense(config.num_classes, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Nadam(lr=config.learn_rate, beta_1=0.9, beta_2=0.999, clipnorm=1.0), metrics=['accuracy'])

model.fit(X_train, y_train,  validation_data=(X_test, y_test), epochs=config.n_epochs,

    callbacks=[WandbCallback(data_type="image", labels=labels, validation_data=(X_valid, y_valid)), tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
wandb.init(project="building-neural-nets", name="learningrate")
%%wandb

# build model

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(config.conv_layer_1_size, kernel_size=(config.kernel_size, config.kernel_size), activation='selu', kernel_initializer='lecun_normal',

         input_shape=(config.img_width, config.img_height, 1)))

tf.keras.layers.BatchNormalization(),

model.add(tf.keras.layers.Conv2D(config.conv_layer_1_size, kernel_size=(config.kernel_size, config.kernel_size), activation='selu', kernel_initializer='lecun_normal'))

tf.keras.layers.BatchNormalization(),

model.add(tf.keras.layers.MaxPooling2D((config.pool_size, config.pool_size)))

tf.keras.layers.AlphaDropout(rate=config.dropout),

model.add(tf.keras.layers.Conv2D(config.conv_layer_2_size, kernel_size=(config.kernel_size, config.kernel_size), activation='selu', kernel_initializer='lecun_normal'))

tf.keras.layers.BatchNormalization(),

model.add(tf.keras.layers.Conv2D(config.conv_layer_2_size, kernel_size=(config.kernel_size, config.kernel_size), activation='selu', kernel_initializer='lecun_normal'))

tf.keras.layers.BatchNormalization(),

model.add(tf.keras.layers.MaxPooling2D(pool_size=(config.pool_size, config.pool_size)))

tf.keras.layers.AlphaDropout(rate=config.dropout),

model.add(tf.keras.layers.Conv2D(config.conv_layer_3_size, (config.kernel_size, config.kernel_size), activation='selu', kernel_initializer='lecun_normal'))

tf.keras.layers.AlphaDropout(rate=config.dropout),

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(config.hidden_layer_size, activation='selu', kernel_initializer='lecun_normal'))

tf.keras.layers.BatchNormalization(),

tf.keras.layers.AlphaDropout(rate=config.dropout),

model.add(tf.keras.layers.Dense(config.num_classes, activation='softmax'))



lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Nadam(lr=config.learn_rate, beta_1=0.9, beta_2=0.999, clipnorm=1.0), metrics=['accuracy'])

model.fit(X_train, y_train,  validation_data=(X_test, y_test), epochs=config.n_epochs,

    callbacks=[WandbCallback(data_type="image", labels=labels, validation_data=(X_valid, y_valid)), tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True), lr_scheduler])
model.evaluate(X_test, y_test)
model.save("fashion_mnist_model.h5")