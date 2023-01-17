import os

import pickle

import numpy as np

import pandas as pd

import tensorflow as tf

from sklearn.metrics import classification_report

from collections import Counter

import matplotlib.pyplot as plt

%matplotlib inline



print(tf.__version__)
def load_dataset():

  df = pd.read_html("https://www.cs.toronto.edu/~kriz/cifar.html")

  cifar10_classes = df[0][0].values.tolist()

  num_classes = len(cifar10_classes)



  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()



  y_train = tf.keras.utils.to_categorical(y_train, num_classes)

  y_test = tf.keras.utils.to_categorical(y_test, num_classes)



  x_train = x_train.astype('float32')

  x_test = x_test.astype('float32')

  x_train = x_train / 255.0

  x_test = x_test / 255.0



  return x_train, y_train, x_test, y_test, np.array(cifar10_classes)
def normalize_images(train, test):

  mean = np.mean(train, axis=(0,1,2,3))

  std = np.std(train, axis=(0,1,2,3))

  train_norm = (train - mean)/(std + 1e-7)

  test_norm = (test - mean)/(std + 1e-7)

  

  return train_norm, test_norm
def define_model():

    weight_decay = 1e-4

    L2 = tf.keras.regularizers.l2(weight_decay)



    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu", kernel_regularizer=L2, input_shape=x_train.shape[1:]),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu", kernel_regularizer=L2),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'), 

        tf.keras.layers.Dropout(0.2), 



        tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu", kernel_regularizer=L2),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu", kernel_regularizer=L2),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),

        tf.keras.layers.Dropout(0.3), 



        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu", kernel_regularizer=L2),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu", kernel_regularizer=L2),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),

        tf.keras.layers.Dropout(0.4), 



        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(units=128, activation='relu'), 

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(units=128, activation='relu'), 

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(units=128, activation='relu'), 

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(units=128, activation='relu'), 

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(units=10, activation='softmax')

    ])



    opt = tf.keras.optimizers.RMSprop(lr=0.001, decay=1e-6)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])



    return model
def summarize_diagnostics(history):

    plt.subplot(211)

    plt.title('Cross Entropy Loss')

    plt.plot(history.history['loss'], color='blue', label='train')

    plt.plot(history.history['val_loss'], color='orange', label='test')



    plt.subplot(212)

    plt.title('Classification Accuracy')

    plt.plot(history.history['accuracy'], color='blue', label='train')

    plt.plot(history.history['val_accuracy'], color='orange', label='test')
def plotImages(images_arr):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.axis('off')

    plt.tight_layout()

    plt.show()
x_train_df, y_train, x_test_df, y_test, classes_names = load_dataset()

x_train_df.shape, y_train.shape, x_test_df.shape, y_test.shape
sample_training_images, _ = next(tf.keras.preprocessing.image.ImageDataGenerator().flow(x_train_df, y_train, batch_size=64))

plotImages(sample_training_images[:5])
x_train, x_test = normalize_images(x_train_df, x_test_df)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=True,

    rotation_range=15

)



batch_size = 64



train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(

    monitor='val_loss', factor=0.5, patience=5, verbose=1, 

    mode='auto', min_delta=1e-10, cooldown=0, min_lr=0

)



early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor='val_loss', min_delta=0, patience=12, verbose=1, mode='auto',

    baseline=None, restore_best_weights=False

)
csv_logger = tf.keras.callbacks.CSVLogger(

    'cifar10.epoch.results.csv', separator='|', append=False)



model_checkpoint = tf.keras.callbacks.ModelCheckpoint(

    "cifar10.partial.hdf5", save_weights_only=True, mode='auto', 

    save_freq='epoch', verbose=0

)
model = define_model()

model.summary()
epochs = 1000



history = model.fit(

    train_generator, 

    steps_per_epoch=x_train.shape[0]//batch_size, 

    epochs=epochs,  

    validation_data=(x_test, y_test), 

    callbacks=[csv_logger, reduce_learning_rate, early_stopping, model_checkpoint],

    verbose=1

)
_, acc = model.evaluate(x_test, y_test, verbose=0)

print('> %.3f' % (acc * 100.0))
model.save('cifar10.h5', overwrite=True, include_optimizer=True, save_format='h5')
summarize_diagnostics(history)
res = pd.read_csv('cifar10.epoch.results.csv', sep='|')

res.tail()
model_load_tf = tf.keras.models.load_model('cifar10.h5')

model_load_tf.summary()
test_loss, test_acc = model_load_tf.evaluate(x_test, y_test)

print(f"Accuracy: {test_acc} Loss: {test_loss}")
Y_test = np.argmax(y_test, axis=1)

y_pred = model.predict_classes(x_test)

print(classification_report(Y_test, y_pred))