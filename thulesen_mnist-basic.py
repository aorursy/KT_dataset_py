import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import layers, models

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tqdm.keras import TqdmCallback # progress bars
img_dims = (28, 28)
# DATA LOADING



path_train = "../input/digit-recognizer/train.csv"

array = pd.read_csv(path_train).to_numpy()

y = array[:, 0].reshape(-1, 1) # first column is the labels

# rest of the columns are the pixel values, reshape to 28x28 and normalize

X = array[:, 1:].reshape(-1, *img_dims, 1) / 255.0

ds_train = tf.data.Dataset.from_tensor_slices((X, y))



path_test = "../input/digit-recognizer/test.csv"

array = pd.read_csv(path_test).to_numpy()

X = array[:, :].reshape(-1, *img_dims, 1) / 255.0

ds_test = tf.data.Dataset.from_tensor_slices((X))
# SPLIT TRAIN INTO TRAIN/DEV



ds = ds_train.shuffle(buffer_size=10000).cache().enumerate()

ds_train = ds.filter(lambda i, data: i % 10 <= 7).map(lambda _, data: data).cache()

ds_dev = ds.filter(lambda i, data: i % 10 > 7).map(lambda _, data: data).cache()
# BATCHING

BATCH_SIZE = 1024



ds_train = ds_train.batch(BATCH_SIZE)

ds_dev = ds_dev.batch(BATCH_SIZE)

ds_test = ds_test.batch(BATCH_SIZE)
experiments = []



def experiment(model, comment, epochs=100, ax=None, ds=ds_train):

    global experiments

    

    model.compile(optimizer='Adam',

                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

                  metrics=['accuracy'],

                 )

    history = model.fit(ds, epochs=epochs, validation_data=ds_dev, verbose=0,

                        callbacks=[TqdmCallback(verbose=1)]

                       ).history

    if ax:

        ax.plot(history['accuracy'])

        ax.plot(history['val_accuracy'])

        ax.set(ylim=((0.9, 1)))

        ax.set(title=comment)

    

    experiments.append({

        'model': model,

        'history': history,

        'acc': history['accuracy'][-1],

        'val_acc': history['val_accuracy'][-1],

    })
comment = 'Logistic regression'

model = models.Sequential([

    layers.InputLayer(input_shape=(*img_dims, 1)),

    layers.Flatten(),

    layers.Dense(10, activation='linear'),

])

model.summary()

fig, ax = plt.subplots()

experiment(model, comment, epochs=100, ax=ax)
comment = 'Deep logistic regression'

model = models.Sequential([

    layers.InputLayer(input_shape=(*img_dims, 1)),

    layers.Flatten(),

    layers.Dense(400, activation='relu'),

    layers.Dense(10, activation='linear'),

])

model.summary()

fig, ax = plt.subplots()

experiment(model, comment, epochs=100, ax=ax)
comment = 'LeNet-5'

model = models.Sequential([

    layers.InputLayer(input_shape=(*img_dims, 1)),

    layers.Conv2D(filters=6, kernel_size=5, activation='relu'),

    layers.MaxPool2D(pool_size=2),

    layers.Conv2D(filters=16, kernel_size=5, activation='relu'),

    layers.MaxPool2D(pool_size=2),

    layers.Flatten(),

    layers.Dense(120, activation='relu'),

    layers.Dense(84, activation='relu'),

    layers.Dense(10),

])

model.summary()

fig, ax = plt.subplots()

experiment(model, comment, epochs=20, ax=ax)
comment = 'LeNet-5 w/ small kernel, more channels'

model = models.Sequential([

    layers.InputLayer(input_shape=(*img_dims, 1)),

    layers.Conv2D(filters=32, kernel_size=3, activation='relu'),

    layers.MaxPool2D(pool_size=2),

    layers.Conv2D(filters=64, kernel_size=3, activation='relu'),

    layers.MaxPool2D(pool_size=2),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    layers.Dense(10),

])

model.summary()

fig, ax = plt.subplots()

experiment(model, comment, epochs=50, ax=ax)
data_augmentation = models.Sequential([  

    layers.experimental.preprocessing.RandomRotation(0.05),

    layers.experimental.preprocessing.RandomTranslation(0.1, 0.1),

    layers.experimental.preprocessing.RandomZoom(0.1),

])
plt.figure(figsize=(10, 10))

for images, _ in ds_train.take(1):

    for i in range(9):

        augmented_images = data_augmentation(images)

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(augmented_images[0].numpy().reshape(img_dims))

        plt.axis("off")
comment = 'LeNet-5 (modified) w/ data augmentation'

model = models.Sequential([

    layers.InputLayer(input_shape=(*img_dims, 1)),

    data_augmentation,

    layers.Conv2D(filters=32, kernel_size=3, activation='relu'),

    layers.MaxPool2D(pool_size=2),

    layers.Conv2D(filters=64, kernel_size=3, activation='relu'),

    layers.MaxPool2D(pool_size=2),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    layers.Dense(10),

])

model.summary()

fig, ax = plt.subplots()

experiment(model, comment, epochs=100, ax=ax)
comment = 'LeNet-5 (modified) w/ data augmentation'

final_model = models.Sequential([

    layers.InputLayer(input_shape=(*img_dims, 1)),

    data_augmentation,

    layers.Conv2D(filters=32, kernel_size=3, activation='relu'),

    layers.MaxPool2D(pool_size=2),

    layers.Conv2D(filters=64, kernel_size=3, activation='relu'),

    layers.MaxPool2D(pool_size=2),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    layers.Dense(10),

])

final_model.summary()

fig, ax = plt.subplots()

experiment(final_model, comment, epochs=100, ds=ds_train.concatenate(ds_dev))
predictions = np.argmax(final_model.predict(ds_test), axis=1)

submission = pd.DataFrame({'ImageId': np.arange(1, len(predictions)+1),

                           'Label': predictions,

                          })

submission
submission.to_csv('submission.csv', index=False)