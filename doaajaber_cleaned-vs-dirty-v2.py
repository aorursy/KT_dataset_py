# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
! pip install tf-nightly
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers
! unzip /kaggle/input/platesv2/plates.zip
! ls

! ls plates
image_size=(200,200)

batch_size=10
! ls plates/train
image_size = (200, 200)

batch_size = 10



train_ds = tf.keras.preprocessing.image_dataset_from_directory(

    "plates/train",

    validation_split=0.2,

    subset="training",

    seed=1337,

    image_size=image_size,

    batch_size=batch_size,

)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(

    "plates/train",

    validation_split=0.2,

    subset="validation",

    seed=1337,

    image_size=image_size,

    batch_size=batch_size,

)
import matplotlib.pyplot as plt



plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(int(labels[i]))

        plt.axis("off")
data_augmentation = keras.Sequential(

    [

        layers.experimental.preprocessing.RandomFlip("horizontal"),

        layers.experimental.preprocessing.RandomRotation(0.1),

        layers.experimental.preprocessing.RandomContrast(.2),

       

        layers.experimental.preprocessing.Normalization()





    ]

)
plt.figure(figsize=(10, 10))

for images, _ in train_ds.take(1):

    for i in range(9):

        augmented_images = data_augmentation(images)

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(augmented_images[0].numpy().astype("uint8"))

        plt.axis("off")
train_ds = train_ds.prefetch(buffer_size=10)

val_ds = val_ds.prefetch(buffer_size=10)
def make_model(input_shape, num_classes):

    inputs = keras.Input(shape=input_shape)

    # Image augmentation block

    x = data_augmentation(inputs)



    # Entry block

    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)

    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)



    x = layers.Conv2D(64, 3, padding="same")(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)



    previous_block_activation = x  # Set aside residual



    for size in [128, 256, 512, 728]:

        x = layers.Activation("relu")(x)

        x = layers.SeparableConv2D(size, 3, padding="same")(x)

        x = layers.BatchNormalization()(x)



        x = layers.Activation("relu")(x)

        x = layers.SeparableConv2D(size, 3, padding="same")(x)

        x = layers.BatchNormalization()(x)



        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)



        # Project residual

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(

            previous_block_activation

        )

        x = layers.add([x, residual])  # Add back residual

        previous_block_activation = x  # Set aside next residual



    x = layers.SeparableConv2D(1024, 3, padding="same")(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)



    x = layers.GlobalAveragePooling2D()(x)

    if num_classes == 2:

        activation = "sigmoid"

        units = 1

    else:

        activation = "softmax"

        units = num_classes



    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(units, activation=activation)(x)

    return keras.Model(inputs, outputs)





model = make_model(input_shape=image_size + (3,), num_classes=2)

keras.utils.plot_model(model, show_shapes=True)
epochs = 50







callbacks = [

    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),tf.keras.callbacks.EarlyStopping(

    monitor='val_loss', min_delta=0, patience=5,

    restore_best_weights=True

)

    

]

model.compile(

    optimizer=keras.optimizers.Adam(1e-3),

    loss="binary_crossentropy",

    metrics=["accuracy"],

)

model.fit(

    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,

)
sub_df = pd.read_csv('../input/platesv2/sample_submission.csv')

sub_df.head()
from keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(  

        'plates',

        classes=['test'],

        target_size = (200, 200),

        batch_size = 1,

        shuffle = False,        

        class_mode = None) 
test_generator.reset()

predict = model.predict_generator(test_generator, steps = len(test_generator.filenames))
sub_df['label'] = predict

sub_df['label'] = sub_df['label'].apply(lambda x: 'dirty' if x > 0.5 else 'cleaned')

sub_df.head()
sub_df.to_csv('sub.csv', index=False)