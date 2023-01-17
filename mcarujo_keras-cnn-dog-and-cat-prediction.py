!pip install nb_black -q
%load_ext nb_black
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
train_datagen = ImageDataGenerator(

    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True

)

training_set = train_datagen.flow_from_directory(

    "../input/cat-and-dog/training_set/training_set",

    target_size=(128, 128),

    batch_size=50,

    class_mode="binary",

)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_set = test_datagen.flow_from_directory(

    "../input/cat-and-dog/test_set/test_set",

    target_size=(128, 128),

    batch_size=50,

    class_mode="binary",

)
cnn = tf.keras.models.Sequential()

cnn.add(

    tf.keras.layers.Conv2D(

        filters=32, kernel_size=3, activation="relu", input_shape=[128, 128, 3]

    )

)

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu"))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128, activation="relu"))

cnn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = cnn.fit(x=training_set, validation_data=test_set, epochs=50, shuffle=True)
import plotly.graph_objects as go

import numpy as np



fig = go.Figure()

fig.add_trace(

    go.Scatter(

        x=np.arange(len(history.history["accuracy"])),

        y=history.history["accuracy"],

        mode="lines",

        name="accuracy",

    )

)

fig.add_trace(

    go.Scatter(

        x=np.arange(len(history.history["val_accuracy"])),

        y=history.history["val_accuracy"],

        mode="lines",

        name="val_accuracy",

    )

)

fig.update_layout(title="Training", xaxis_title="Epochs", yaxis_title="Accuracy")

fig.show()