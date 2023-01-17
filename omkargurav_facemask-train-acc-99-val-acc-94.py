import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

from tensorflow.keras.preprocessing import image_dataset_from_directory

from tensorflow.keras.utils import plot_model



import numpy as np

import matplotlib.pyplot as plt
data_dir = '/kaggle/input/face-mask-dataset/data'

image_size = (224, 224)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,

                                                              validation_split = 0.2,

                                                              subset = "training",

                                                              seed = 42,

                                                              image_size = image_size,

                                                               shuffle = True,

                                                              batch_size = 40)



test_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,

                                                             validation_split = 0.2,

                                                             subset = "validation",

                                                             seed = 42,

                                                             image_size = image_size,

                                                              shuffle = True,

                                                             batch_size = 40)
## Configuring dataset for performance

AUTOTUNE = tf.data.experimental.AUTOTUNE

training_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

testing_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
model = tf.keras.models.Sequential([

    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),

    Conv2D(32,(3,3),activation="relu",padding="same",input_shape=(224,224,3)),

    MaxPooling2D(),

    

    Conv2D(64,(3,3),padding="same",activation="relu"),

    MaxPooling2D(),

    

    Conv2D(128,(3,3),padding="same",activation="relu"),

    MaxPooling2D(),

    Flatten(),

       

    Dense(256, activation='relu'),

    Dense(2, activation= 'softmax')

])

model._name = "FaceMaskNet"

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

plot_model(model,show_shapes=True)
history = model.fit(training_ds,

                    validation_data = testing_ds,

                    epochs=10)

model.summary()

plot_model(model,show_shapes=True)
def plot_graph(history,string):

    plt.figure(figsize=(12,8))

    plt.plot(history.history[string],label=str(string))

    plt.plot(history.history["val_"+str(string)],label="val_"+str(string))

    plt.xlabel("Epochs")

    plt.ylabel(str(string))

    plt.show()

    

plot_graph(history,"accuracy")

plot_graph(history,"loss")
#model.save("FaceMaskNet.h5")
from IPython.display import FileLink

#FileLink('FaceMaskNet.h5')