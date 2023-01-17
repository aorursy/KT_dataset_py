# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk("/kaggle/input/pokemon-images-dataset/pokemon/pokemon"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, Conv2DTranspose, MaxPooling2D, UpSampling2D
from tensorflow.keras import Input, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import numpy as np

def add_noise(img):
    VARIABILITY = 10
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    return np.clip(img, 0.0, 255.0)

data_generator = ImageDataGenerator(
    fill_mode="constant",
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=[-80,80],
    height_shift_range=[-80,80],
    validation_split=0.2,
    brightness_range=[0.9,1.0],
    rotation_range=40,
    zoom_range=0.1,
    shear_range=0.2,
    data_format="channels_last",
    rescale=1.0/255.0,
)

DIRECTORY = "/kaggle/input/pokemon-images-dataset/pokemon_jpg/"
train_generator = data_generator.flow_from_directory(
    DIRECTORY, 
    subset="training",
    batch_size=8,
    class_mode="input",
    shuffle=True,
    target_size=(256, 256),
)
validation_generator = data_generator.flow_from_directory(
    DIRECTORY, 
    subset="validation",
    batch_size=8,
    class_mode="input",
    shuffle=True,
    target_size=(256, 256),
)

encoder_input = Input(shape=(256,256,3))
x = Conv2D(48, (3, 3), activation="relu", padding="same")(encoder_input)
x = MaxPooling2D((2, 2), padding="same")(x)
x = Conv2D(96, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), padding="same")(x)
x = Conv2D(192, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), padding="same")(x)
encoded = Conv2D(32, (1, 1), activation="relu", padding="same")(x)

latent_size = (32,32,32)

decoder_input = Input(shape=latent_size)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(decoder_input)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

encoder = Model(encoder_input, encoded)
decoder = Model(decoder_input, decoded)
autoencoder = Model(encoder_input, decoder(encoded))
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, verbose=2, mode="auto", restore_best_weights=True
)

autoencoder.fit(train_generator, validation_data=validation_generator, epochs=1000, verbose=1, callbacks=[early_stopping])
