# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator
model = Sequential()

model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(640, 480, 3)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (5, 5), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
generator = ImageDataGenerator(rescale=1 / 255)

train = generator.flow_from_directory(

    "/kaggle/input/collapsiblesmulticolor/img/train",

    classes=["clean", "dirty"],

    class_mode="binary",

    target_size=(640, 480),

)

validation = generator.flow_from_directory(

    "/kaggle/input/collapsiblesmulticolor/img/validation",

    classes=["clean", "dirty"],

    class_mode="binary",

    target_size=(640, 480)

)

test = generator.flow_from_directory(

    "/kaggle/input/collapsiblesmulticolor/img/test",

    classes=["clean", "dirty"],

    class_mode="binary",

    target_size=(640, 480)

)
callbacks = [keras.callbacks.EarlyStopping(patience=3, min_delta=0.01)]

history = model.fit_generator(train, epochs=40, validation_data=validation, callbacks=callbacks)
history.history
import matplotlib.pyplot as plt
plt.plot(history.history["loss"])

plt.plot(history.history["val_loss"])

plt.show()
test_score = model.evaluate_generator(test)
test_score
model.save_weights('dirt_measure_2.model')