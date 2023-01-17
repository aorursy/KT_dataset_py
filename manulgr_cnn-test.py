from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import initializers
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.3)
train_data = data_gen.flow_from_directory('../input/flowers/flowers', target_size=(250,250),
                                         batch_size=32, subset='training')
test_data = data_gen.flow_from_directory('../input/flowers/flowers', target_size=(250,250),
                                         batch_size=32, subset='validation')
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(250,250,3)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(5, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])
history = model.fit_generator(train_data, epochs=10, validation_data=test_data)