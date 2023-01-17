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
        
        os.path.join(dirname, filename)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, AveragePooling2D,BatchNormalization
import os
num_classes = 81
img_rows, img_cols = 100, 100
batch_size = 16
train_data_dir = '/kaggle/input/fruits/fruits-360/Training/'
validation_data_dir = '/kaggle/input/fruits/fruits-360/Test/'



train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(100, 100),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(100, 100),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
model = Sequential()

model.add(Conv2D(64, (3, 3),input_shape= (100, 100, 3)))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))

model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))




model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))

model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))


model.add(MaxPool2D(pool_size=(2, 2)))


model.add(Conv2D(256, (3, 3)))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))

model.add(Conv2D(256, (3, 3)))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))

model.add(MaxPool2D(pool_size=(3, 3)))




model.add(Flatten())

model.add(Dense(600)) #900
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(300)) #400
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(150)) #250
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(131,activation="softmax"))

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 6,
                          restore_best_weights = True)

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                              factor = 0.2,
                              patience = 3,
                              verbose = 1,
                              min_delta = 0.0001)

callbacks = [earlystop, reduce_lr]

model.compile(loss = 'categorical_crossentropy',
              optimizer = "adam",
              metrics = ['accuracy'])


print(model.summary())

epochs = 50
history = model.fit_generator(
    train_generator,
    use_multiprocessing=True,
    workers=16,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator
    )
metrics = pd.DataFrame(history.history)
metrics[["loss","val_loss"]].plot()
metrics[["accuracy","val_accuracy"]].plot()
model.evaluate(validation_generator,verbose=0)
