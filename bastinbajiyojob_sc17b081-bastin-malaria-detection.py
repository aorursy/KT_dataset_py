import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from matplotlib.image import imread
parasitized_sample =imread('../input/cell-images-for-detecting-malaria/cell_images/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_162.png')
uninfected_sample = imread('../input/cell-images-for-detecting-malaria/cell_images/Uninfected/C100P61ThinF_IMG_20150918_144104_cell_128.png')


plt.figure()
plt.title('Parasitized Cell')
plt.imshow(parasitized_sample)
plt.show()


plt.figure()
plt.title('Uninfected Cell')
plt.imshow(uninfected_sample)
plt.show()
parasitized_sample.shape
image_shape = (128,128,3)
datagen = ImageDataGenerator(rescale=1/255.0,validation_split=0.3)

train_dataset = datagen.flow_from_directory(directory = '../input/cell-images-for-detecting-malaria/cell_images/cell_images/', target_size=(128,128),
                                           class_mode = 'binary',
                                           batch_size = 16,
                                           subset='training')
validation_dataset = datagen.flow_from_directory(directory = '../input/cell-images-for-detecting-malaria/cell_images/cell_images/', target_size=(128,128),
                                           class_mode = 'binary',
                                           batch_size = 16,
                                           subset='validation')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())


model.add(Dense(128))
model.add(Activation('relu'))

# Dropouts help reduce overfitting by randomly turning neurons off during training.
# Here we say randomly turn off 50% of neurons.
model.add(Dropout(0.5))

# Last layer, its binary so we use sigmoid
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
early_stop = EarlyStopping(monitor='val_loss',patience=2)
model.fit_generator(train_dataset,epochs=10,validation_data=validation_dataset,callbacks=[early_stop])
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
losses[['accuracy','val_accuracy']].plot()
