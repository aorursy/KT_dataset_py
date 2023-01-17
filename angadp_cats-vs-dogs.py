import pandas as pd
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
train_data = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.15,
    fill_mode='nearest',
    width_shift_range=0.15,
    height_shift_range=0.15
)
test_data = ImageDataGenerator(
    rescale=1./255,
    fill_mode='nearest'
)

train_generator = train_data.flow_from_directory('../input/cat-and-dog/training_set/training_set', target_size=(230, 230), batch_size=32, class_mode='binary')
test_generator = test_data.flow_from_directory('../input/cat-and-dog/test_set/test_set', target_size=(230, 230), batch_size=32, class_mode='binary')
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, Flatten
model = Sequential()
model.add(Conv2D(128, (3,3), input_shape=(230,230, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3,3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3,3)))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', metrics=['accuracy'],loss='binary_crossentropy')
callback = EarlyStopping(monitor='loss', patience=6, restore_best_weights=True)
history = model.fit_generator(train_generator, 150, callbacks=[callback], validation_data=test_generator)
acc = history.history['accuracy']
loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='training accuracy')
plt.plot(epochs, val_acc, 'b', label='test accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.legend()
plt.show()