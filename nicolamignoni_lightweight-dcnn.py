import os
import numpy as np
import pandas as pd
import csv

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

TRAIN_PERC = 0.8
DATA_LENGHT = 42000
VAL_LENGHT = DATA_LENGHT*(1-TRAIN_PERC)
TEST_LENGHT = 28000
BATCH_SIZE = 600
data = pd.read_csv('../input/digit-recognizer/train.csv', dtype='float32')
test = pd.read_csv('../input/digit-recognizer/test.csv', dtype='float32')

data_input = data.drop(columns=['label']).values.reshape(DATA_LENGHT,28,28,1)/255.0
data_labels = data['label'].values

test_input = test.values.reshape(TEST_LENGHT,28,28,1)/255.0

x_train, x_val, y_train, y_val = train_test_split(data_input, data_labels, train_size=TRAIN_PERC, 
                                                  random_state=35261) 
generator = ImageDataGenerator()
generator.fit(x_train)
model = Sequential()

model.add(Conv2D(64, (3,3), padding='same', input_shape=(28,28,1)))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
BatchNormalization()
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
BatchNormalization()
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
BatchNormalization()
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
BatchNormalization()
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()
cp = ModelCheckpoint('digit_model_dcti.{epoch:02d}-{accuracy:.2f}.h5', monitor='val_accuracy',
                     save_best_only=True, verbose=1, mode='max')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(generator=generator.flow(x_train, y_train, BATCH_SIZE), epochs=10, 
                    validation_data=(x_val, y_val), validation_steps=int(VAL_LENGHT/BATCH_SIZE), 
                    callbacks=[cp])
predictions = model.predict_classes(test_input)
submission = np.array([range(1, TEST_LENGHT+1), predictions], np.int16).T
np.savetxt('submission.csv', submission, fmt='%d', delimiter=',', header='ImageId,Label', comments='', newline='\r\n')