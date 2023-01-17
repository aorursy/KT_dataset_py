# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (200, 200, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dropout(rate = 0.2))
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dropout(rate = 0.2))
classifier.add(Dense(units = 2, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.summary()
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('../input/chest_xray/chest_xray/train',
                                                 target_size = (200, 200),
                                                 batch_size = 256,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('../input/chest_xray/chest_xray/test',
                                            target_size = (200, 200),
                                            batch_size = 256,
                                            class_mode = 'categorical')
train_steps = training_set.samples // 256
val_steps = test_set.samples // 256

from keras.callbacks import ModelCheckpoint
file_name="classifier.hdf5"
checkpoint = ModelCheckpoint(file_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

classifier.fit_generator(training_set,
                         steps_per_epoch = train_steps,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = val_steps,
                        callbacks=[checkpoint])
from keras.models import load_model
classifier = load_model('classifier.hdf5')
from sklearn.metrics import classification_report, confusion_matrix

y_pred = classifier.predict_generator(test_set, val_steps+1)
y_pred = np.argmax(y_pred, axis=1)

print('Confusion Matrix')
print(confusion_matrix(test_set.classes, y_pred))

print('Classification Report')
target_names = ['NORMAL', 'PNEUMONIA']
print(classification_report(test_set.classes, y_pred, target_names=target_names))
from mlxtend.plotting import plot_confusion_matrix

plot = confusion_matrix(test_set.classes, y_pred)
plot_confusion_matrix(conf_mat=plot ,  figsize=(6, 6))