# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import keras
import numpy as np
import pandas as pd
import sys
import os
%matplotlib inline
train_dir = '/kaggle/input/10-monkey-species/training/training/'
validation_dir = '/kaggle/input/10-monkey-species/validation/validation/'
valid_dir = '/kaggle/input/10-monkey-species/validation/validation/'
label_files = '/kaggle/input/10-monkey-species/monkey_labels.txt'
labels = pd.read_csv(label_files, header=0)
labels.head(10)
height = 128
width = 128
channals = 3
batch_sizes = 64
num_classes = 10
train_data_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255, 
                                                                    rotation_range=40, 
                                                                    width_shift_range=0.2, 
                                                                    height_shift_range=0.2,
                                                                    shear_range=0.2,
                                                                    zoom_range=0.2,
                                                                    horizontal_flip=True,
                                                                    fill_mode='nearest')
train_generator = train_data_generator.flow_from_directory(train_dir, target_size = (height, width), batch_size = batch_sizes, seed = 7, shuffle = True, class_mode = 'categorical')
validation_data_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255 )
validation_generator = validation_data_generator.flow_from_directory(validation_dir, target_size = (height, width), batch_size = batch_sizes,
                                                                    seed = 7, class_mode = 'categorical', shuffle = False)
train_num = train_generator.samples
validation_num = validation_generator.samples
model  = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters = 32, kernel_size=3, padding='same', activation='relu', input_shape = [height, width,channals]))
model.add(keras.layers.Conv2D(filters = 32, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size = 2))

model.add(keras.layers.Conv2D(filters = 64, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters = 64, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size = 2))

model.add(keras.layers.Conv2D(filters = 128, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters = 128, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size = 2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
epoch = 100
history = model.fit_generator(train_generator, 
                              steps_per_epoch=train_num//batch_sizes, 
                              epochs=epoch, 
                              validation_data=validation_generator,
                              validation_steps=valid_num//batch_sizes)
def plot_learning_curve(history, label, minIndex, MaxIndex):
    data = {}
    data[label] = history.history[label]
    data['val_' + label] = history.history['val_'+label]
    pd.DataFrame(data).plot(figsize=(10, 5))
    plt.ylim(minIndex, MaxIndex)
    plt.show()
    
plot_learning_curve(history, 'accuracy', 0, 1)
plot_learning_curve(history, 'loss', 0., 2.5)
height = 224
width = 224
channals = 3
batch_sizes = 24
num_classes = 10
train_data_generator = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=keras.applications.resnet50.preprocess_input,
                                                                    rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                                                    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
train_generator = train_data_generator.flow_from_directory(train_dir, target_size=(height, width), batch_size = batch_sizes, seed = 7,shuffle = True, class_mode='categorical')
validation_data_generator = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=keras.applications.resnet50.preprocess_input)
validation_generator = validation_data_generator.flow_from_directory(validation_dir, target_size=(height, width), batch_size=batch_sizes,shuffle = False, seed = 7, class_mode='categorical')
train_num = train_generator.samples
valid_num  = valid_generator.samples
print(train_num, valid_num)
resnet50_fine_tune = keras.models.Sequential()
resnet50_fine_tune.add(keras.applications.ResNet50(include_top = False,
                                                  pooling = 'avg',
                                                  weights = 'imagenet'))
resnet50_fine_tune.add(keras.layers.Dense(num_classes, activation = 'softmax'))
resnet50_fine_tune.layers[0].trainable = False

resnet50_fine_tune.compile(loss="categorical_crossentropy",
             optimizer="sgd", metrics=['accuracy'])
resnet50_fine_tune.summary()
epochs = 50
batch_size = 24
history = resnet50_fine_tune.fit_generator(train_generator,
                                           steps_per_epoch = train_num // batch_size,
                                           epochs = epochs,
                                           validation_data = validation_generator,
                                           validation_steps=valid_num // batch_size )
plot_learning_curve(history, 'accuracy', 0, 1)
plot_learning_curve(history, 'loss', 0., 2)


