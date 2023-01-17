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
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers import Flatten
from keras.models import Sequential 
import cv2
import matplotlib.pyplot as plt
upic='../input/cell-images-for-detecting-malaria/cell_images/Uninfected/C100P61ThinF_IMG_20150918_144104_cell_131.png'
apic='../input/cell-images-for-detecting-malaria/cell_images/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_165.png'
plt.figure(1, figsize = (15 , 7))
plt.subplot(1 , 2 , 1)
plt.imshow(cv2.imread(upic))
plt.title('Uninfected Cell')
plt.xticks([]) , plt.yticks([])

plt.subplot(1 , 2 , 2)
plt.imshow(cv2.imread(apic))
plt.title('Infected Cell')
plt.xticks([]) , plt.yticks([])

plt.show()
datagen = ImageDataGenerator(rescale=1./255, validation_split = 0.2)
train_datagen = datagen.flow_from_directory('../input/cell-images-for-detecting-malaria/cell_images/cell_images', 
                                                 target_size = (128, 128), 
                                                 batch_size = 16, 
                                                 class_mode = 'binary',
                                           subset = 'training')
validate_datagen = datagen.flow_from_directory('../input/cell-images-for-detecting-malaria/cell_images/cell_images',
                                              target_size = (128,128),
                                               batch_size = 16,
                                               class_mode = 'binary',
                                               subset = 'validation')
cnn = Sequential()
cnn.add(Conv2D(input_shape = [128, 128, 3], kernel_size = 5, filters = 16, activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = 2))
cnn.add(Dropout(0.2))
cnn.add(Conv2D(kernel_size = 3, filters = 32, activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = 2))
cnn.add(Dropout(0.3))
cnn.add(Conv2D(kernel_size = 3, filters = 64, activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = 2))
cnn.add(Dropout(0.3))
cnn.add(Conv2D(kernel_size = 1, filters = 128, activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = 2))
cnn.add(Dropout(0.4))
cnn.add(Flatten())
cnn.add(Dense(units = 128, activation = 'relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(units = 1, activation = 'sigmoid'))
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
cnn.fit(x = train_datagen, validation_data = validate_datagen, epochs = 25)
scores = cnn.evaluate(validate_datagen, verbose=1)
print("Efficiency: %2.f%%" % (scores[1]*100))
cnn.summary()
