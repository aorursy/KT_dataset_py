# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential

import os
os.getcwd()
os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray')
img_name = 'IM-0119-0001.jpeg'
img_normal = load_img('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/' + img_name)
plt.imshow(img_normal)
plt.show()
img_name = 'person1016_bacteria_2947.jpeg'
img_infected = load_img('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/' + img_name)

plt.imshow(img_infected)
plt.show
num_train_normal = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/')
num_train_pneumonia = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/')
sns.set_style('whitegrid')
sns.barplot(x = ['NORMAL', 'PNEUMONIA'], y = [len(num_train_normal),len(num_train_pneumonia)])
infected = len(num_train_pneumonia)
normal = len(num_train_normal)
total = infected + normal
print('INFECTED = ', infected)
print('NORMAL = ', normal)
print('TOTAl = ', total)
image_height = 150 
image_width = 150
batch_size = 16
num_epoch = 10
model = Sequential()

model.add(Conv2D(64,(3,3), strides = (1,1),input_shape=(image_height,image_width, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size= (2,2)))

model.add(Conv2D(64,(3,3), strides = (1,1), activation = 'relu'))
model.add(MaxPooling2D(pool_size= (2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(64,(3,3), strides = (1,1),activation = 'relu'))
model.add(MaxPooling2D(pool_size= (2,2)))

model.add(Conv2D(64,(3,3), strides = (1,1), activation = 'relu'))
model.add(MaxPooling2D(pool_size= (2,2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 1, activation= 'sigmoid'))


model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()
train_data_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/train'
validation_data_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/val'
test_data_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/test'
train_datagen = ImageDataGenerator(rescale = 1. / 255, shear_range = 0.1)
test_datagen = ImageDataGenerator(rescale = 1. / 255)
val_datagen = ImageDataGenerator(rescale = 1./ 255)
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size =(image_width, image_height), batch_size = batch_size,
                                                    class_mode = 'binary')
test_generator = test_datagen.flow_from_directory(test_data_dir, target_size =(image_width, image_height), batch_size = batch_size,
                                                    class_mode = 'binary')
val_generator = val_datagen.flow_from_directory(validation_data_dir, target_size =(image_width, image_height), batch_size = batch_size,
                                                    class_mode = 'binary')
model.fit_generator(
train_generator,
steps_per_epoch = total// batch_size,
epochs = num_epoch,
validation_data = val_generator, 
validation_steps = 17// batch_size)