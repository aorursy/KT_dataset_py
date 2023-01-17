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
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range=0.2, horizontal_flip=True)



training_set = train_datagen.flow_from_directory('../input/covid-face-mask-detection-dataset/New Masks Dataset/Train/',
                                                 target_size=(127,127),
                                                 batch_size=32,
                                                 class_mode='binary')

validation_datagen = ImageDataGenerator(rescale = 1./255)
validation_set = validation_datagen.flow_from_directory('../input/covid-face-mask-detection-dataset/New Masks Dataset/Validation/', target_size = (127,127), batch_size = 32, class_mode = 'binary')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, Flatten, MaxPool2D, Dense, BatchNormalization
cnn = Sequential()
cnn.add(Conv2D(32, activation = 'relu', kernel_size = 3, input_shape = [127,127,3]))
cnn.add(BatchNormalization())

#cnn.add(Conv2D(32, activation = 'relu', kernel_size = 3))
#cnn.add(BatchNormalization())

#cnn.add(MaxPool2D(pool_size = (2,2)))

#cnn.add(Conv2D(64, activation = 'relu', kernel_size = 3))
#cnn.add(BatchNormalization())

cnn.add(Conv2D(64, activation = 'relu', kernel_size = 3))
cnn.add(BatchNormalization())

cnn.add(MaxPool2D(pool_size = (2,2)))
cnn.add(Flatten())
cnn.add(Dense(64, activation = 'relu'))
cnn.add(Dropout(0.2))

cnn.add(Dense(128, activation = 'relu'))
cnn.add(Dropout(0.2))

cnn.add(Dense(1, activation = 'sigmoid'))

cnn.compile(optimizer='adam', metrics = 'accuracy', loss = 'binary_crossentropy')
cnn.fit(x=training_set,epochs = 50, validation_data =validation_set)
from keras.preprocessing import image
test_image = image.load_img('../input/photom/aditya-saxena-01R4fryNgUM-unsplash.jpg', target_size = (127,127,3))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
answer = cnn.predict_classes(test_image)
training_set.class_indices
if result[0][0]==1:
    prediction = 'mask'
else :
    prediction = 'unmask'
print(prediction)