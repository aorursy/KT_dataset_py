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
import numpy as np 
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
example_pic1 = cv2.imread('../input/fer2013/train/neutral/Training_10002154.jpg')
example_pic1 = cv2.cvtColor(example_pic1,cv2.COLOR_BGR2RGB)
plt.imshow(example_pic1)
example_pic1.shape
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rotation_range=30, 
                               width_shift_range=0.1, 
                               height_shift_range=0.1, 
                               rescale=1./255, 
                               shear_range=0.2, 
                               zoom_range=0.2, 
                               horizontal_flip=True, 
                               fill_mode='nearest' 
                              )
plt.imshow(train_datagen.random_transform(example_pic1))

training_set = train_datagen.flow_from_directory('../input/fer2013/train',
                                                 target_size = (48, 48),                                   
                                                 batch_size = 64,
                                                 color_mode='grayscale',
                                                 class_mode = 'categorical')
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('../input/fer2013/test',
                                            target_size = (48, 48),                                            
                                            batch_size = 64,
                                            color_mode='grayscale',
                                            class_mode = 'categorical')
