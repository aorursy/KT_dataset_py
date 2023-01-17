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

        file = (os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import tensorflow

from tensorflow.keras.layers import Input, Dense, Flatten

from keras import Model

from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing import image

from keras.models import Sequential
image_size = [224, 224]
incep_v3 = InceptionV3(input_shape = image_size + [3], weights = 'imagenet', include_top = False)
for layer in incep_v3.layers:

    layer.trainable = False
from glob import glob

folders = glob('../input/cotton-disease-prediction/train/*')
folders
x = Flatten()(incep_v3.output)
prediction = Dense(len(folders), activation = 'softmax')(x)
model = Model(inputs = incep_v3.input, outputs = prediction)
model.summary()
model.compile( loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('../input/cotton-disease-prediction/train',

                                                 target_size = (224, 224),

                                                 batch_size = 32,

                                                 class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('../input/cotton-disease-prediction/val',

                                            target_size = (224, 224),

                                            batch_size = 32,

                                            class_mode = 'categorical')
device_name = tensorflow.test.gpu_device_name()

if "GPU" not in device_name:

    print("GPU device not found")

print('Found GPU at: {}'.format(device_name))
r = model.fit_generator(

  training_set,

  validation_data=test_set,

  epochs=20,

  steps_per_epoch=len(training_set),

  validation_steps=len(test_set)

)
import matplotlib.pyplot as plt

plt.plot(r.history['loss'], label='train loss')

plt.plot(r.history['val_loss'], label='val loss')

plt.legend()

plt.show()

plt.savefig('LossVal_loss')



# plot the accuracy

plt.plot(r.history['accuracy'], label='train acc')

plt.plot(r.history['val_accuracy'], label='val acc')

plt.legend()

plt.show()

plt.savefig('AccVal_acc')