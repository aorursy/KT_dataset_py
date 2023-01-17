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
# Import libreries
from keras.layers import Dense, Flatten, Lambda, Input
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
image_size = [224,224]
vgg = VGG16(input_shape = image_size + [3], weights = 'imagenet', include_top = False)

# Dont's train the existing weight
for layer in vgg.layers:
  layer.trainable = False
folders = glob('../input/chest-xray-pneumonia/chest_xray/train/*')
x = Flatten()(vgg.output)
predictions = Dense(len(folders), activation='softmax')(x)
model = Model(inputs= vgg.input, outputs= predictions)
model.summary()

model.compile(loss = 'categorical_crossentropy',
              optimizer ='adam',
              metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range =0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range =0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/train', 
                                                 target_size = (224,224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
testing_set = test_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/test',
                                               target_size = (224,224),
                                               batch_size = 32,
                                               class_mode = 'categorical')
velidating_set = val_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/val',
                                               target_size = (224,224),
                                               batch_size = 32,
                                               class_mode = 'categorical')

# fit the model
r = model.fit_generator(
    training_set,
    validation_data = velidating_set ,
    epochs=5, 
    steps_per_epoch = len(training_set),
    validation_steps = len(velidating_set)
)
plt.plot(r.history['loss'], label = 'train_loss')
plt.plot(r.history['val_loss'], label = 'val loss')
plt.legend()
plt.show() 
print("Loss of the model is - " , model.evaluate(testing_set)[0]*100 , "%")
print("Accuracy of the model is - " , model.evaluate(testing_set)[1]*100 , "%")
