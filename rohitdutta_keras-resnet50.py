import os
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
image_shape = (224, 224)
image_size = 224
n_channels = 3
resnet = ResNet50(input_shape= (224,224,3),weights=None, include_top=True, classes=2)
data_dir = '../input/chest-xray-pneumonia/chest_xray/'

test_path = data_dir+'/test/'
valid_path = data_dir+'/val/'
train_path = data_dir+'/train/'
# Check for number of images
print('Train Set')
print('Normal : ' , len(os.listdir(train_path+'NORMAL')))
print('Pneumonia : ' , len(os.listdir(train_path+'PNEUMONIA')))

print()

print('Validation Set')
print('Normal : ' , len(os.listdir(valid_path+'NORMAL')))
print('Pneumonia : ' , len(os.listdir(valid_path+'PNEUMONIA')))

print()

print('Test Set')
print('Normal : ' , len(os.listdir(test_path+'NORMAL')))
print('Pneumonia : ' , len(os.listdir(test_path+'PNEUMONIA')))
image_gen = ImageDataGenerator(rescale = 1./255,
                               shear_range = 0.2,
                               zoom_range = 0.2,
                               horizontal_flip = True)
batch_size = 32

train_image_gen = image_gen.flow_from_directory(train_path,
                                                target_size=image_shape,
                                                color_mode='rgb',
                                                batch_size=batch_size,
                                                class_mode='categorical')
test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape,
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='categorical')
resnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
resnet.summary()
r = resnet.fit_generator(train_image_gen,
                        validation_data=test_image_gen,
                        epochs=5
                        )
import pandas as pd
history = pd.DataFrame(resnet.history.history)
history[['loss','val_loss']].plot()
