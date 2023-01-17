import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
# Technically not necessary in newest versions of jupyter
%matplotlib inline
my_data_dir='../input/fruits/fruits-360'
os.listdir(my_data_dir)
test_path = my_data_dir+'/Test/'
train_path = my_data_dir+'/Training/'
os.listdir(train_path+'/Apple Braeburn/')[0]
apple_braeburn_path = train_path+'/Apple Braeburn'+'/r_236_100.jpg'
apple_braeburn_img= imread(apple_braeburn_path)
plt.imshow(apple_braeburn_img)
apple_braeburn_img.shape
watermelon_path = train_path+'/Watermelon/'+os.listdir(train_path+'/Watermelon/')[0]
watermelon_img = imread(watermelon_path)
plt.imshow(watermelon_img)
len(os.listdir(train_path+'/Watermelon'))
watermelon_img.shape
dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/Watermelon'):
    
    img = imread(test_path+'/Watermelon'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)
sns.jointplot(dim1,dim2)
np.mean(dim1)
np.mean(dim2)
image_shape = (100,100,3)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )

plt.imshow(watermelon_img)
plt.imshow(image_gen.random_transform(watermelon_img))
plt.imshow(image_gen.random_transform(watermelon_img))
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

# Dropouts help reduce overfitting by randomly turning neurons off during training.
# Here we say randomly turn off 50% of neurons.
model.add(Dropout(0.5))

model.add(Dense(131, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.summary()
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=0)
batch_size = 16
train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='categorical')
test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='categorical',shuffle=False)
train_image_gen.class_indices
import warnings
warnings.filterwarnings('ignore')


results = model.fit_generator(train_image_gen,epochs=2,
                              validation_data=test_image_gen,
                              callbacks=[early_stop])

