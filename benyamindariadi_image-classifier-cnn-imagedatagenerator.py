import os

#Dataset file should has a directory structure like this

print(os.listdir('/kaggle/input/lego-v1-train-test/LEGO brick images v1'))



print(os.listdir('/kaggle/input/lego-v1-train-test/LEGO brick images v1/train'))

print(os.listdir('/kaggle/input/lego-v1-train-test/LEGO brick images v1/test'))
data_dir= r'/kaggle/input/lego-v1-train-test/LEGO brick images v1/' 
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.image import imread

test_path=data_dir+'/test/'

train_path=data_dir+'/train/'
os.listdir(train_path)

os.listdir(train_path+'11214 Bush 3M friction with Cross axle')

len(os.listdir(train_path+'11214 Bush 3M friction with Cross axle'))
os.listdir(train_path+'11214 Bush 3M friction with Cross axle')[0]
Example_one_image=train_path+'11214 Bush 3M friction with Cross axle'+'/201706171006-0002.png' 
imread(Example_one_image).shape
plt.imshow(imread(Example_one_image))
image_shape= (100,100,1) #in order to make computation faster, reduce the size of images
from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(rescale=1/255,rotation_range=90,width_shift_range=0.05,height_shift_range=0.05,shear_range=0.1,zoom_range=0.1,horizontal_flip=True,fill_mode='nearest') 

plt.imshow(imread(Example_one_image))
plt.imshow(image_gen.random_transform(imread(Example_one_image))) 
image_gen.flow_from_directory(train_path)

image_gen.flow_from_directory(test_path)
train_image_gen = image_gen.flow_from_directory(train_path,target_size=(100,100),color_mode='grayscale',batch_size=16,class_mode='categorical')         



test_image_gen = image_gen.flow_from_directory(test_path,

                                               target_size=(100,100),

                                               color_mode='grayscale',

                                               batch_size=16,

                                               class_mode='categorical',

                                               shuffle=False)
train_image_gen.class_indices 
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
model = Sequential()



model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=image_shape, activation='relu',))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(filters=64, kernel_size=(4,4),input_shape=image_shape, activation='relu',))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(filters=64, kernel_size=(4,4),input_shape=image_shape, activation='relu',))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())



model.add(Dense(256))   

model.add(Activation('relu'))

model.add(Dropout(0.5))





model.add(Dense(16))

model.add(Activation('softmax'))      





model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss',patience=5)
model.fit_generator(train_image_gen,epochs=25, validation_data=test_image_gen, callbacks=[early_stop])
model.metrics_names

model.evaluate_generator(test_image_gen) 
losses = pd.DataFrame(model.history.history)

losses[['loss','val_loss']].plot()

losses[['accuracy','val_accuracy']].plot()
pred_probabilities = model.predict_generator(test_image_gen) 
predictions = pred_probabilities > 0.5

predictions