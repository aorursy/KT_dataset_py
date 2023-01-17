from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
IMAGE_SIZE=[224,224]

train_path='/kaggle/input/dogs-cats-images/dataset/training_set'
train_path='/kaggle/input/dogs-cats-images/dataset/test_set'

vgg=VGG16(input_shape=IMAGE_SIZE + [3],weights='imagenet',include_top=False)

for layer in vgg.layers:
    layer.trainable=False
    
folders=glob('/kaggle/input/dogs-cats-images/dataset/training_set/*')
    
x=Flatten()(vgg.output)
print(x)

prediction=Dense(len(folders),activation='softmax')(x)

for layer in vgg.layers:
    layer.trainable = False

model = Model(inputs=vgg.input, outputs=prediction)


model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory('/kaggle/input/dogs-cats-images/dataset/training_set',target_size=(224,224),batch_size=32,class_mode='categorical')
test_set=test_datagen.flow_from_directory('/kaggle/input/dogs-cats-images/dataset/test_set',target_size=(224,224),batch_size=32,class_mode='categorical')
r=model.fit_generator(training_set,validation_data=test_set,epochs=5,steps_per_epoch=len(training_set),validation_steps=len(test_set))


model.save('new11.h5')
