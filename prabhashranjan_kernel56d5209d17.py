%reset -f

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import time
from PIL import Image 

img_width, img_height = 150, 150
from zipfile import ZipFile

train_data_dir = "..//input//chest-xray-pneumonia//chest_xray//train"

nb_train_samples = 40
validation_data_dir = "..//input//chest-xray-pneumonia//chest_xray//val"
nb_validation_samples = 16
batch_size = 32
epochs = 6

test_generator_samples = 40

test_batch_size = 64
K.image_data_format()
K.backend()
model = Sequential()

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:                                           # So, Tensorflow!
    input_shape = (img_width, img_height, 3)
model.add(Conv2D(
	             filters=24,                      
	                                               
	             kernel_size=(3, 3),               
	             strides = (1,1),                  
	                                               
	             input_shape=input_shape,          
	             use_bias=True,                     
	             padding='same',
	             name="Ist_conv_layer",
	             )
         )
model.summary()
model.add(Activation('sigmoid'))

model.summary()
model.add(Conv2D(
	             filters=16,                       
	                                               
	             kernel_size=(3, 3),               
	             strides = (1,1),                  
	                                               
	             use_bias=True,                     
	             padding='same',                   
	             name="2nd_conv_layer"
	             )
         )
model.summary()
model.add(Activation('tanh'))           
model.summary()
model.add(MaxPool2D())

model.summary()

model.add(Flatten())

model.summary()
model.add(Dense(64))

model.add(Activation('relu'))    

model.summary()
model.add(Dense(32))

model.add(Activation('relu'))    
model.summary()
model.add(Dense(1))

model.add(Activation('sigmoid'))   

model.summary()
model.compile(
              loss='binary_crossentropy',  
              optimizer='rmsprop',         
              metrics=['accuracy'])     
def preprocess(img):
                    return img
tr_dtgen = ImageDataGenerator(
                              rescale=1. / 255,      
                              shear_range=0.2,       
                              zoom_range=0.2,
                              horizontal_flip=True,
                              preprocessing_function=preprocess
                              )
train_generator = tr_dtgen.flow_from_directory(
                                               train_data_dir,       
                                               target_size=(img_width, img_height),  
                                               batch_size=batch_size,  
                                               class_mode='binary'   

                                                )

val_dtgen = ImageDataGenerator(rescale=1. / 255)

validation_generator = val_dtgen.flow_from_directory(
                                                     validation_data_dir,
                                                     target_size=(img_width, img_height),   # Resize images
                                                     batch_size=batch_size,    # batch size to augment at a time
                                                     class_mode='binary'  # Return 1D array of class labels
                                                     )



start = time.time()   
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in train_generator:
        model.fit(x_batch, y_batch)
        batches += 1
        print ("Epoch: {0} , Batches: {1}".format(e,batches))
        if batches > 210:    
            
            break

end = time.time()
(end - start)/60

result = model.evaluate(validation_generator,
                                  verbose = 1,
                                  steps = 4        
                                  )
result  
pred = model.predict(validation_generator, steps = 2)

pred
test_data_dir =   "..//input//chest-xray-pneumonia//chest_xray//test"

test_datagen = ImageDataGenerator(rescale=1. / 255)



test_generator = test_datagen.flow_from_directory(
        test_data_dir,                         
        target_size=(img_width, img_height),   
        batch_size = test_batch_size,            
        class_mode=None)                       


im = test_generator    

images = next(im)   


images.shape    

results = model.predict(images)

results             
#Plot the images and check with
#     results
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

plt.figure(figsize= (10,10))

for i in range(results.shape[0]):
    plt.subplot(5,5,i+1)
    imshow(images[i])
    
    plt.show()

