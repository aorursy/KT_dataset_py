import os
import pandas as pd
 
from tensorflow.keras import layers
from tensorflow.keras import Model
 
from tensorflow.keras.applications.inception_v3 import InceptionV3
 
 
from tensorflow.keras.models import Model,Sequential
 
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.optimizers import RMSprop , Adam , Adamax , Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
local_weights_file = "../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
pretrained_model = InceptionV3(input_shape=(224,224, 3), 
                              include_top=False,
                              weights = None)
pretrained_model.load_weights(local_weights_file)
for layer in pretrained_model.layers:
    layer.trainable = False
pretrained_model.summary()
last_layer = pretrained_model.get_layer('mixed7')
last_output = last_layer.output
x = layers.Flatten()(last_output)
x = layers.Dropout(0.5)(x)
#x = layers.BatchNormalization()(x)
x  = layers.Dense(512 , activation='relu')(x)
x = layers.Dropout(0.5)(x)
 
#x = layers.BatchNormalization()(x)
   
x = layers.Dense(2, activation='softmax')(x)
model = Model(pretrained_model.input, x)
model.summary()
model.compile(optimizer = Adam(lr=0.0001),
             loss='categorical_crossentropy',
             metrics=['accuracy'])
train_dir = "../input/chest-xray-pneumonia/chest_xray/train"
valid_dir = "../input/chest-xray-pneumonia/chest_xray/val"
train_datagen = ImageDataGenerator(
         
        rescale=1./255,
        featurewise_std_normalization = True , 
        featurewise_center=True,
         
        fill_mode = 'nearest',  
        rotation_range=40,  
        zoom_range = 0.4,  
        shear_range = 0.4,
         
        
        horizontal_flip=0.5 
         
        
       
        )
valid_datagen = ImageDataGenerator(
         
        rescale=1./255,
        featurewise_std_normalization = True , 
        featurewise_center=True,
         
        fill_mode = 'nearest',  
        rotation_range=40,  
        zoom_range = 0.4,  
        shear_range = 0.4,
         
        
        horizontal_flip=0.5 
         
        
       
        )
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=128,
                                                   target_size=(224, 224),
                                                    shuffle = True, 
                                                    
                                                   class_mode='categorical' 
                                                    )
valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                                    batch_size=128,
                                                   target_size=(224, 224),
                                                    shuffle = True, 
                                                    
                                                   class_mode='categorical' 
                                                    )
history = model.fit(train_generator,
                   epochs=10,
                    validation_data = valid_generator , 
                    batch_size = 128
                  
                   
                  )
labels = {
    0 : "Normal" , 
    1 : "Pneumonia"
}
model.save("Inception_model.h5")
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
path = "../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1000_virus_1681.jpeg"

img = image.load_img(path, target_size=(224, 224))
x = image.img_to_array(img)/255
x = np.expand_dims(x, axis=0)
classes = model.predict([x])
classes
pred_labels = np.argmax(classes, axis = 1)
pred_labels
for i in pred_labels:
    print(labels[i])
