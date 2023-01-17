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
   
x = layers.Dense(10, activation='softmax')(x)
model = Model(pretrained_model.input, x)
model.summary()
model.compile(optimizer = Adam(lr=0.0001),
             loss='categorical_crossentropy',
             metrics=['accuracy'])
train_dir = "../input/10-monkey-species/training/training"
valid_dir = "../input/10-monkey-species/validation/validation"
train_datagen = ImageDataGenerator(
         
        rescale=1./255,
        
         
        fill_mode = 'nearest',  
        rotation_range=40,  
         
        horizontal_flip=0.5 
         
        
       
        )
valid_datagen = ImageDataGenerator(
         
        rescale=1./255,
         
        fill_mode = 'nearest',  
        rotation_range=40,  
         
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
model.save("Monkey_species.h5")
path = "../input/new-monkey/Alouatta_palliata_5_CR.jfif"

img = image.load_img(path , target_size = (224,224))
img = image.img_to_array(img)
img = np.expand_dims(img , axis=0)
c = model.predict([img])
pred_labels = np.argmax(c, axis = 1)
    
pred_labels


labels = {
    0 : "alouattapalliata" , 
    1 : "erythrocebuspatas" , 
    2 : "cacajaocalvus" , 
    3 : "macacafuscata" , 
    4 : "cebuellapygmea" , 
    5 : "cebuscapucinus" , 
    6 : "micoargentatus" , 
    7 : "saimirisciureus" , 
    8 : "aotusnigriceps" , 
    9 : "trachypithecusjohnii"
}
labesl[pred_labels[0]]