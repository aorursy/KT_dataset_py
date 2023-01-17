 
import os
import pandas as pd
 
from tensorflow.keras import layers
from tensorflow.keras import Model
 
from tensorflow.keras.applications.vgg16 import VGG16
 
 
from tensorflow.keras.models import Model,Sequential
 
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.optimizers import RMSprop , Adam , Adamax , Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
    
pretrained_model = VGG16(input_shape=(224,224, 3), 
                              include_top=False,
                              weights = 'imagenet')
for layer in pretrained_model.layers:
    layer.trainable = False
pretrained_model.summary()

x = layers.Flatten()(pretrained_model.output)
 
x  = layers.Dense(4096 , activation='relu')(x)
x  = layers.Dense(4096 , activation='relu')(x)
 
   
x = layers.Dense(2, activation='softmax')(x)
model = Model(pretrained_model.input, x)
model.summary()
model.compile(optimizer = Adam(lr=0.0001),
             loss='categorical_crossentropy',
             metrics=['accuracy'])
train_dir = "../input/skin-cancer-malignant-vs-benign/train"
train_datagen = ImageDataGenerator(
         
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
history = model.fit(train_generator,
                   epochs=10,
                     
                    batch_size = 128
                  
                   
                  )
model.save("Skin_cancer_vgg16.h5")
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
labels = {
    0 : "benign" , 
    1 : "Malignant"
}
path = r"../input/skin-cancer-malignant-vs-benign/test/malignant/1022.jpg"
img = image.load_img(path, target_size=(224, 224))
x = image.img_to_array(img)/255
x = np.expand_dims(x, axis=0)
classes = model.predict([x])
pred_labels = np.argmax(classes, axis = 1)
pred_labels
for i in pred_labels:
    print(labels[i])