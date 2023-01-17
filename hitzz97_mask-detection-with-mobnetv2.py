import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import (
    Conv2D,Flatten,LeakyReLU as Leaky,
    BatchNormalization,Dropout,
    MaxPooling2D,Dense
)
datagen=ImageDataGenerator(rescale=1/255.)
#will make custom data generator to support image resizing

#def gen(target_size=(224,224),path="",batch_size=32,class_mode='binary'):
#    assert path!="" 
#    folders=os.listdir(path)
#    total_imgs=0
#    for f in folders:
#        total_imgs+=len(
train=datagen.flow_from_directory('/kaggle/input/face-mask-12k-images-dataset/Face Mask Dataset/Train',class_mode='binary')
test=datagen.flow_from_directory('/kaggle/input/face-mask-12k-images-dataset/Face Mask Dataset/Test',class_mode='binary')
train.class_indices
train.batch_size
keras.backend.clear_session()

model=keras.models.Sequential([
    Conv2D(input_shape=(256,256,3),filters=32,kernel_size=(3,3),padding="Same"),
    Leaky(0.01),
    Conv2D(32,kernel_size=(3,3),padding="same"),
    Leaky(0.01),
    MaxPooling2D(strides=(2,2)),
    BatchNormalization(),
    
    Conv2D(64,(3,3),padding="same"),
    Leaky(0.01),
    Conv2D(64,(3,3),padding="same"),
    Leaky(0.01),
    MaxPooling2D (strides=(2,2)),
    #Dropout(0.1),
    #BatchNormalization (),
    
    Conv2D(128,(3,3),padding="same"),
    Leaky(0.01),
    Conv2D(128,(3,3),padding="same"),
    Leaky(0.01),
    MaxPooling2D(strides=(2,2)),
    #Dropout(0.2),
    #BatchNormalization (),
    
    Conv2D(256,(3,3),padding="same"),
    Leaky(0.01),
    Conv2D(256,(3,3),padding="same"),
    Leaky(0.01),
    MaxPooling2D(strides=(2,2)),
    Dropout(0.2),
    #BatchNormalization (),
    
  
    Flatten(),
    
    Dense(256,kernel_regularizer=keras.regularizers.l2(0.01)),
    Leaky(0.01),
    Dropout (0.3),
    
    Dense(1,activation="sigmoid"),
])
model.summary()
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy',])
model.fit(train,epochs=10,validation_data=(test),shuffle=True)
# 17M Parameters Approx 250Mb size on saving 
model.save('/kaggle/working/mask_detection_model.h5') 
import pandas as pd
loss=pd.DataFrame(model.history.history)
loss.plot()
from pathlib import Path
Path('/kaggle/working/mask_detection_model_2.h5' ).stat()
#Decreasing the model shape vertically by 2

keras.backend.clear_session()

model=keras.models.Sequential([
    Conv2D(input_shape=(256,256,3),filters=16,kernel_size=(3,3),padding="Same"),
    Leaky(0.01),
    Conv2D(16,kernel_size=(3,3),padding="same"),
    Leaky(0.01),
    MaxPooling2D(strides=(2,2)),
    BatchNormalization(),
    
    Conv2D(32,(3,3),padding="same"),
    Leaky(0.01),
    Conv2D(32,(3,3),padding="same"),
    Leaky(0.01),
    MaxPooling2D (strides=(2,2)),
    #Dropout(0.1),
    #BatchNormalization (),
    
    Conv2D(64,(3,3),padding="same"),
    Leaky(0.01),
    Conv2D(64,(3,3),padding="same"),
    Leaky(0.01),
    MaxPooling2D(strides=(2,2)),
    #Dropout(0.2),
    #BatchNormalization (),
    
    Conv2D(128,(3,3),padding="same"),
    Leaky(0.01),
    Conv2D(128,(3,3),padding="same"),
    Leaky(0.01),
    MaxPooling2D(strides=(2,2)),
    Dropout(0.2),
    #BatchNormalization (),
    
  
    Flatten(),
    
    Dense(128,kernel_regularizer=keras.regularizers.l2(0.01)),
    Leaky(0.01),
    Dropout (0.3),
    
    Dense(1,activation="sigmoid"),
])
model.summary()
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy',])
model.fit(train,epochs=10,validation_data=(test),shuffle=True)
# 4M parameters Approx 51Mb size on saving model
model.save('/kaggle/working/mask_detection_model_2.h5')
import pandas as pd
loss=pd.DataFrame(model.history.history)
loss.plot()