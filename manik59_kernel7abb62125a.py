from keras.layers import Input,Lambda,Dense,Flatten
from keras.models import Model,Sequential,load_model
from keras.applications.vgg16 import VGG16 ,preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import numpy as np
import pandas as pd
import tensorflow as tf


Image_size=[224,224]

train_path='../input/indian-currency-notes/indian_currency_new/training'
test_path='../input/indian-currency-notes/indian_currency_new/validation'

vgg = VGG16(input_shape=Image_size+[3],weights='imagenet',include_top=False)

for layer in vgg.layers:
    layer.trainable=False

folders=glob('../input/indian-currency-notes/indian_currency_new/training/*')

x=Flatten()(vgg.output)

first=Dense(2048,activation='relu')(x)
second=Dense(256,activation='relu')(first)
prediction = Dense(8,activation='softmax')(second)


model=Model(inputs=vgg.input,outputs=prediction)

model.summary()

model.compile(
loss='categorical_crossentropy',
optimizer='adam',
metrics=['accuracy']
)

train_datagen=ImageDataGenerator(rescale=1./255,
                                shear_range=.2,
                                zoom_range=.1,
                                 horizontal_flip=True
                                )
test_datagen=ImageDataGenerator(rescale=1./255)
train_set=train_datagen.flow_from_directory(train_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

test_set=test_datagen.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
model1=model.fit_generator(train_set,validation_data=test_set,shuffle=True,epochs=20,steps_per_epoch=len(train_set),
  validation_steps=len(test_set))
model.save('./Indian_currency_recog.h5')
model=load_model('./Indian_currency_recog.h5')

from keras.preprocessing.image import load_img


def preprocess_img(img):
    
    img=image.load_img(img,target_size=(224,224))
    
    img=image.img_to_array(img)
    
    img=np.expand_dims(img,axis=0)
    
    # normalisation
    
    img=preprocess_input(img)
    
    return img

img=preprocess_img("../input/indian-currency-notes/indian_currency_new/validation/200/35.jpg")

label=model.predict(img)
print(label)
print(label.argmax())

