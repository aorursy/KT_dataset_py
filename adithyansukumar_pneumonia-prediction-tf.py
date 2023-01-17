import os
from matplotlib.image  import imread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
train_path='../input/chest-xray-pneumonia/chest_xray/train'
test_path='../input/chest-xray-pneumonia/chest_xray/val'
normal=train_path+'/NORMAL'+'/IM-0152-0001.jpeg'
pneumonia=train_path+'/PNEUMONIA'+'/person1005_virus_1688.jpeg'
normal_img=imread(normal)
pneumonia_img=imread(pneumonia)
plt.imshow(normal_img)
plt.imshow(pneumonia_img)
normal_img.shape
pneumonia_img.shape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten,Dropout
from tensorflow.keras.applications import VGG16
conv_base=VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))
model = Sequential()
model.add(conv_base)
model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.summary()
conv_base.trainable=False
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_data_gen=ImageDataGenerator(rotation_range=20,
                                  width_shift_range=0.10,
                                  height_shift_range=0.10,
                                  rescale=1./255,
                                  shear_range=0.10,
                                  zoom_range=0.1,
                                  horizontal_flip=True,
                                  fill_mode='nearest')
test_data_gen=ImageDataGenerator(rescale=1./255)
train_data=train_data_gen.flow_from_directory(train_path,target_size=(150,150),color_mode='rgb',batch_size=16,class_mode='binary')
test_data=test_data_gen.flow_from_directory(test_path,target_size=(150,150),color_mode='rgb',batch_size=16,class_mode='binary')
from tensorflow.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss')
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
results=model.fit_generator(train_data,steps_per_epoch=100,epochs=100,validation_data=test_data,callbacks=[es])
conv_base.trainable=True
set_trainable=False
for layer in conv_base.layers:
    if layer.name=='block5_conv1':
        set_trainable=True
    elif layer.name=='block5_conv2':
        set_trainable=True
    elif layer.name=='block5_conv3':
        set_trainable=True
    if set_trainable:
        layer.trainable=True
    else:
        layer.trainable=False
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
results=model.fit_generator(train_data,steps_per_epoch=100,epochs=100,validation_data=test_data,callbacks=[es])
losses=pd.DataFrame(model.history.history)
losses
losses[['loss','val_loss']].plot()
losses[['accuracy','val_accuracy']].plot()
test='../input/chest-xray-pneumonia/chest_xray/test'
test_data2=test_data_gen.flow_from_directory(test,target_size=(150,150),color_mode='rgb',batch_size=16,class_mode='binary')
model.evaluate_generator(test_data2)
train_data.class_indices
from tensorflow.keras.preprocessing import image
test1=image.load_img('../input/pneumonia-test/pneumonia1.jpg',target_size=(150,150,3))
test1
test1=image.img_to_array(test1)
test1=np.expand_dims(test1,axis=0)
int(model.predict(test1))
test2=image.load_img('../input/normaltest2/normal2.jpg',target_size=(150,150,3))
test2
test2=image.img_to_array(test2)
test2=np.expand_dims(test2,axis=0)
int(model.predict(test2))
