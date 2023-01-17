import numpy as np
import pandas as pd
import os
import cv2
direc="../input/plantvillag/PlantVillag/crowdai"
imagepaths=[]
for lbl in os.listdir(direc):
    for img_file in os.listdir(direc+"/"+lbl):
        imagepaths.append(direc+"/"+lbl+"/"+img_file)
#     print(lbl)
print(len(imagepaths))
def getLabel(f_name):
    lbl=np.zeros((38,1))
    lbl[int(f_name.split('/')[-2].split('_')[-1])]=1
    return lbl
    
def preprocessData(filepaths):
    prep_data=[]
    for img_file in filepaths:
        img = cv2.imread(img_file)
#         print(img.shape)
        img= cv2.resize(img,(64,64))
        label=getLabel(img_file)
        prep_data.append([img,label])
    return np.array(prep_data)
data=preprocessData(imagepaths)
data[0][1]
# import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
train_x=np.array([i[0] for i in data[:-2000]])
train_y=np.array([i[1].flatten() for i in data[:-2000]])
test_x=np.array([i[0] for i in data[-2000:]])
test_y=np.array([i[1].flatten() for i in data[-2000:]])
print(train_y.shape)
model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=(64,64,3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(64,(3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

# Fully connected layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(38,activation='softmax'))

# model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
print(model.summary())
model.fit(train_x, train_y, validation_data=(test_x, test_y),epochs=10,batch_size=128)

_model=model.to_json()
with open("Model1.json","w") as json_file:
    json_file.write(_model)
model.save_weights("weights1.h5")
