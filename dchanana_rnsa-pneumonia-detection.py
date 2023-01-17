import numpy as np
import pandas as pd
import tensorflow as tf
import pydicom
import os
import zipfile
import matplotlib.pyplot as plt
import pylab
import png
import keras
from tqdm import tqdm
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, LeakyReLU
from matplotlib.patches import Rectangle
datapath = "../input/"
print(os.listdir(datapath))
box = pd.read_csv(datapath+"stage_1_train_labels.csv")
pIds = box['patientId']
aux = pd.read_csv(datapath+"stage_1_detailed_class_info.csv")
train = pd.concat([box.loc[1:20293], aux.loc[1:20293].drop(labels=['patientId'], axis=1)], axis=1)
valid = pd.concat([box.loc[20294:28989], aux.loc[20294:28989].drop(labels=['patientId'], axis=1)], axis=1)
def visBox(pId):
    """
    Method to visualize boxes around lung opacities in training set
    Takes an entry from the parsed dict
    """
    dcmdata = pydicom.read_file(datapath+'stage_1_train_images/'+pId+'.dcm')
    dcmimg = dcmdata.pixel_array
    boxData = train[train['patientId'] == pId]
    plt.figure(figsize=(20,10))
    for i in boxData.index:
        rect = Rectangle((boxData['x'][i],boxData['y'][i]),boxData['width'][i],boxData['height'][i],fill=False,color='red')
        plt.gca().add_patch(rect)
    plt.axis('off')
    plt.imshow(dcmimg, cmap=pylab.cm.binary)
    plt.colorbar()
lstFileNames = []
i=0
for filename in tqdm(os.listdir(datapath+'stage_1_train_images/')):
    ds = pydicom.dcmread(datapath+'stage_1_train_images/'+filename)
    shape = ds.pixel_array.shape
    lstFileNames.append(filename.replace('.dcm',''))
    # Convert to float to avoid overflow or underflow losses.
    image_2d = ds.pixel_array.astype(float)
    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)
    # Write the PNG file
    with open('../working/'+filename.replace('.dcm',''), 'wb+') as png_file:
        w = png.Writer(shape[1], shape[0], greyscale=True)
        w.write(png_file, image_2d_scaled)
    i=i+1
    if i==30:
        with zipfile.ZipFile('imgs', 'w') as myzip:
            for f in lstFileNames:   
                myzip.write(f)
                os.remove('../working/'+f)
        i=0
        lstFileNames=[]
trainBox = box.replace(to_replace=float('nan'),value=0)
trainBox['patientId'] = box['patientId'].astype(str)+'.dcm'
train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
train_generator = train_datagen.flow_from_dataframe(
        directory=datapath+'stage_1_train_images',
        dataframe=trainBox,
        x_col='patientId',
        y_col=['x','y','width','height'], 
        has_ext=True,
        class_mode='other',
        target_size=(1024,1024),
        batch_size=32)
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(1024,1024,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=1000,decay=.99),
              metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=1024/16, epochs=317)
trainBox = box.replace(to_replace=float('nan'),value=0)
trainBox['patientId'] = box['patientId'].astype(str)+'.dcm'
trainBox

os.listdir('../working/')