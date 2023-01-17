import os
import cv2
from sklearn.utils import shuffle
import numpy as np
def Process(path):   
    Images=[]
    Labels=[]
    label=0
    for labels in os.listdir(path):
        if labels=='glacier':
            label=2
        elif labels=='sea':
            label=4
        elif labels=='buildings':
            label=0
        elif labels=='forest':
            label=1
        elif labels=='street':
            label=5
        elif labels=='mountain':
            label=3
        for images in os.listdir(path+labels):
            image=cv2.imread(path+labels+r'/'+images)
            image=cv2.resize(image,(224,224))
            Images.append(image)
            Labels.append(label)
    return shuffle(Images,Labels,random_state=817328462)
    
    
path='../input/intel-image-classification/seg_train/seg_train/'
os.listdir(path)
Images,Labels=Process(path)
print(len(Images),len(Labels))
Images = np.array(Images) #converting the list of images to numpy array.
Labels = np.array(Labels)
print("Shape of Images:",Images.shape)
print("Shape of Labels:",Labels.shape)
#import keras 
#X_train=Images/255
#Y_train=keras.utils.to_categorical(Labels,6)

    
path_test='../input/intel-image-classification/seg_test/seg_test/'
X_test,Y_test=Process(path_test)
print(len(X_test),len(Y_test))
X_test = np.array(X_test) #converting the list of images to numpy array.
Y_test = np.array(Y_test)
#import keras 
#X_test=Images/255
#Y_test=keras.utils.to_categorical(Labels,6)

import matplotlib.pyplot as plot
def get_classlabel(class_code):
    labels = {2:'glacier', 4:'sea', 0:'buildings', 1:'forest', 5:'street', 3:'mountain'}
    return labels[class_code]


f,ax = plot.subplots(5,5) 
f.subplots_adjust(0,0,3,3)
for i in range(0,5,1):
    for j in range(0,5,1):
        rnd_number = np.random.randint(0,len(Images))
        ax[i,j].imshow(Images[rnd_number])
        ax[i,j].set_title(get_classlabel(Labels[rnd_number]))
        ax[i,j].axis('off')

train_counts=np.unique(Labels, return_counts=True)
fig = plot.figure()
ax = fig.add_axes([0,0,2,2])
for i in range(6):
    x=train_counts[0][i]
    y=train_counts[1][i]
    ax.bar(get_classlabel(x),y)
import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.utils as Utils
from keras.utils.vis_utils import model_to_dot

from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten,Activation,BatchNormalization
from keras.optimizers import Adamax
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K

from keras.applications import VGG16
VGG16 = VGG16(weights='imagenet', include_top=False)
for i,layers in enumerate(VGG16.layers):
    print(i,layers.__class__.__name__)
def addTopOFModel(bottom_model,classes):
    top_model=bottom_model.output
    top_model=GlobalAveragePooling2D()(top_model)
    top_model=Dense(1024,activation='relu')(top_model)
    top_model=Dense(512,activation='relu')(top_model)
    top_model=Dense(classes,activation='softmax')(top_model)
    return top_model
    
from keras.layers import GlobalAveragePooling2D
classes=6
FC=addTopOFModel(VGG16,classes)
model=Model(inputs=VGG16.input,outputs=FC)
print(model.summary())

model.compile(optimizer=Optimizer.Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history = model.fit(Images,Labels, batch_size=128, epochs=7, validation_split = 0.2)
test=model.evaluate(X_test,Y_test)
print('Test loss:', test[0])
print('Test accuracy:', test[1])
model.save_weights("model.h5")