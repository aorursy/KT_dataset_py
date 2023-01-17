# Importing necessary libraries

from keras.layers import Input, Lambda, Dense, Flatten, Dropout, BatchNormalization
from keras import regularizers
from keras.models import Model, Sequential
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.utils import to_categorical
from PIL import Image
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import cv2
import os
from keras.applications.resnet50 import ResNet50

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
parasitized_data = os.listdir('../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/')
print(parasitized_data[:10]) #the output we get are the .png files

uninfected_data = os.listdir('../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/')
print('\n')
print(uninfected_data[:10])
plt.figure(figsize = (10,10))
for i in range(4):
    plt.subplot(1, 4, i+1)
    img = cv2.imread('../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized' + "/" + parasitized_data[i])
    plt.imshow(img)
    plt.title('PARASITIZED : 1')
    plt.tight_layout()
plt.show()
plt.figure(figsize = (10,10))
for i in range(4):
    plt.subplot(1, 4, i+1)
    img = cv2.imread('../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected' + "/" + uninfected_data[i+1])
    plt.imshow(img)
    plt.title('UNINFECTED : 0')
    plt.tight_layout()
plt.show()
infected = os.listdir('../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/') 
uninfected = os.listdir('../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/')
data = []
labels = []

for i in infected:
    try:
    
        image = cv2.imread("../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/"+i)
        image_array = Image.fromarray(image , 'RGB')
        resize_img = image_array.resize((64 , 64))
        data.append(np.array(resize_img))
        label = to_categorical(1, num_classes=2)
        labels.append(1)
        
    except AttributeError:
        print('')
    
for u in uninfected:
    try:
        
        image = cv2.imread("../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/"+u)
        image_array = Image.fromarray(image , 'RGB')
        resize_img = image_array.resize((64 , 64))
        data.append(np.array(resize_img))
        label = to_categorical(0, num_classes=2)
        labels.append(0)
        
    except AttributeError:
        print('')

data = np.array(data)
labels = np.array(labels)

np.save('Data' , data)
np.save('Labels' , labels)
n = np.arange(data.shape[0])
np.random.shuffle(n)
data = data[n]
labels = labels[n]
data = data.astype(np.float32)
labels = labels.astype(np.int32)
from sklearn.model_selection import train_test_split

train_x , test_x , train_y , test_y = train_test_split(data , labels , 
                                            test_size = 0.2 ,
                                            random_state = 111)
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# don't train existing weights
for layer in resnet.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = True
    else:
        layer.trainable = False
# useful for getting number of classes

num_classes = 2

model = Sequential()

model.add(resnet)

model.add(Flatten())

model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01), activation='relu'))

model.add(Dropout(0.3))

model.add(BatchNormalization())

model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01), activation='relu'))

model.add(Dropout(0.3))

model.add(BatchNormalization())

model.add(Dense(num_classes, activation='softmax'))

model.summary()

# (4) Compile 
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_aug = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,  
    zoom_range=0.2,        
    horizontal_flip=True,
    vertical_flip=True)  

val_aug= ImageDataGenerator(
    rescale=1./255)

training_set = train_aug.flow(
    train_x,
    train_y,
    batch_size=16)

test_set = val_aug.flow(
    test_x,
    test_y,
    batch_size=16)

r = model.fit_generator(training_set,
                    steps_per_epoch=len(training_set),
                    epochs=5,
                    validation_data=test_set,
                    validation_steps=len(test_set))
model_score = model.evaluate_generator(test_set,steps=50)
print("Model Test Loss:",model_score[0])
print("Model Test Accuracy:",model_score[1])
