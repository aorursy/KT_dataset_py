import numpy as np 
import pandas as pd 
import tensorflow as tf
import sklearn
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from keras.preprocessing.image import img_to_array
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import backend as K
from keras.layers import BatchNormalization
import zipfile

with zipfile.ZipFile("../input/dogs-vs-cats/train.zip","r") as z:
    z.extractall(".")
    
with zipfile.ZipFile("../input/dogs-vs-cats/train.zip","r") as z:
    z.extractall(".")


categories = ["cat", "dog"]
data_dir = "/kaggle/working/train"
IMG_SIZE = 64

def create_img_array(data_dir):  # function that creates a 3D array holding the images
    img_array_list = []
    label_list = []
    path = data_dir
        
    for img in tqdm(os.listdir(path)):
            
        try:  # some of the images have error 
            if img.startswith('cat'):
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) 
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                img_array = img_to_array(img_array)
                img_array_list.append(img_array)
                label_list.append(0) # cats labelled 0
                
            elif img.startswith('dog'):
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                img_array = img_to_array(img_array)
                img_array_list.append(img_array)
                label_list.append(1) # dogs labelled 1
                
                
        except Exception as e:
            print(str(e))
            
    return img_array_list, label_list


img, labels = create_img_array(data_dir) 
img = np.asarray(img) # convert the list with img arrays to array
labels = np.array(labels) # convert list with labels to array
labels
img.shape
from keras.utils import to_categorical  # encode the labels using one hot encoding
labels = to_categorical(labels, num_classes = 2)
labels
from sklearn.model_selection import train_test_split # random split into train test data

train_x, test_x, train_y, test_y = train_test_split(img, labels, random_state = 42, test_size = 0.2)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
train_x = train_x.astype('float32')/255.0 # normalise the pixel values to between [0, 1] works better on CNN
test_x = test_x.astype('float32') / 255.0
model = Sequential()
K.common.image_dim_ordering() == 'th'
model.add(Conv2D(128, (3, 3), input_shape = (IMG_SIZE, IMG_SIZE, 1), activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (2, 2))) # pooling is to reduce the size of the images as much as possible, downsampling
model.add(Dropout(0.2))# dropout is a regularization technique to reduce overfitting, by giving each node a probability of being dropped
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2))) 
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())# get 1D array to feed into the Dense layer
model.add(Dense(units = 512, activation = 'tanh'))
model.add(Dropout(0.2))
model.add(Dense(units = 256, activation = 'tanh'))
model.add(Dropout(0.2))
model.add(Dense(units = 2, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
### chose to run just 10 epochs due to time, could have ran up to 50 epochs to get better results 
model_results = model.fit(train_x, train_y, epochs = 10, batch_size = 64, validation_data = (test_x, test_y), verbose = 1)

score, acc = model.evaluate(test_x, test_y, batch_size = 64)
print(acc)
pred = model.predict(test_x) # can see model's prediction perccentages on whether image is cat or dog
pred
plt.plot(model_results.history['accuracy'])
plt.plot(model_results.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'val'], loc = 'upper left')
plt.show()

# the larger the training accuracy compared to val_accuracy, the greater the overfitting problem
plt.plot(model_results.history['loss'])
plt.plot(model_results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'val'], loc = 'upper left')
plt.show()

# If you have loss noticeably lower than val_loss it is the sign of overfitting.
# A model that is underfit will have high training and high testing error while an overfit model
# will have extremely low training error but a high testing error.
from keras.preprocessing.image import ImageDataGenerator
train_x, test_x, train_y, test_y = train_test_split(img, labels, test_size = 0.2)
train_x = train_x.astype('float32')/255.0 # normalise the pixel values to between [0, 1] works better on CNN
test_x = test_x.astype('float32') / 255.0
train_generator = ImageDataGenerator(rotation_range = 40, height_shift_range = 0.2, width_shift_range = 0.2,horizontal_flip = True) 

# need to know which augmentations are relevant to the data
train_generator.fit(train_x)

train_generator = train_generator.flow(train_x, train_y, batch_size = 64, shuffle = False)

model.fit_generator(train_generator, steps_per_epoch = len(train_x) / 64,epochs = 15, validation_data = 
                   (test_x, test_y))
pred = model.predict(test_x)
pred
score, acc = model.evaluate(test_x, test_y, batch_size = 64)
print(acc) # accuracy with data augmentation
plt.plot(model_results.history['accuracy'])
plt.plot(model_results.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'val'], loc = 'upper left')
plt.show()
plt.plot(model_results.history['loss'])
plt.plot(model_results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'val'], loc = 'upper left')
plt.show()

