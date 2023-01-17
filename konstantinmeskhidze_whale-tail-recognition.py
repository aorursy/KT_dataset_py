# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.datasets import load_files       
from keras.utils import np_utils
import PIL
from PIL import ImageFile  
import cv2
from glob import glob
from tqdm import tqdm

import sklearn

import keras
from keras.utils import np_utils
from keras.preprocessing import image   

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_imgs = "../input/train"
test_imgs = "../input/test"

resize = 224
!ls "../input/train" -1A | wc -l
!ls "../input/test" -1A | wc -l
# Funtion to create a list of images' names
def dataset_list_load (image_dir):
    lstFilesJGP = []
    for dirName, subdirList, fileList in os.walk(image_dir):
        for filename in fileList:
            if ".jpg" in filename.lower():  # check whether the file's JPEG
                lstFilesJGP.append(os.path.join(dirName,filename))
    return lstFilesJGP

# Funtion to create a list of images' names
def dataset_list_load_name (image_dir):
    lstFilesJGP = []
    for dirName, subdirList, fileList in os.walk(image_dir):
        for filename in fileList:
            if ".jpg" in filename.lower():  # check whether the file's JPEG
                lstFilesJGP.append(filename)
    return lstFilesJGP
test_files = dataset_list_load_name(test_imgs)

print('There are %d test whale images.'% len(test_files))
print(test_files[1:10])
from keras.preprocessing import image                  
from tqdm import tqdm
import PIL

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_image_list).astype('float32')/255
test_tensors = paths_to_tensor(test_image_list).astype('float32')/255
whale_labels = pd.read_csv('../input/train.csv')
whale_labels.info()
whale_labels.head(10)  
whale_labels['Id'].value_counts()
whale_labels = whale_labels.loc[whale_labels['Id'] != 'new_whale']
whale_labels.info()
whale_labels.head(10)  
whale_labels['Id'].value_counts()
num_classes = len(whale_labels['Id'].unique())
print(num_classes)

Ids_enum = {cat: k for k,cat in enumerate(whale_labels.Id.unique())}
from keras.preprocessing import image                  
from tqdm import tqdm
import PIL
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True      

resize = 224
     
im_arrays = []
labels = []
fs = {} ##dictionary with original size of each photo 
for index, row in tqdm(whale_labels.iterrows()): 
    # CV2 using
    im = cv2.imread(os.path.join(train_imgs,row['Image']),0)
    norm_image = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    new_image = cv2.resize(norm_image,(resize,resize))
    new_image = np.reshape(new_image,[resize,resize,1])
    im_arrays.append(new_image)
    fs[row['Image']] = norm_image.shape
    
    # PIL Using
    #new_image = image.load_img(os.path.join(train_imgs,row['Image']), target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    #x = image.img_to_array(new_image)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    #np.expand_dims(x, axis=0)
    #im_arrays.append(x)
    
    labels.append(Ids_enum [row['Id']])
 
#We rescale the images by dividing every pixel in every image by 255.
#train_tensors = np.vstack(im_arrays).astype('float32')/255 # PIL
train_tensors = np.array(im_arrays).astype('float32')/255 # CV2

train_labels = np.array(labels)
train_targets = np_utils.to_categorical(train_labels, num_classes)

print(type(train_tensors), train_tensors.size, train_tensors.shape)
print(type(train_labels), train_labels.size, train_labels.shape)
print(type(train_targets), train_targets.size, train_targets.shape)
import random

whale_choose = whale_labels.sample()
print(whale_choose)
file_name = whale_choose['Image'].values[0]
whale_id = whale_choose['Id'].values[0] 
img_tail = mpimg.imread(os.path.join("../input/train",file_name))

print(file_name)
print(whale_id)
print(img_tail.shape)
imgplot = plt.imshow(img_tail)

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()

### TODO: Define your architecture.
model.add(Conv2D(filters = 16, kernel_size = 2, padding = 'same', activation='relu', 
                                         input_shape = train_tensors.shape[1:]))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(filters = 32, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(GlobalAveragePooling2D())
model.add(Dense(num_classes, activation = 'softmax'))

model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
split_percentage = 0.1

X_train_tens = train_tensors[int(train_tensors.shape[0]*split_percentage):, :, :, :]
X_train_targets = train_targets[int(train_tensors.shape[0]*split_percentage):, :]

Y_valid_tens = train_tensors[: int(train_tensors.shape[0]*split_percentage), :, :, :]
Y_valid_targets = train_targets[: int(train_tensors.shape[0]*split_percentage), :]

print(type(X_train_tens), X_train_tens.size, X_train_tens.shape)
print(type(X_train_targets), X_train_targets.size, X_train_targets.shape)

print(type(Y_valid_tens), Y_valid_tens.size, Y_valid_tens.shape)
print(type(Y_valid_targets), Y_valid_targets.size, Y_valid_targets.shape)


from keras.callbacks import ModelCheckpoint  

epochs = 20

checkpointer = ModelCheckpoint(filepath='weights.best.from_scratch.hdf5',
                               verbose=1, save_best_only= True)

model.fit(X_train_tens, X_train_targets, 
          validation_data = (Y_valid_tens, Y_valid_targets),
          epochs=epochs, batch_size=50, callbacks=[checkpointer], verbose=1)
# See that file with coefficients is created
import os
print(os.listdir())
model.load_weights('weights.best.from_scratch.hdf5')
from keras.preprocessing import image                  
from tqdm import tqdm
import PIL
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True          

resize = 224
     
test_arrays = []

for index in tqdm(test_files[:]): 
    # CV2 using    
    im = cv2.imread(os.path.join(test_imgs,index),0)
    norm_image = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    test_new_image = cv2.resize(norm_image,(resize,resize))
    test_new_image = np.reshape(test_new_image,[resize,resize,1])
    test_arrays.append(test_new_image)
test_tensors = np.array(test_arrays).astype('float32')/255 # CV2
# get index of predicted whale tails for each image in test set
whale_tail_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
Num = 55
print(whale_tail_predictions[:Num])
print(whale_labels.Id.unique()[1])
print(whale_labels.Id.unique()[65])
print(whale_labels.Id.unique()[165])
# Make predictions on test images and write a submission file            
file_path = 'submission_simple_model.csv'
with open(file_path , 'w') as file:
    file.write("Image,Id\n")
    for list_index in tqdm(enumerate(test_files)):
        sub_str = ""
        sub_str += test_files[list_index[0]]
        sub_str += ","
        sub_str += whale_labels.Id.unique()[whale_tail_predictions[list_index[0]]] 
        file.write(sub_str+"\n")