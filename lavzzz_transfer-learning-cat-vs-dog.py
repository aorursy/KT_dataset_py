# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import shutil
print(os.listdir("../input"))

original_dataset_dir = "../input/train/"
base_dir = "../input/"

# Any results you write to the current directory are saved as output.
#define some directories to be used for data subsets and outputs 

train_dir = '../input/train'
test_dir = "../input/test"
temp_dir = '.'
output_dir = '.'
image_shape = (150,150,3) 
##useful libraries 
import os 
import matplotlib.pyplot as plt 
from scipy.ndimage import imread
from scipy.misc import imsave 
import numpy as np  
import time
%matplotlib inline 
##visualize a cat image and its shape from training directory 
random_cat_path = os.path.join(train_dir,'cat.25.jpg')
random_cat_show = plt.imread(random_cat_path)
plt.imshow(random_cat_show)
random_cat_show.shape
## visualize a dog image and shape from training directory 
random_dog_path = os.path.join(train_dir, 'dog.10.jpg')
random_dog_show = plt.imread(random_dog_path)
plt.imshow(random_dog_show)
random_dog_show.shape
##useful keras helper functions to pre-process images into the defined image_shape 
from keras.preprocessing import image

def loadAndResizeImage(img, w, h):
    return image.load_img(img, target_size = (w,h))

#random example with cat 

img = os.path.join(train_dir, 'dog.2000.jpg')
w = image_shape[0]
h = image_shape[1]
resized_image = loadAndResizeImage(img, w, h)
plt.imshow(resized_image)
def normalizedArrayfromImagePath(image_path, image_shape):
    
    """
    takes image path, uses loadAndResizeImage function and converts PIL image to array and returns it
    arguments image_path - path to a specific image 
    image_shape - tuple of size 3 with elements representing width, heigh, channels
    """
    
    _img = loadAndResizeImage(image_path, image_shape[0], image_shape[1])
    _normalizedArray = image.img_to_array(_img)/ 255 
    return(_normalizedArray)

def loadResizedNormalizedImages(basepath, path_array, img_shape):
    """
    arguments - 
        basepath - directory where images are contained
        path_array - # of images needed 
        img_shape - used to calculate the size of the array returned - tuple of size 3 
    """
    images = np.empty((len(path_array), img_shape[0], img_shape[1], img_shape[2]), dtype = np.float32)
    for i in range(len(path_array)):
        image_path = os.path.join(basepath, path_array[i])
        images[i] = normalizedArrayfromImagePath(image_path, img_shape)
    return images 
    


train_ex = 1000 
_train_dir_list = os.listdir(train_dir)
train_x = _train_dir_list[:train_ex]
len(train_x)

train_images_X = loadResizedNormalizedImages(train_dir, train_x, image_shape)
train_images_X.shape
validation_ex = 100 
validation_x = _train_dir_list[train_ex:train_ex + validation_ex]
len(validation_x)
validation_images_X = loadResizedNormalizedImages(train_dir, validation_x, image_shape)
validation_images_X.shape
def getYlabel(img):
    """
    get the y label 'cat' or 'dog' for given image path 
    
    """
    if 'cat' in img:
        return 0
    else:
        return 1
    
def getYlabelAsArray(X):
    Y = np.empty(len(X))
    for i in range (len(X)):
        Y[i] = getYlabel(X[i])
    return Y

        
train_y = getYlabelAsArray(train_x)
validation_y = getYlabelAsArray(validation_x)
## we will use InceptionV3 as transfer learn mechanism
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
base_model = InceptionV3(weights = "imagenet", include_top = False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation = "relu")(x)
x = Dropout(0.5)(x)
x = Dense(256, activation = "relu")(x)
x = Dropout(0.5)(x)
x = Dense(32, activation = "relu")(x)
predictions = Dense(1, activation = "sigmoid")(x)

model= Model(inputs= base_model.input, output = predictions)
for layer in base_model.layers:
    layer.trainable = False 
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ["accuracy"])

model.fit(x = train_images_X, y = train_y, batch_size = 16, epochs = 6, verbose =1,validation_data = (validation_images_X, validation_y))

model.save(os.path.join(output_dir,'model.h5'))
from keras.models import load_model
model = load_model(os.path.join(output_dir, 'model.h5'))
##time to test generalization over some other training images. We choose the last 300 training images 

evaluate_ex = 300 
_evaluate_dir_list = os.listdir(train_dir)
evaluate_x = _evaluate_dir_list[-evaluate_ex:]
len(evaluate_x)

evaluate_images_X = loadResizedNormalizedImages(train_dir, evaluate_x, image_shape)
evaluate_y = getYlabelAsArray(evaluate_x)




model.evaluate(x = evaluate_images_X, y = evaluate_y, batch_size = 10, verbose =1)

##test_dir = "../input/test"
"""
"""
test_ex = len(os.listdir(test_dir))
_test_dir_list = os.listdir(test_dir)
test_x = _test_dir_list[:]

test_images_X = loadResizedNormalizedImages(test_dir, test_x, image_shape)






test_images_X.shape
test_y = model.predict(test_images_X)
test_y.shape
import numpy as np
##predarray = np.asarray(Bin)
import csv

csvfile = os.path.join(output_dir, 'newAlgo.csv')

with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in test_y:
        writer.writerow(val) 
        
import pandas 
df = pandas.read_csv('newAlgo.csv')
print(df)
