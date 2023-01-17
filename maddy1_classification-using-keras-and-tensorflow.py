# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image

from sklearn.metrics import confusion_matrix
import itertools

import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
import random as rnd

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile

print(os.listdir('../input/flowers/flowers'))

root_dir = '../input/flowers/flowers'
rows = 3
cols = 2
fig, ax = plt.subplots(rows, cols, frameon=False, figsize=(10, 10))
fig.suptitle('Random Image from Each Flower Class', fontsize=20)
sorted_food_dirs = sorted(os.listdir(root_dir))
for i in range(rows):
    for j in range(cols):
        try:
            food_dir = sorted_food_dirs[i*cols + j]
        except:
            break
        all_files = os.listdir(os.path.join(root_dir, food_dir))
        rand_img = np.random.choice(all_files)
        img = plt.imread(os.path.join(root_dir, food_dir, rand_img))
        ax[i][j].imshow(img)
        ec = (0, 1, .1)
        fc = (0, .7, .2)
        ax[i][j].text(0, -20, food_dir, size=6, rotation=0,
                ha="left", va="top", 
                bbox=dict(boxstyle="round", ec=ec, fc=fc))
plt.setp(ax, xticks=[], yticks=[])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
labels = []
flowers = []
total = 0
# ../input/
PATH = os.path.abspath(os.path.join('..', 'input', 'flowers', 'flowers'))
print("Importing images of daisy flowers")
# ../input/sample/images/
SOURCE_IMAGES = os.path.join(PATH, "daisy")

# ../input/sample/images/*.jpg
daisy = glob(os.path.join(SOURCE_IMAGES, "*.jpg"))

# Prepare labels for daisy

i = 0
while i < len(daisy):
    labels.append("daisy")
    flowers.append(daisy[i])
    i += 1
    total += 1

print("Imported "+ str(i) + " images of daisy flowers")
 #importing dandelions

# ../input/sample/images/
SOURCE_IMAGES = os.path.join(PATH, "dandelion")

# ../input/sample/images/*.jpg
dandelion = glob(os.path.join(SOURCE_IMAGES, "*.jpg"))

# Prepare labels for dandelion

i = 0
while i < len(dandelion):
    labels.append("dandelion")
    flowers.append(dandelion[i])
    i += 1
    total += 1
    
print("Imported "+ str(i) + " images of dandelion flowers")

# importing roses

# ../input/sample/images/
SOURCE_IMAGES = os.path.join(PATH, "rose")

# ../input/sample/images/*.jpg
rose = glob(os.path.join(SOURCE_IMAGES, "*.jpg"))

# Prepare labels for daisy

i = 0
while i < len(rose):
    labels.append("rose")
    flowers.append(rose[i])
    i += 1
    total += 1

print("Imported "+ str(i) + " images of rose flowers")
# importing sunflower

# ../input/sample/images/
SOURCE_IMAGES = os.path.join(PATH, "sunflower")

# ../input/sample/images/*.jpg
sunflower = glob(os.path.join(SOURCE_IMAGES, "*.jpg"))

# Prepare labels for daisy

i = 0
while i < len(sunflower):
    labels.append("sunflower")
    flowers.append(sunflower[i])
    i += 1
    total += 1

print("Imported "+ str(i) + " images of sunflowers flowers")

# importing tulips

# ../input/sample/images/
SOURCE_IMAGES = os.path.join(PATH, "tulip")

# ../input/sample/images/*.jpg
tulip = glob(os.path.join(SOURCE_IMAGES, "*.jpg"))

# Prepare labels for daisy

i = 0
while i < len(tulip):
    labels.append("tulip")
    flowers.append(tulip[i])
    i += 1
    total += 1

print("Imported "+ str(i) + " images of tulip flowers")
print("Totals:")
print("Imported " + str(total) + " images in total")
def proc_images():

    x = [] # images as arrays
    y = [] # labels Infiltration or Not_infiltration
    WIDTH = 100
    HEIGHT = 100
    n = 0

    for img in flowers:
        base = os.path.basename(img)

        # Read and resize image
        full_size_image = cv2.imread(img)
        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        y.append(labels[n])
        n += 1

    print("Resized "+ str(n) + " images")
    return x,y 
print("Resizing Images...")
x,y = proc_images()

df = pd.DataFrame()
df["labels"]=y
df["flowers"]=x
fig,ax=plt.subplots(5,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range (2):
        l=rnd.randint(0,len(y))
        ax[i,j].imshow(x[l])
        ax[i,j].set_title('Flower: '+y[l])
        
plt.tight_layout()
        
le=LabelEncoder()
Y=le.fit_transform(y)
Y=to_categorical(Y,5)
X=np.array(x)
X=X/255

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

# # modelling starts using a CNN.

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (240,240,3)))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
 

model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(5, activation = "softmax"))
batch_size=64
epochs=25

from keras.callbacks import ReduceLROnPlateau
red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)
model.compile(optimizer=Adam(lr=0.01),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()
