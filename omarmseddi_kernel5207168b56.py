# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from numpy.random import seed

seed(101)

from tensorflow import set_random_seed

set_random_seed(101)



import pandas as pd

import numpy as np





import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation

from tensorflow.keras.models import Sequential

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.optimizers import Adam



import os

import cv2



from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

import itertools

import shutil

import matplotlib.pyplot as plt

%matplotlib inline

IMAGE_SIZE = 96

IMAGE_CHANNELS = 3



SAMPLE_SIZE = 80000 # the number of images we use from each of the two classes

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_data = pd.read_csv('../input/pices-de-monnaie/train.csv')

print(df_data.shape)
df_data['label'].value_counts()
df_data.head()
y = df_data['label']



df_train, df_val = train_test_split(df_data, test_size=0.10, random_state=101, stratify=y)



print(df_train.shape)

print(df_val.shape)
df_train['label'].value_counts()
no_tumor_tissue = os.path.join(train_dir, 'dinar')

os.mkdir(no_tumor_tissue)

has_tumor_tissue = os.path.join(train_dir, 'deuxdinars')

os.mkdir(has_tumor_tissue)

has_tumor_tissue = os.path.join(train_dir, 'cinqdinars')

os.mkdir(has_tumor_tissue)

has_tumor_tissue = os.path.join(train_dir, 'dixmilimes')

os.mkdir(has_tumor_tissue)

has_tumor_tissue = os.path.join(train_dir, 'vingtmilimes')

os.mkdir(has_tumor_tissue)

has_tumor_tissue = os.path.join(train_dir, 'cinqentesmilimes')

os.mkdir(has_tumor_tissue)

has_tumor_tissue = os.path.join(train_dir, 'centmilimes')

os.mkdir(has_tumor_tissue)

has_tumor_tissue = os.path.join(train_dir, 'deuxcentsmilimes')

os.mkdir(has_tumor_tissue)

has_tumor_tissue = os.path.join(train_dir, 'cinqesmilimes')

os.mkdir(has_tumor_tissue)





# create new folders inside val_dir

no_tumor_tissue = os.path.join(val_dir, 'dinar')

os.mkdir(no_tumor_tissue)

has_tumor_tissue = os.path.join(val_dir, 'deuxdinars')

os.mkdir(has_tumor_tissue)

has_tumor_tissue = os.path.join(val_dir, 'cinqdinars')

os.mkdir(has_tumor_tissue)

has_tumor_tissue = os.path.join(val_dir, 'dixmilimes')

os.mkdir(has_tumor_tissue)

has_tumor_tissue = os.path.join(val_dir, 'vingtmilimes')

os.mkdir(has_tumor_tissue)

has_tumor_tissue = os.path.join(val_dir, 'cinqentesmilimes')

os.mkdir(has_tumor_tissue)

has_tumor_tissue = os.path.join(val_dir, 'centmilimes')

os.mkdir(has_tumor_tissue)

has_tumor_tissue = os.path.join(val_dir, 'deuxcentsmilimes')

os.mkdir(has_tumor_tissue)

has_tumor_tissue = os.path.join(val_dir, 'cinqentesmilimes')

os.mkdir(has_tumor_tissue)
os.listdir('base_dir/train_dir')
def Image_read(image):

    img = cv2.imread(image)

    x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return x

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import numpy as np

import glob,cv2

imgList = glob.glob("../input/trainimages/train/train/*")

img = Image_read("../input/trainimages/train/train/559210.jpg")

print(img)

plt.figure()

plt.imshow(img) 

plt.show() 

minx=0

maxx=224

miny=0

maxy=224

test=True

testy=True

for i in range(224):

    for j in range(224):

        if img[i][j]<100:

            if(test==True):

                minx=j

                test=False

            if(j<maxx):

                maxx=j

            if(testy==True):

                miny=i

                testy=False

            if(i<maxy):

                maxy=i

print(minx,maxx,miny,maxy)

retval, thresh_gray = cv2.threshold(img, thresh=150, maxval=255,type=cv2.THRESH_BINARY_INV)

plt.figure()

plt.imshow(thresh_gray) 

plt.show() 

contours,h = cv2.findContours(thresh_gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE   )

for cnt in contours:

    (x,y),radius = cv2.minEnclosingCircle(cnt)

    center = (int(x),int(y))

    radius = int(radius)

    print(center,radius)



cv2.circle(img,(164,112),125,(0,255,0),2)

plt.figure()

plt.imshow(img) 

plt.show()             

            
import glob

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import  os # data 



# Image processing

from PIL import Image, ImageFile

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



# Plotting

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

X_data = []

Y_data = []

imgList = glob.glob("../input/trainimages/train/*")

data=pd.read_csv("../input/pices-de-monnaie/train.csv")

print(list(data[data.img==725416].label)[0])
folder = "../input/trainimages/train/train/"

import os, sys

from IPython.display import display

from IPython.display import Image as _Imgdis

imgList = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

print("Working with {0} images".format(len(imgList)))

print("Image examples: ")

for i in range(40, 42):

    print(imgList[i])

    display(_Imgdis(filename=folder + "/" + imgList[i], width=224, height=224))
# Original Dimensions

image_width = 224

image_height = 224

ratio = 2



image_width = int(image_width / ratio)

image_height = int(image_height / ratio)



channels = 3

nb_classes = 1

X_data=[]

Y_data=[]

X_data = np.ndarray(shape=(len(imgList)-1, image_height, image_width, channels),

                     dtype=np.float32)



i = 0

k=imgList[:100]

for im in k:

    img = load_img(folder + "/" + im)  # this is a PIL image

    img.thumbnail((image_width, image_height))

    # Convert to Numpy Array

    x = img_to_array(img)  

    x = x.reshape((112, 112,3))

    # Normalize

    x = (x - 128.0) / 128.0

    if list(data[data.img==int(im[:-4])].label):

        X_data[i] = x

        Y_data.append (list(data[data.img==int(im[:-4])].label)[0])

        i+=1

print(X_data[0])

print(X_data[0][0].size)

for i in range(100):

    print(Y_data[i])
from sklearn.model_selection import train_test_split



#Splitting 

X_train, X_test, y_train, y_test = train_test_split(X_data[:100], Y_data, test_size=0.2, random_state=33)

print("Train set size: {0}, Val set size: {1}".format(len(X_train),len(X_test)))

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
import keras.backend as K

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

from keras.layers import Activation, Dense

from keras.utils.np_utils import to_categorical



from keras.optimizers import SGD, Adam, Adagrad, RMSprop

def load_CNN(output_size):

  K.clear_session()

  model = Sequential()

  model.add(Conv2D(128, (5, 5),

               input_shape=(112,112,3),

               activation='relu'))

  model.add(MaxPool2D(pool_size=(2, 2)))

  #model.add(BatchNormalization())



  model.add(Conv2D(64, (3, 3), activation='relu'))

  model.add(MaxPool2D(pool_size=(2, 2)))

  #model.add(BatchNormalization())



  model.add(Conv2D(32, (3, 3), activation='relu'))

  model.add(MaxPool2D(pool_size=(2, 2)))

  #model.add(BatchNormalization())



  model.add(Flatten())

  model.add(Dense(512, activation='relu'))

  model.add(Dense(output_size, activation='softmax'))

  return model
model = load_CNN(1)

model.summary()
from keras.layers import Dropout

from keras.layers import BatchNormalization

from keras.callbacks import EarlyStopping

model.compile(loss='sparse_categorical_crossentropy',

              optimizer=Adam(lr=0.0005),

              metrics=['accuracy'])



weights = model.get_weights()

early_stop_loss = EarlyStopping(monitor='loss', patience=3, verbose=1)

early_stop_val_acc = EarlyStopping(monitor='val_acc', patience=3, verbose=1)

model_callbacks=[early_stop_loss, early_stop_val_acc]
batch_sizes = [4, 8, 16, 32, 64, 128]

histories_acc = []

histories_val = []

for batch_size in batch_sizes:

    h = model.fit(X_train, y_train,

                batch_size=batch_size,

                epochs=25,

                verbose=0,

                callbacks=[early_stop_loss],

                shuffle=True,

                validation_data=(X_test, y_test))