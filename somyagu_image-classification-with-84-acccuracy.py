# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# load the image
from keras.preprocessing.image import load_img
image = load_img("/kaggle/input/intel-image-classification/seg_train/seg_train/street/5713.jpg")
import os

from os import listdir
listdir("/kaggle/input")
listdir("/kaggle/input/intel-image-classification")
listdir("/kaggle/input/intel-image-classification/seg_train")
listdir("/kaggle/input/intel-image-classification/seg_train/seg_train")
listdir("/kaggle/input/intel-image-classification/seg_train/seg_train/street")
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

from keras.preprocessing.image import load_img
x_train=[]
y_train=[]

k="/kaggle/input/intel-image-classification/seg_train/seg_train"
for file1 in listdir(k):
    file2=k+"/"+file1
    for file3 in listdir(file2):
        file4=file2+"/"+file3
        image = load_img(file4,target_size=(64,64,3))
        img_array = img_to_array(image)
        x_train.append(img_array)
        y_train.append(file1)

x_test=[]
y_test=[]

k="/kaggle/input/intel-image-classification/seg_test/seg_test"
for file1 in listdir(k):
    file2=k+"/"+file1
    for file3 in listdir(file2):
        file4=file2+"/"+file3
        image = load_img(file4,target_size=(64,64,3))
        img_array = img_to_array(image)
        x_test.append(img_array)
        y_test.append(file1)

len(x_train),len(y_train)
len(x_test),len(y_test)
import numpy as np
x_train=np.array(x_train)
y_train=np.array(y_train)
x_test=np.array(x_test)
y_test=np.array(y_test)
x_train.shape,y_train.shape
from sklearn.preprocessing import LabelEncoder
k = LabelEncoder()
y_train = k.fit_transform(y_train)
y_test= k.fit_transform(y_train)
from keras.utils import to_categorical
y_train=to_categorical(y_train)
y_train.shape
from keras.applications.vgg16 import VGG16
model=VGG16(input_shape=(64,64,3),include_top=False)
#Import libraries to create model
from keras import Model
from keras.optimizers import SGD
from keras.layers import Flatten, Dense
def Intel_Image_classification():
    model=VGG16(input_shape=(64,64,3),include_top=False)
    for layer in model.layers:
        layer.trainable = False
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation="relu", kernel_initializer="he_uniform")(flat1)
    #class2 = Dense(62, activation="relu", kernel_initializer="he_uniform")(class1)
    output = Dense(6, activation="softmax")(class1)
    model1 = Model(inputs=model.inputs, outputs=output)
    opt = SGD(lr=0.01, momentum=0.9)
    model1.compile(optimizer=opt,loss="categorical_crossentropy", metrics=["accuracy"])
    return model1
obj=Intel_Image_classification()
x_train=x_train.astype(float)
x_test=x_test.astype(float)
train_norm=x_train/255
test_norm=x_test/255
obj.fit(train_norm,y_train,batch_size=32, epochs=10,verbose=1)
