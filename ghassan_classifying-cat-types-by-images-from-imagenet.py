# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.image as mpimg
# Any results you write to the current directory are saved as output.
from keras import backend as K
from keras.preprocessing import image                  

# Any results you write to the current directory are saved as output.
class1=os.listdir('../input/1/1')
class2=os.listdir('../input/2/2')
class3=os.listdir('../input/3/3')
class4=os.listdir('../input/4/4')
class5=os.listdir('../input/5/5')
class6=os.listdir('../input/6/6')
def load_image(path):
    img = image.load_img(path,target_size=(224,224))
    return image.img_to_array(img)
classes=[class1,class2,class3,class4,class5,class6]
def index_helper(i):
    return sum([len(k) for k in classes[:i]])
all_images_len=sum([len(i) for i in classes])
all_images=np.zeros([all_images_len,224,224,3])
for i in range(6):
    for j in range(len(classes[i])):
        all_images[j+index_helper(i),:,:,:]=load_image('../input/'+str(i+1)+'/'+str(i+1)+'/'+classes[i][j])

all_classes=np.zeros(all_images_len)
all_classes[:index_helper(1)]=1
all_classes[index_helper(1):index_helper(2)]=2
all_classes[index_helper(2):index_helper(3)]=3
all_classes[index_helper(3):index_helper(4)]=4
all_classes[index_helper(4):index_helper(5)]=5
all_classes[index_helper(5):index_helper(6)]=6
print(np.unique(all_classes,return_counts=True))
print([len(i) for i in classes])

processed_images=all_images/255
import gc
del all_images
gc.collect()
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(processed_images, all_classes, stratify=all_classes,test_size=0.20, random_state=42)

del processed_images
gc.collect()
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
Y_train=Y_train-1
Y_test=Y_test-1
Y_train_hot=np.eye(6)[Y_train.astype(int)]
Y_test_hot=np.eye(6)[Y_test.astype(int)]
X_in=Input(X_train.shape[1:])
x = ZeroPadding2D((1,1))(X_in)
x = Conv2D(64, 3, 3, activation='relu')     (x)
x = ZeroPadding2D((1,1))                           (x)
x = Conv2D(64, 3, 3, activation='relu')     (x)
x = MaxPooling2D((2,2), strides=(2,2))             (x)

x = ZeroPadding2D((1,1))                           (x)
x = Conv2D(128, 3, 3, activation='relu')    (x)
x = ZeroPadding2D((1,1))                           (x)
x = Conv2D(128, 3, 3, activation='relu')    (x)
x = MaxPooling2D((2,2), strides=(2,2))             (x)

x = ZeroPadding2D((1,1))                           (x)
x = Conv2D(256, 3, 3, activation='relu')    (x)
x = ZeroPadding2D((1,1))                           (x)
x = Conv2D(256, 3, 3, activation='relu')    (x)
x = ZeroPadding2D((1,1))                           (x)
x = Conv2D(256, 3, 3, activation='relu')    (x)
x = MaxPooling2D((2,2), strides=(2,2))             (x)

x = ZeroPadding2D((1,1))                           (x)
x = Conv2D(512, 3, 3, activation='relu')    (x)
x = ZeroPadding2D((1,1))                           (x)
x = Conv2D(512, 3, 3, activation='relu')    (x)
x = ZeroPadding2D((1,1))                           (x)
x = Conv2D(512, 3, 3, activation='relu')    (x)
x = MaxPooling2D((2,2), strides=(2,2))             (x)

x = ZeroPadding2D((1,1))                           (x)
x = Conv2D(512, 3, 3, activation='relu')    (x)
x = ZeroPadding2D((1,1))                           (x)
x = Conv2D(512, 3, 3, activation='relu')    (x)
x = ZeroPadding2D((1,1))                           (x)
x = Conv2D(512, 3, 3, activation='relu')    (x)
x = MaxPooling2D((2,2), strides=(2,2))             (x)

x = Flatten()                                      (x)
x = Dense(100, activation='relu')                 (x)
x = Dropout(0.5)                                   (x)
x = Dense(50, activation='relu')                 (x)
x = Dropout(0.5)                                   (x)
x = Dense(6, activation='softmax')                 (x)
model=Model(inputs=X_in,outputs=x,name="meow")
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
model.fit(x=X_train,y=Y_train_hot,epochs=4,batch_size=20)

model.evaluate(x=X_test, y=Y_test_hot)
