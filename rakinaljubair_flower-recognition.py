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
print(os.listdir('../input/flowers-recognition/flowers'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
import cv2 
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from tqdm import tqdm
print(os.listdir('../input/flowers-recognition/flowers'))
im_size=150
DAISY_DIR='../input/flowers-recognition/flowers/daisy'
DANDELION_DIR='../input/flowers-recognition/flowers/dandelion'
ROSE_DIR='../input/flowers-recognition/flowers/rose'
SUNFLOWER_DIR='../input/flowers-recognition/flowers/sunflower'
TULIP_DIR='../input/flowers-recognition/flowers/tulip'

name=[]
imdata=[]

def data_process(flower_name,DIR):
    for img in tqdm(os.listdir(DIR)):
        label=flower_name
        path=os.path.join(DIR,img)
        im=cv2.imread(path,cv2.IMREAD_COLOR)
        if img.endswith("jpg"):
            imd=cv2.resize(im,(im_size,im_size),cv2.INTER_AREA)
            name.append(str(label))
            imdata.append(np.array(imd))
        else:
            continue

    

data_process('Dandelion',DANDELION_DIR)
data_process('Daisy',DAISY_DIR)
data_process('Rose',ROSE_DIR)
data_process('Tulip',TULIP_DIR)
data_process('Sunflower',SUNFLOWER_DIR)
import random 
fig,ax=plt.subplots(5,2)
fig.set_size_inches(20,20)
for i in range(5):
    for j in range (2):
        l=random.randint(0,len(name))
        ax[i,j].imshow(imdata[l])
        ax[i,j].set_title('Flower: '+name[l])
        

plt.imshow(imdata[869])
from sklearn import preprocessing
LE=preprocessing.LabelEncoder()
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
Y = label_binarizer.fit_transform(name)


np.unique(Y)
Y.shape
im=np.array(imdata)
X=im/255
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)
y_train=y_train.astype(float)
y_test=y_test.astype(float)
x_train.dtype
y_train.shape
model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),input_shape=(150,150,3),
                                                         activation='relu'),
                                 tf.keras.layers.MaxPooling2D(2,2),
                                 tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
                                 tf.keras.layers.MaxPooling2D(2,2),
                                 tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'),
                                 tf.keras.layers.MaxPooling2D(2,2),
                                 tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'),
                                 tf.keras.layers.MaxPooling2D(2,2),
                                 tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu'),
                                 tf.keras.layers.MaxPooling2D(2,2),
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(512,activation='relu'),
                                 tf.keras.layers.Dense(128,activation='relu'),
                                 tf.keras.layers.Dense(128,activation='relu'),
                                 tf.keras.layers.Dense(32,activation='relu'),
                                 tf.keras.layers.Dense(512,activation='relu'),
                                 tf.keras.layers.Dense(5,activation='softmax')                                  
                                 ])
model.summary()
from keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.6, min_lr=0.0001)
model.compile(loss='categorical_crossentropy',optimizer='RMSprop',metrics=['accuracy'])


history = model.fit(x_train,y_train, batch_size = 128 , epochs =20,validation_data=(x_test,y_test),callbacks = [learning_rate_reduction])


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()