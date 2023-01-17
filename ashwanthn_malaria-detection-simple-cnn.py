import cv2
import os
from PIL import Image
import keras
os.environ['KERAS_BACKEND'] = 'tensorflow'

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
np.random.seed(1000)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
img_dir = '/kaggle/input/cell-images-for-detecting-malaria/cell_images/'
dataset= []
label=[]
uninfected_img = os.listdir(img_dir + 'Uninfected/')
for i,img_name in enumerate(uninfected_img):
    if(img_name.split('.')[1] == 'png'):
        img = cv2.imread(img_dir + 'Uninfected/' + img_name)
        img = Image.fromarray(img,'RGB')
        img = img.resize((64,64))
        dataset.append(np.array(img))
        label.append(1)

    

para_img = os.listdir(img_dir + 'Parasitized/')
for i,img_name in enumerate(para_img):
    if(img_name.split('.')[1] == 'png'):
        img = cv2.imread(img_dir + 'Parasitized/' + img_name)
        img = Image.fromarray(img,'RGB')
        img = img.resize((64,64))
        dataset.append(np.array(img))
        label.append(0)
    
label1 = np.array(label)
type(label1)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,Dense,Conv2D,MaxPooling2D,BatchNormalization,Flatten
model  = Sequential()
model.add(Conv2D(32,(3,3), input_shape = (64,64,3),activation = 'relu', padding = 'same',data_format = 'channels_last'))
model.add(MaxPooling2D(pool_size = (2,2),data_format = 'channels_last'))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2),data_format = 'channels_last'))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(activation = 'relu',units = 512))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.2))
model.add(Dense(activation = 'relu',units = 256))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.2))
model.add(Dense(activation = 'sigmoid',units = 1))
model.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(dataset,label1,test_size = 0.25,random_state = 900)

classify = model.fit(np.array(X_train),y_train,batch_size = 10,epochs=20, validation_split=0.2)


y_pred = model.predict(np.array(X_test))
y_hat = y_pred
y_hat = y_hat>0.5
y_test1 = y_test>0.5
y_test1
y_hat
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test1,y_hat)
model.save('malaria_cnn.h5')
