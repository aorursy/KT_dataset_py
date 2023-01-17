import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt 

import cv2

import warnings

warnings.filterwarnings('ignore')

import keras 

import os 

from tqdm import tqdm

train=pd.read_csv('../input/train.csv')#load the train cvs file 

train.head(4)#checking the few line of train
train.describe()#description of train file 
plt.figure(figsize=(7,7))

sns.countplot(train['has_cactus'])

plt.show()
#let check some image file from train file 

image_file=os.listdir("../input/train/train")
image_file[0]#image name 

is_cactus=[]

for i in tqdm(range(len(train['id']))):

    for j in range(len(train['id'])):

        if(image_file[i]==train['id'][j]):

            is_cactus.append(train['has_cactus'][j])
df = pd.DataFrame(list(zip(image_file,is_cactus,)), columns =['image_name', 'is_cactus']) 
# let prepare the Data For 

path="../input/train/train/"+image_file[0]

import cv2
image=cv2.imread(path)

plt.imshow(image)
trainig_data=[]

for i in  tqdm(range(len(df['is_cactus']))):

    p="../input/train/train/"+image_file[i]

    image=cv2.imread(p,cv2.IMREAD_COLOR)

    image=image/255

    trainig_data.append( image)



X=np.array(trainig_data)    
Y=df['is_cactus']

import keras

from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val =train_test_split(X,Y,test_size=0.2,random_state=2)
print("Shape of X  train",X_train.shape)

print("Shape of X validate ",X_val.shape)

print("Shape of Y train",y_train.shape)

print("Shape of Y validate",y_val.shape)
from keras.preprocessing.image import ImageDataGenerator 

from keras.models import Sequential 

from keras.layers import Conv2D, MaxPooling2D 

from keras.layers import Activation, Dropout, Flatten, Dense 

from keras import backend as K
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))

model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='Adadelta', loss='binary_crossentropy', metrics=['accuracy'])
history=model.fit(X_train,y_train, epochs=30,batch_size=32,validation_data=(X_val,y_val))
plt.figure(figsize=(10,7))

plt.title('Loss')

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show();
plt.figure(figsize=(10,7))

plt.title('Accuracy')

plt.plot(history.history['acc'], label='train')

plt.plot(history.history['val_acc'], label='test')

plt.legend()

plt.show();