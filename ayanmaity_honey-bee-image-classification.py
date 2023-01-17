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
df = pd.read_csv('../input/bee_data.csv')
img_names = list(df['file'])
df.head()
from skimage.io import imread
import matplotlib.pyplot as plt
img_paths = []

for img in img_names:
    img_paths.append('../input/bee_imgs/bee_imgs/'+img)
img0 = imread(img_paths[0])
img2 = imread(img_paths[2])
plt.subplot(121)
plt.imshow(img0)
plt.subplot(122)
plt.imshow(img2)
from skimage.transform import rescale, resize, downscale_local_mean
img_list = []
for img in img_paths:
    img_list.append(resize(imread(img),(100,100,3)))
img_np = np.array(img_list)
health = df['health']
pollen = df['pollen_carrying']
caste = df['caste']
df['health'].unique()
c=0
for i in df['health'].unique():
    health[health==i] = c
    c=c+1
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
health_en = enc.fit_transform(np.array(health).reshape(-1,1))
p = np.random.permutation(len(health))
img_np = img_np[p]
health_en = health_en[p]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(img_np, health_en, test_size=0.15, random_state=100)
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=180,  
            zoom_range = 0.1,  
            width_shift_range=0.2,  
            height_shift_range=0.2, 
            horizontal_flip=True, 
            vertical_flip=True,
            validation_split=0.1)
datagen.fit(X_train)
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,MaxPool2D,BatchNormalization
model_h = Sequential()
model_h.add(Conv2D(filters=16,kernel_size=(3,3),padding='same',input_shape=(100, 100, 3)))
model_h.add(MaxPool2D(2))
model_h.add(BatchNormalization(axis=-1))
model_h.add(Conv2D(filters=32,kernel_size=(3,3),padding='same'))
model_h.add(MaxPool2D(2))
model_h.add(BatchNormalization(axis=-1))
model_h.add(Flatten())
model_h.add(Dense(256,activation='relu'))
#model_h.add(Dense(128,activation='relu'))
model_h.add(Dense(6,activation='relu'))
model_h.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = model_h.fit_generator(datagen.flow(X_train,y_train.todense(), batch_size=256),steps_per_epoch=50,epochs=10,verbose=1)
model_h.evaluate(X_train,y_train)
model_h.evaluate(X_test,y_test)
loss_curve = hist.history['loss']
plt.plot(list(range(len(loss_curve))),loss_curve)
model_h.save('my_model1.h5')
from IPython.display import FileLink, FileLinks
FileLinks('.') 
np.save('img_np.npy',img_np)
