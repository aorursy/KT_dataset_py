import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

from keras.utils import to_categorical
dir_data = '../input/flowers-recognition/flowers'

print(os.listdir(dir_data)[1:])
category = os.listdir(dir_data)[1:]

for bunga in category:  
    path = os.path.join(dir_data,bunga)
    for img in os.listdir(path): 
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap='gray')
        plt.show()

        break 
    break 

print(img_array)
IMG_SIZE = 150

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()
training_data = []

def create_training_data():
    for bunga in category:  # do dogs and cats

        path = os.path.join(dir_data,bunga) 
        class_num = category.index(bunga)
        
        for img in tqdm(os.listdir(path)): 
            try:
                    img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE) 
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    training_data.append([new_array, class_num])
            except Exception as e:  # in the interest in keeping the output clean...
                pass
        
create_training_data()

import random

random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

#print(X[:5])
#print(y[:5])
len(X)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)
X.shape
y.shape
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

x_train.shape
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.
x_test = x_test / 255.
y_test.shape
y_trainfix = to_categorical(y_train)
y_testfix = to_categorical(y_test)
print(y_trainfix[:5])
print(y_testfix[:5])
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
batch_size = 1
epochs = 20
num_classes = 5
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(150,150,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.summary()
x_train.shape, y_trainfix.shape
trainpak = model.fit(x_train, y_trainfix, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_testfix))