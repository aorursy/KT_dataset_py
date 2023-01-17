import pandas as pd
import numpy as np
import os,cv2,random
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Convolution2D,Flatten,Activation,MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'

ROWS = 64
COLS = 64
CHANNELS = 3

dog_train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
cat_train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]
test_images = [TEST_DIR+i for i in os.listdir(TEST_DIR)]

train_images = dog_train_images[:1000] + cat_train_images[:1000]
random.shuffle(train_images)
test_images = test_images[:25]

def read_images(path):
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    return cv2.resize(img,(ROWS,COLS),interpolation=cv2.INTER_CUBIC)

def prep_data(images):
    size = len(images)
    data = np.ndarray((size,ROWS,COLS,CHANNELS),np.uint8)
    
    for i,image_file in enumerate(images):
        image = read_images(image_file)
        data[i] = image
    return data

train = prep_data(train_images)
test = prep_data(test_images)
label = []
def labeling(images):
    for i in images:
        if 'dog' in i:
            label.append(0)
        else :
            label.append(1)

labeling(train_images)
sns.countplot(label)
def show_images(i):
    img1 = read_images(cat_train_images[i])
    img2 = read_images(dog_train_images[i])
    
    pair = np.concatenate((img1,img2),axis=1)
    plt.figure()
    plt.imshow(pair)
    plt.show()

for i in range (40,44):
    show_images(i)
optimizer = 'adam'
loss = 'binary_crossentropy'

model = Sequential()
model.add(Convolution2D(32,3,3,border_mode='same',input_shape = (ROWS,COLS,CHANNELS),activation='relu'))
model.add(Convolution2D(32,3,3,border_mode='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,3,3,border_mode='same',activation='relu'))
model.add(Convolution2D(64,3,3,border_mode='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
          
model.add(Convolution2D(128,3,3,border_mode='same',activation='relu'))
model.add(Convolution2D(128,3,3,border_mode='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
          
model.add(Convolution2D(256,3,3,border_mode='same',activation='relu'))
model.add(Convolution2D(256,3,3,border_mode='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
          
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
          
model.compile(loss=loss,metrics = ['accuracy'],optimizer=optimizer)
early_stopping = EarlyStopping(patience=2,monitor='val_loss')
model.fit(train,label,epochs=3,callbacks=[early_stopping],batch_size=32)
