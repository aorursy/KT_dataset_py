

import matplotlib.pyplot as plt

import numpy as np 

import pandas as pd 

import os 
train_dir='../input/plant-seedlings-classification/train'

test_dir='../input/plant-seedlings-classification/test'



categories = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',

              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']

print(categories)
def category_to_label(category):

    if category == 'Black-grass': return [1,0,0,0,0,0,0,0,0,0,0,0]

    elif category == 'Charlock': return [0,1,0,0,0,0,0,0,0,0,0,0]

    elif category == 'Cleavers': return [0,0,1,0,0,0,0,0,0,0,0,0]

    elif category == 'Common Chickweed': return [0,0,0,1,0,0,0,0,0,0,0,0]

    elif category == 'Common wheat': return [0,0,0,0,1,0,0,0,0,0,0,0]

    elif category == 'Fat Hen': return [0,0,0,0,0,1,0,0,0,0,0,0]

    elif category == 'Loose Silky-bent': return [0,0,0,0,0,0,1,0,0,0,0,0]

    elif category == 'Maize': return [0,0,0,0,0,0,0,1,0,0,0,0]

    elif category == 'Scentless Mayweed': return [0,0,0,0,0,0,0,0,1,0,0,0]

    elif category == 'Shepherds Purse': return [0,0,0,0,0,0,0,0,0,1,0,0]

    elif category == 'Small-flowered Cranesbill': return [0,0,0,0,0,0,0,0,0,0,1,0]

    elif category == 'Sugar beet': return [0,0,0,0,0,0,0,0,0,0,0,1] 
import os 

import cv2

from random import shuffle

def create_train_data():

    train=[]

    for category in categories:

        for img in os.listdir(os.path.join(train_dir,category)):

            label=category_to_label(category)

            image_path=os.path.join(train_dir,category,img)

            img=cv2.imread(image_path,1)

            GREEN_MIN = np.array([25, 52, 72],np.uint8)

            GREEN_MAX = np.array([102, 255, 255],np.uint8)

            img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

            img = cv2.inRange(img, GREEN_MIN, GREEN_MAX)

            img=cv2.resize(img,(128,128))

            img=img/255

            

            train.append([np.array(img),label])

    

    shuffle(train)

    return(train)
train_data=create_train_data()
train_data
def create_test_data():

    test=[]

    for img in os.listdir(test_dir):

        img_num = img

        image_path=os.path.join(test_dir,img)

        img=cv2.imread(image_path,1)

        GREEN_MIN = np.array([25, 52, 72],np.uint8)

        GREEN_MAX = np.array([102, 255, 255],np.uint8)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        img = cv2.inRange(img, GREEN_MIN, GREEN_MAX)        

        img=cv2.resize(img,(128,128))

        img=img/255

        

        test.append([np.array(img),img_num])

    shuffle(test)

    return(test)
test_data=create_test_data()
test_data
x_train=np.array([i[0] for i in train_data]).reshape(-1,128,128,1)

y_train=[i[1] for i in train_data]
x_train.shape
y_train=np.vstack(y_train)
y_train
y_train.shape
x_test=np.array([i[0]for i in test_data])

test_image_name=[i[1] for i in test_data]
x_test
test_image_name
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(

    x_train, y_train, test_size=0.15,random_state=42)
print(x_train.shape)
x_valid.shape
import tensorflow.keras as keras

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Activation, add, Add, Dropout, BatchNormalization

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.utils import to_categorical

model=Sequential()



model.add(Conv2D(32, kernel_size=(5, 5), activation='relu',

                 kernel_initializer='he_normal', input_shape=(128, 128, 1)))  

model.add(MaxPooling2D(pool_size=(2, 2)))  

model.add(Conv2D(64, kernel_size=(5, 5), activation='relu',

                 kernel_initializer='he_normal'))  

model.add(MaxPooling2D(pool_size=(2, 2)))  

model.add(Conv2D(32, kernel_size=(5, 5), activation='relu',

                 kernel_initializer='he_normal')  )

model.add(MaxPooling2D(pool_size=(2, 2)))  



model.add(Flatten())  

model.add(Dense(200, activation='relu'))

model.add(Dense(100, activation='relu'))

model.add(Dense(12, activation='softmax'))  



model.compile(

    loss=keras.losses.categorical_crossentropy,

    optimizer='adam',

    metrics=['accuracy']

)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

    width_shift_range=0.2,  # 3.1.1 左右にずらす

    height_shift_range=0.2,  # 3.1.2 上下にずらす

    horizontal_flip=True,  # 3.1.3 左右反転

    # 3.2.1 Global Contrast Normalization (GCN) (Falseに設定しているのでここでは使用していない)

    samplewise_center=False,

    samplewise_std_normalization=False,

    zca_whitening=False)  # 3.2.2 Zero-phase Component Analysis (ZCA) Whitening (Falseに設定しているのでここでは使用していない)
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.preprocessing.image import ImageDataGenerator

early_stopping = EarlyStopping(patience=3, verbose=1)

model.fit_generator(datagen.flow(x_train, y_train, batch_size=50),

                    steps_per_epoch=x_train.shape[0] // 100, epochs=10, validation_data=(x_valid, y_valid),callbacks=[early_stopping])
x_test=x_test.reshape(-1,128,128,1)
x_test.shape
pre=model.predict(x_test)
pre.shape
pre=np.argmax(pre,axis=1)
pre
def label_to_category (label):

    if label == 0: return  'Black-grass'

    elif label == 1: return 'Charlock'

    elif label == 2: return 'Cleavers'

    elif label == 3: return 'Common Chickweed'

    elif label == 4: return 'Common wheat'

    elif label == 5: return 'Fat Hen'

    elif label == 6: return 'Loose Silky-bent'

    elif label == 7: return 'Maize'

    elif label == 8: return 'Scentless Mayweed'

    elif label == 9: return 'Shepherds Purse'

    elif label == 10: return 'Small-flowered Cranesbill'

    elif label == 11: return 'Sugar beet'

    
pred_categories=[]

for label in pre :

    pred_category=label_to_category(label)

    pred_categories.append(pred_category)
pred_categories
sub=pd.read_csv('../input/plant-seedlings-classification/sample_submission.csv')
submission=-pd.DataFrame()

submission['file']=test_image_name

submission['species']=pred_categories
submission

submission.to_csv('submission.csv',index=False)
verify_csv=pd.read_csv('submission.csv')
verify_csv