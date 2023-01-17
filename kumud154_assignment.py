import numpy as np 
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style="whitegrid")
import os
import glob as gb
import cv2
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
path = '../input/flowers-recognition/flowers/flowers/'
for folder in os.listdir(path):
    print(folder)

def shapeCount (Gpath):
    size = []
    pathname = str(Gpath+'*/*.jpg')
    imgsDir = gb.glob(pathname= pathname)
    for imgDir in tqdm(imgsDir):
        img = plt.imread(imgDir)
        size.append(img.shape)
    print("number of Imges = ",len(size))
    return pd.Series(size).value_counts()
print (shapeCount(path))
I_SIZE = 200
classes = ['daisy','sunflower','tulip','rose','dandelion']
def reArangeData(data):
    # shuffle
    import random
    random.shuffle(data)
    
   
    X = []
    y = []
    for img,lable in data:
        X.append(img)
        y.append(lable)
        
   
    return np.array(X),np.array(y)


def loadImages (path):
    Dlist = []
    for folder in os.listdir(path):
        pathname = str(path +folder+'/*.jpg')
        files = gb.glob(pathname= pathname)
        for file in tqdm(files):
            image = cv2.imread(file ,cv2.IMREAD_COLOR )
           
            image_array = cv2.resize(image , (I_SIZE ,I_SIZE))
            Dlist.append( [image_array , classes.index(folder)] )
    print(len(Dlist))
    return reArangeData(Dlist)
X , y = loadImages(path)
X=X/255
plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(X[i])
    plt.axis('off')
    plt.title(classes[y[i]])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
print('X_train shape is : ',X_train.shape)
print('y_train shape is : ',y_train.shape)
print("---------------------------------------------")
print('X_test shape id : ',X_test.shape)
print('y_test shape id : ',y_test.shape)
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False, 
        featurewise_std_normalization=False, 
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=10,  
        zoom_range = 0.1, 
        width_shift_range=0.1,  
        height_shift_range=0.1, 
        horizontal_flip=False,  
        vertical_flip=False) 

datagen.fit(X_train)
relu = tf.nn.relu
softmax = tf.nn.softmax
input_shape = (I_SIZE , I_SIZE,3)

model = keras.models.Sequential([
    keras.layers.Conv2D(128,kernel_size=(3,3) ,activation=relu , input_shape = input_shape ),
    keras.layers.MaxPool2D(4,4),
    keras.layers.Dropout(rate=0.3),
    keras.layers.Conv2D(512,kernel_size=(3,3),activation=relu),
    
    keras.layers.MaxPool2D(4,4),
    keras.layers.Conv2D(128,kernel_size=(3,3),activation=relu),
    keras.layers.Dropout(rate=0.3),
    keras.layers.Conv2D(64,kernel_size=(3,3),activation=relu),
   
    keras.layers.Flatten(),
    keras.layers.Dense(512,activation=relu),
    keras.layers.Dense(128,activation=relu),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(64,activation=relu),
    keras.layers.Dense(5,activation=softmax)
])
loss = keras.losses.sparse_categorical_crossentropy
adam = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=adam,
             loss = loss ,
             metrics=['accuracy'])

print(model.summary())
epochs = 70
batch_size = 32
train_img_gen = datagen.flow(X_train, y_train, batch_size=batch_size)
test_image_gen = datagen.flow(X_test, y_test, batch_size=batch_size)
ThisModel = model.fit_generator(train_img_gen,
                                verbose=1,
                                steps_per_epoch=len(X_train) / 32,
                                validation_data=test_image_gen,
                                epochs=epochs
                               )
Err,Acc = model.evaluate(X_test,y_test)
print("Err : " , Err)
print("Acc : ", Acc)
plt.plot(ThisModel.history['accuracy'])
plt.plot(ThisModel.history['loss'])
plt.plot(ThisModel.history['val_accuracy'])
plt.plot(ThisModel.history['val_loss'])

y_predict = model.predict_classes(X_test)
print(y_predict[1] , y_test[1])
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))
              
plt.figure(figsize=(10,10))
st = 10
for i in range(st,st+16):
    plt.subplot(4,4,i+1-st)
    plt.imshow(X_test[i])
    plt.title("True : {} , Predict : {}".format(y_test[i] , y_predict[i]))
    plt.axis('off')
model.save('model.h5')