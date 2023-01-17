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
        #print(os.path.join(dirname, filename))
        pass

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from random import shuffle
IMG_SIZE=224
path='/kaggle/input/gender-classification-dataset/'

training=[]
test=[]

train_limit=1800
test_limit=100
counter=0

for i in os.listdir(path+'Training/'+'female/'):
    if counter>train_limit:
        counter=0
        break
    counter+=1
    img=cv2.imread(path+'Training/'+'female/'+i)
    img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    training.append(tuple([img,0]))
    
for i in os.listdir(path+'Training'+'/male/'):
    if counter>train_limit:
        counter=0
        break
    counter+=1
    img=cv2.imread(path+'Training'+'/male/'+i)
    img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    training.append(tuple([img,1]))

for i in os.listdir(path+'Validation'+'/female/'):
    if counter>test_limit:
        counter=0
        break
    counter+=1
    img=cv2.imread(path+'Validation'+'/female/'+i)
    img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    test.append(tuple([img,0]))
for i in os.listdir(path+'Validation'+'/male/'):
    if counter>test_limit:
        counter=0
        break
    counter+=1
    img=cv2.imread(path+'/'+'Validation'+'/male/'+i)
    img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    test.append(tuple([img,1]))

shuffle(training)
shuffle(test)







    
X=[]
Y=[]



for features,labels in training:
    X.append(features)
    Y.append(labels)

del training
X_t=[]
Y_t=[]

for features,labels in test:
    X_t.append(features)
    Y_t.append(labels)

del test

X=np.array(X)#.reshape(-1,IMG_SIZE,IMG_SIZE,3)
#X=tf.keras.utils.normalize(X)
Y=np.array(Y)

plt.imshow(X[0])
print(Y[0])
X_t=np.array(X_t)#.reshape(-1,IMG_SIZE,IMG_SIZE,3)
#X_t=tf.keras.utils.normalize(X_t)
Y_t=np.array(Y_t)



plt.imshow(X_t[0])
print(Y_t[0])

class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy') > 0.95 or abs(logs.get('accuracy')-logs.get('val_accuracy'))>=4):
            self.model.stop_training = True

callback=myCallBack()

model=tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(64,(3,3), input_shape=X.shape[1:],activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2,2))

model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2,2))

model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2,2))

model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2,2))

model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2,2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256,
                                activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.01)))
#model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(256,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)))
#model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(256,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Dense(2,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])



model.fit(X,Y,epochs=50,validation_data=(X_t,Y_t),callbacks=[callback])
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TF Lite model.
with tf.io.gfile.GFile('/kaggle/working/model.tflite', 'wb') as f:
    f.write(tflite_model)
os.path.getsize('/kaggle/working/model.tflite')/1024/1024
