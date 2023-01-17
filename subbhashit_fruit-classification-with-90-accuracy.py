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
import cv2
import os 
from PIL import Image
images = []       
labels = [] 
train_path = '../input/fruit-images-for-object-detection/train_zip/train'
for filename in os.listdir('../input/fruit-images-for-object-detection/train_zip/train'):
    if filename.split('.')[1] == 'jpg':
        img = cv2.imread(os.path.join(train_path,filename))
        ary=Image.fromarray(img,'RGB')
        r=ary.resize((50,50))
        labels.append(filename.split('_')[0])
        images.append(np.array(r))
np.unique(labels)
from sklearn.preprocessing import LabelEncoder
la=LabelEncoder()
labels=pd.DataFrame(labels)
labels=la.fit_transform(labels[0])
labels
import matplotlib.pyplot as plt
figure=plt.figure(figsize=(15,10))
ax=figure.add_subplot(121)
ax.imshow(images[0])
bx=figure.add_subplot(122)
bx.imshow(images[60])
plt.show()
images=np.array(images)
np.save("image",images)
np.save("labels",labels)
image=np.load("image.npy",allow_pickle=True)
labels=np.load("labels.npy",allow_pickle=True)
s=np.arange(image.shape[0])
np.random.shuffle(s)
image=image[s]
labels=labels[s]
num_classes=len(np.unique(labels))
len_data=len(image)
x_train,x_test=image[(int)(0.1*len_data):],image[:(int)(0.1*len_data)]
y_train,y_test=labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]
import keras
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten,MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.layers import Activation, Convolution2D, Dropout, Conv2D,AveragePooling2D, BatchNormalization,Flatten,GlobalAveragePooling2D
from keras import layers
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
l2_reg=0.001
opt=Adam(lr=0.001)
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2,2), input_shape=(50,50, 3), activation='relu',kernel_regularizer=l2(l2_reg)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu',kernel_regularizer=l2(l2_reg)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(2,2), activation='relu',kernel_regularizer=l2(l2_reg)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.summary()
filepath="weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
history=model.fit(x_train,y_train,batch_size=128,epochs=110,verbose=1,validation_split=0.33,callbacks=[checkpoint])
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
figure=plt.figure(figsize=(15,15))
ax=figure.add_subplot(121)
ax.plot(history.history['accuracy'])
ax.plot(history.history['val_accuracy'])
ax.legend(['Training Accuracy','Val Accuracy'])
bx=figure.add_subplot(122)
bx.plot(history.history['loss'])
bx.plot(history.history['val_loss'])
bx.legend(['Training Loss','Val Loss'])
test_path = '../input/fruit-images-for-object-detection/test_zip/test'
t_labels=[]
t_images=[]
for filename in os.listdir('../input/fruit-images-for-object-detection/test_zip/test'):
    if filename.split('.')[1] == 'jpg':
        img = cv2.imread(os.path.join(test_path,filename))
        ary=Image.fromarray(img,'RGB')
        r=ary.resize((50,50))
        t_labels.append(filename.split('_')[0])
        t_images.append(np.array(r))
t_images=np.array(t_images)
np.save("t_image",t_images)
t_image=np.load("image.npy",allow_pickle=True)
pred=np.argmax(model.predict(t_image),axis=1)
prediction = la.inverse_transform(pred)
t_image=np.expand_dims(t_images[10],axis=0)
pred_t=np.argmax(model.predict(t_image),axis=1)
prediction_t = la.inverse_transform(pred_t)
print(prediction_t[0])
plt.imshow(t_images[10])