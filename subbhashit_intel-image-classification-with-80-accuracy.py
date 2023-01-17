# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/intel-image-classification/seg_train/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#buildings,street,mountain,glacier,sea
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn
import glob
image=[]
label=[]
for i in glob.glob("/kaggle/input/intel-image-classification/seg_train/seg_train/buildings/*.jpg"):
    img=cv2.imread(i)
    ar=Image.fromarray(img,'RGB')
    r=ar.resize((50,50))
    image.append(np.array(r))
    label.append('buildings')
for i in glob.glob("/kaggle/input/intel-image-classification/seg_train/seg_train/street/*.jpg"):
    img=cv2.imread(i)
    ar=Image.fromarray(img,'RGB')
    r=ar.resize((50,50))
    image.append(np.array(r))
    label.append('street')
for i in glob.glob("/kaggle/input/intel-image-classification/seg_train/seg_train/mountain/*.jpg"):
    img=cv2.imread(i)
    ar=Image.fromarray(img,'RGB')
    r=ar.resize((50,50))
    image.append(np.array(r))
    label.append('mountain')
for i in glob.glob("/kaggle/input/intel-image-classification/seg_train/seg_train/glacier/*.jpg"):
    img=cv2.imread(i)
    ar=Image.fromarray(img,'RGB')
    r=ar.resize((50,50))
    image.append(np.array(r))
    label.append('glacier')
for i in glob.glob("/kaggle/input/intel-image-classification/seg_train/seg_train/sea/*.jpg"):
    img=cv2.imread(i)
    ar=Image.fromarray(img,'RGB')
    r=ar.resize((50,50))
    image.append(np.array(r))
    label.append('sea')
for i in glob.glob("/kaggle/input/intel-image-classification/seg_train/seg_train/forest/*.jpg"):
    img=cv2.imread(i)
    ar=Image.fromarray(img,'RGB')
    r=ar.resize((50,50))
    image.append(np.array(r))
    label.append('forest')
len(image)
import matplotlib.pyplot as plt
figure=plt.figure(figsize=(15,10))
ax=figure.add_subplot(121)
ax.imshow(image[0])
bx=figure.add_subplot(122)
bx.imshow(image[60])
plt.show()
label=pd.DataFrame(label)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
labels=le.fit_transform(label[0])
labels
unlabels=np.unique(labels)
unlabels
image=np.array(image)
np.save("image",image)
np.save("labels",labels)
image=np.load("image.npy")
labels=np.load("labels.npy")
s=np.arange(image.shape[0])
np.random.shuffle(s)
image=image[s]
labels=labels[s]
num_classes=len(np.unique(labels))
len_data=len(image)
x_train,x_test=image[(int)(0.1*len_data):],image[:(int)(0.1*len_data)]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
train_len=len(x_train)
test_len=len(x_test)
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
model.add(Dropout(0.1))
model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu',kernel_regularizer=l2(l2_reg)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(filters=128, kernel_size=(2,2), activation='relu',kernel_regularizer=l2(l2_reg)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.summary()
filepath="weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
history=model.fit(x_train,y_train,batch_size=128,epochs=20,verbose=1,validation_split=0.33,callbacks=[checkpoint])
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
test=[]
for i in glob.glob("/kaggle/input/intel-image-classification/seg_test/seg_test/*/*.jpg"):
    img=cv2.imread(i)
    ar=Image.fromarray(img,'RGB')
    r=ar.resize((50,50))
    test.append(np.array(r))
len(test)
data1=np.array(test)
np.save("image1",data1)
image1=np.load("image1.npy")
pred=np.argmax(model.predict(image1),axis=1)
prediction = le.inverse_transform(pred)
t_image=np.expand_dims(image1[100],axis=0)
pred_t=np.argmax(model.predict(t_image),axis=1)
prediction_t = le.inverse_transform(pred_t)
print(prediction_t[0])
plt.imshow(image1[100])