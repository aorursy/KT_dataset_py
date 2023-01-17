# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
img_np = np.load('../input/img_np.npy')
y_np = np.load('../input/y_np.npy')
shuffle = np.random.permutation(y_np.shape[0])
X = img_np[shuffle]
y = y_np[shuffle]
y[y=='daisy'] = 0
y[y=='dandelion'] = 1
y[y=='rose'] = 2
y[y=='sunflower'] = 3
y[y=='tulip'] = 4
y = y.astype('int')
y_one = np.zeros([y.shape[0],5])
for i,c in enumerate(y):
    for j in range(5):
        if c==j:
            y_one[i,j] = 1
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y_one,train_size=0.9)
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=180,  
            zoom_range = 0.1,  
            width_shift_range=0.2,  
            height_shift_range=0.2, 
            horizontal_flip=True, 
            vertical_flip=True,
            )
datagen.fit(X_train)
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Dropout,Activation,BatchNormalization ,Flatten,MaxPool2D
model = Sequential()
model.add(Conv2D(8,(3,3),padding='same',activation='relu',input_shape=(120,120,3)))
model.add(MaxPool2D(2))
model.add(BatchNormalization(axis=-1))

model.add(Dropout(0.2))

model.add(Conv2D(16,(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(2))
model.add(BatchNormalization(axis=-1))

model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
#model.add(Dense(64,activation='relu'))
model.add(Dense(5,activation='softmax'))
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
hist = model.fit_generator(datagen.flow(X_train,y_train, batch_size=1024),steps_per_epoch=25,epochs=10,verbose=1)
model.evaluate(X_train,y_train)
model.evaluate(X_test,y_test)
model.save('ass5.h5')
from IPython.display import FileLink, FileLinks
FileLinks('.') 
from skimage import color
X_train_gr = np.zeros([X_train.shape[0],120,120])
X_test_gr = np.zeros([X_test.shape[0],120,120])
for i in range(X_train.shape[0]):
    X_train_gr[i] = color.rgb2gray(X_train[i])
for i in range(X_test.shape[0]):
    X_test_gr[i] = color.rgb2gray(X_test[i])
model_gr = Sequential()
model_gr.add(Conv2D(6,(3,3),padding='valid',activation='relu',input_shape=(120,120,1)))
model_gr.add(BatchNormalization(axis=-1))
model_gr.add(Conv2D(8,(3,3),padding='valid',activation='relu'))
model_gr.add(BatchNormalization(axis=-1))
model_gr.add(Conv2D(10,(3,3),padding='valid',activation='relu'))
model_gr.add(BatchNormalization(axis=-1))
model_gr.add(MaxPool2D(2))
model_gr.add(Flatten())
model_gr.add(Dense(256,activation='relu'))
model_gr.add(Dense(64,activation='relu'))
model_gr.add(Dense(5,activation='softmax'))
model_gr.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
hist_gr = model_gr.fit(X_train_gr.reshape(-1,120,120,1),y_train,epochs=10,batch_size=64,verbose=1,validation_split=0.1)
model_gr.evaluate(X_train_gr.reshape(-1,120,120,1),y_train)
model_gr.evaluate(X_test_gr.reshape(-1,120,120,1),y_test)
plt.subplot(121)
plt.title('Traiining Accuracy')
plt.bar(x=['Model 1','Model 2'],height=[0.9511568123393316, 0.9439588689405691],color=['b','g'])
plt.subplot(122)
plt.title('Test Accuracy')
plt.bar(x=['Model 1','Model 2'],height=[ 0.6050808323723614,0.4757505777113432],color=['b','g'])
plt.show()
for i in range(5):
    plt.imshow(X_test[i])
    pred = np.argmax(model.predict(X_test[i].reshape(1,120,120,3)))
    plt.title('Flower = '+str(pred))
    plt.show()
np.unique(y_np)
hist.history.keys()
loss_rgb = hist.history['loss']
epoch_rgb = list(range(len(loss_rgb)))
loss_gr = hist_gr.history['loss']

plt.ylabel('loss ')
plt.xlabel('epoch')
plt.plot(epoch_rgb,loss_rgb,label='rgb model')
plt.plot(epoch_rgb,loss_gr,label='gray model')
plt.legend()
plt.show()
acc_rgb = hist.history['acc']
epoch_rgb = list(range(len(loss_rgb)))
acc_gr = hist_gr.history['acc']

plt.ylabel('accuracy ')
plt.xlabel('epoch')
plt.plot(epoch_rgb,acc_rgb,label='rgb model')
plt.plot(epoch_rgb,acc_gr,label='gray model')
plt.legend()
plt.show()
loss_rgb = hist.history['val_acc']
epoch_rgb = list(range(len(loss_rgb)))
loss_gr = hist_gr.history['val_acc']

plt.ylabel('validation loss ')
plt.xlabel('epoch')
plt.plot(epoch_rgb,loss_rgb,label='rgb model')
plt.plot(epoch_rgb,loss_gr,label='gray model')
plt.legend()
plt.show()
