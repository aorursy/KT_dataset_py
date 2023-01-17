# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import cv2
import os
file=open("../input/digit-recognizer/train.csv")
data_train=pd.read_csv(file)
y_train=np.array(data_train.iloc[:,0])
x_train=np.array(data_train.iloc[:,1:])
file=open("../input/digit-recognizer/test.csv")
data_test=pd.read_csv(file)
x_test=np.array(data_test)
print(x_train.shape,y_train.shape,x_test.shape)
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def show_img(x):
    plt.figure(figsize=(8,7))
    if x.shape[0]>100:
        print(x.shape[0])
        n_imgs=16
        n_samples=x.shape[0]
        x=x.reshape(n_samples,28,28)
        for i in range(16):
            plt.subplot(4,4,i+1)
            plt.imshow(x[i])
        plt.show()
    else:
        plt.imshow(x)
        plt.show()
show_img(x_train)
x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)
input_shape=(28,28,1)
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train/=255
x_test/=255
print(x_train.shape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout
model=Sequential()
model.add(Conv2D(28,kernel_size=(3,3),input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x=x_train,y=y_train,epochs=10)
model.evaluate(x_test)
image_index=4444
plt.imshow(x_test[image_index].reshape(28,28),cmap="Greys")
pred=model.predict(x_test[image_index].reshape(1,28,28,1))
print(pred.argmax())