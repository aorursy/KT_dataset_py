import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
dire='../input/gender-recognition-200k-images-celeba/Dataset/Test'
cat=['Female','Male']
size=90
data=[]
for cate in cat:
    fold=os.path.join(dire,cate)
    label=cat.index(cate)
    for img in os.listdir(fold):
        img_path=os.path.join(fold,img)
        img_arr=cv2.imread(img_path)
        img_arr=cv2.resize(img_arr,(size,size))
        data.append([img_arr,label])
len(data)

random.shuffle(data)
x=[]
y=[]

for features,label in data:
    x.append(features)
    y.append(label)
x=np.array(x)
y=np.array(y)

x=x/255
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.callbacks import TensorBoard
model=Sequential()
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128,input_shape=x.shape[1:],activation='relu'))

model.add(Dense(2,activation='softmax'))
model.compile(optimizer="adam",loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x,y,epochs=5,validation_split=0.1,batch_size=32)