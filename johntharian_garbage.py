import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
import random
import seaborn as sns
from sklearn.model_selection import train_test_split
categories=['cardboard','glass','metal','paper','plastic','trash']
data_dir='../input/garbage-classification/Garbage classification/Garbage classification'
data_x=[]
data_y=[]
data=[]
IMG_SIZE = 100

for category in categories:
    path=os.path.join(data_dir,category)
    class_num=categories.index(category)
    
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img))
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        data_x.append(new_array)
        data_y.append(class_num)
#         data.append([img_array,class_num])
len(data_x)
len(data_y)
len(data)

temp = list(zip(data_x,data_y)) 
random.shuffle(temp) 
X,y = zip(*temp) 
X=np.asarray(X,dtype=None)
y=np.asarray(y,dtype=None)
X.shape
y.shape
plt.imshow(X[1])
y[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train[1].shape
print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))
i=1005
plt.imshow(X_train[i])
print(y_train[i])
W_grid=15
L_grid=15

fig,axes=plt.subplots(L_grid,W_grid,figsize=(25,25))
axes=axes.ravel()

n_training=len(X_train)

for i in np.arange(0,L_grid*W_grid):
    index=np.random.randint(0,n_training)
    axes[i].imshow(X_train[index])
    axes[i].set_title(y_train[index])
    axes[i].axis('off')
    
plt.subplots_adjust(hspace=0.4)
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
num_class=6
y_train
y_test
y_train=keras.utils.to_categorical(y_train,num_class)
y_test=keras.utils.to_categorical(y_test,num_class)
y_test
'''normalise the data by dividing with 255'''
X_train=X_train/255
X_test=X_test/255
X_train
'''we need input shape without the size'''
X_train.shape
Input_shape=X_train.shape[1:]
Input_shape
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
cnn_model=Sequential()
cnn_model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',
                     input_shape=Input_shape))
cnn_model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Dropout(0.3))


# cnn_model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
# cnn_model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
# cnn_model.add(MaxPooling2D(2,2))
# cnn_model.add(Dropout(0.2))


cnn_model.add(Flatten())

cnn_model.add(Dense(units=512,activation='relu'))

cnn_model.add(Dense(units=512,activation='relu'))

cnn_model.add(Dense(units=6,activation='softmax')) #output layer i.e 10 units
cnn_model.compile(loss='categorical_crossentropy',
                   optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
                  metrics=['accuracy'])
history=cnn_model.fit(X_train,y_train,batch_size=32,epochs=10,shuffle=True)
evaluation=cnn_model.evaluate(X_test,y_test)

print(f"Test accuracy:{evaluation}")
predicted_class=cnn_model.predict_classes(X_test)
predicted_class
y_test
y_test=y_test.argmax(1)
y_test
def class_convert(classess):
    pred=[]
    for i in classess:
        if i ==0:
            pred.append('Cardboard')
        elif i==1:
            pred.append('Glass')
        elif i==2:
            pred.append('Metal')
        elif i==3:
            pred.append('Paper')
        elif i==4:
            pred.append('Plastic')
        elif i==5:
            pred.append('Trash')
    return pred
pred_class=class_convert(predicted_class)
y_class=class_convert(y_test)
L=7
W=7
fig,axes=plt.subplots(L,W,figsize=(17,17))
axes=axes.ravel()

for i in np.arange(0,L*W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('Prediction={}\n True={}'.format(pred_class[i],y_class[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace=2,hspace=2)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,predicted_class)
cm
corr=0
false=0
for i in range(len(y_test)):
    if y_test[i]==predicted_class[i]:
         corr=corr+1
    else:
        false=false+1
print("Correct:",corr)
print("False",false)
import os

directory=os.path.join(os.getcwd(),'saved_models')

if not os.path.isdir(directory):
    os.makedirs(directory)

model_path=os.path.join(directory,'keras_cifar10_1.h5')
cnn_model.save(model_path)