import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Conv2D,MaxPooling2D,BatchNormalization,Dropout,ZeroPadding2D

mnist_train=pd.read_csv('../input/digit-recognizer/train.csv')
inputs_test=pd.read_csv('../input/digit-recognizer/test.csv')
mnist_train.head()
mnist_train.shape
inputs_train=mnist_train.drop('label',axis=1)
target_train=mnist_train['label']
inputs_train.head()
target_train.head()
type(inputs_train)
type(target_train)
inputs_train=np.array(inputs_train)
target_train=np.array(target_train)
inputs_test=np.array(inputs_test)
figure,subplot1=plt.subplots(10,5)
figure.set_figheight(30)
figure.set_figwidth(15)
curr_num=0
num_count=0
sub1=0
sub2=0
for x in range(len(target_train)):
    if target_train[x]==curr_num:
        subplot1[sub1][sub2].imshow(inputs_train[x].reshape(28,28),cmap=plt.cm.binary)
        sub2+=1
        num_count+=1
        if num_count==5:
            if curr_num==9:
                break
            else:
                curr_num+=1
                num_count=0
    if sub2==5:
        sub2=0
        sub1+=1
    if sub1==10:
        break
scalar=MinMaxScaler(feature_range=(0,1))
inputs_train=scalar.fit_transform(inputs_train)
inputs_test=scalar.fit_transform(inputs_test)
inputs_train.shape
inputs_test.shape
inputs_train=inputs_train.reshape(inputs_train.shape[0],28,28,1)
inputs_test=inputs_test.reshape(inputs_test.shape[0],28,28,1)
model=Sequential()
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Dropout(0.2))
model.add(Conv2D(100,(5,5),input_shape=inputs_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Dropout(0.2))
model.add(Conv2D(100,(5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Dropout(0.2))
model.add(Conv2D(100,(5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(inputs_train,target_train,batch_size=100,epochs=10,validation_split=0.1)
pred=model.predict_classes(inputs_test)
my_submission=pd.DataFrame({'ImageId':[],'Label':[]},dtype=int)
for x in range(len(pred)):
    my_submission=my_submission.append({'ImageId':x+1,'Label':pred[x]},ignore_index=True)
my_submission.head()
my_submission.to_csv('my_submission.csv',index=False)
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
digitlst=['digit0.png','digit1.png','digit2.png','digit3.png','digit4.png','digit5.png','digit6.png','digit7.png','digit8.png','digit9.png']
predarr=np.empty(0,dtype='int8')
figure,subplot=plt.subplots(2,5)
sub1=0
sub2=0
count=0

for y in digitlst:
    path='../input/digittestfreehand/'+y
    digit=plt.imread(path)
    digit=rgb2gray(digit)
    subplot[sub1][sub2].imshow(digit,cmap=plt.cm.binary)
    zeros=0
    ones=0
    digit=digit.reshape(784)
    for x in range(len(digit)):
        if digit[x]==0:
            zeros+=1
            digit[x]=0.999
        elif x>0.999:
            ones+=1
            digit[x]=0
    digit=digit.reshape(1,28,28,1)
    pred=model.predict_classes(digit)
    predarr=np.append(predarr,pred)
    sub2+=1
    if sub2%5==0:
        sub1+=1
        sub2=0
for x in np.nditer(predarr):
    print(x)