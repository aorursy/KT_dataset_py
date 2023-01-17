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
#import Library
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras import models
from keras.layers import Conv2D,Input,MaxPool2D,BatchNormalization,Dense,Add,Activation,Flatten
import numpy as np
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")
#preparing and load data,seperate data
data_train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
data_test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

train_df=data_train.copy()
test_df=data_test.copy()

print("train_df shape :",train_df.shape)
print("test_df shape :",test_df.shape)
Y=np.array(train_df["label"])
X=np.array(train_df.drop(["label"],axis=1))

print("X shape: ", X.shape)
print("Y shape: ", Y.shape)
f,ax=plt.subplots(1,6,figsize=(20,10))
index=5
for i in range(0,6):
    digit=X[index]
    label=Y[index]
    digit=np.reshape(digit,(28,28))
    ax[i].imshow(digit,plt.cm.binary)
    ax[i].set_title("label:{}".format(label))
    index+=15
#prepare data for train and test model
train_images=X.reshape((42000,28,28,1))
train_images=train_images.astype("float32")/255

train_labels=to_categorical(Y)

test_images=np.array(test_df).reshape((28000,28,28,1))
test_images=test_images.astype("float32")/255
X_train,X_test,y_train,y_test=train_test_split(train_images,train_labels,test_size=0.15,random_state=42)

print("X_train: ", X_train.shape)
print(" X_test: ", X_test.shape)
print("y_train: ", y_train.shape)
print(" y_test: ", y_test.shape)
#create model
model=models.Sequential()
model.add(Conv2D(128,(5,5),activation="relu",input_shape=(28,28,1)))
model.add(MaxPool2D((2,2),strides=2))
model.add(Conv2D(256,(3,3),activation="relu"))
model.add(Conv2D(64,(5,5),activation="relu"))
model.add(MaxPool2D((2,2),strides=2))
model.add(Conv2D(32,(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dense(512,activation="relu"))
model.add(Dense(10,activation="softmax"))
#show model
model.summary()
#compile model
model.compile(optimizer="rmsprop",loss="categorical_crossentropy",
              metrics=["accuracy"])
history=model.fit(X_train,y_train,epochs=12,batch_size=32)
#if you want to save model you run this code
#model.save('my_model_digit.h5')
test_loss,test_acc=model.evaluate(X_test,y_test)
print("test_acc: ",test_acc)
print("test_loss",test_loss)
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,5))
ax1.plot(history.history['accuracy'])
ax1.set_title('model accuracy')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'test'], loc='upper left')

ax2.plot(history.history['loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'test'], loc='upper left');

f,ax=plt.subplots(1,5,figsize=(20,10))
index=100
for i in range(0,5):
    digit=X_test[index]
    label=np.argmax(y_test[index])
    score=model.predict_proba(digit.reshape(1,28,28,1)).max()
    predict=model.predict_classes(digit.reshape(1,28,28,1))
    ax[i].imshow(digit.reshape(28,28),plt.cm.binary)
    ax[i].set_title("label:{}\npredict:{},score:{:.2f}".format(label,predict,score))
    index+=1
f,ax=plt.subplots(1,6,figsize=(20,10))
index=5000
for i in range(0,6):
    digit=X_test[index]
    label=np.argmax(y_test[index])
    score=model.predict_proba(digit.reshape(1,28,28,1)).max()
    predict=model.predict_classes(digit.reshape(1,28,28,1))
    ax[i].imshow(digit.reshape(28,28),plt.cm.binary)
    ax[i].set_title("label:{}\npredict:{},score:{}".format(label,predict,score))
    index+=1
f,ax=plt.subplots(1,6,figsize=(20,10))
index=2658
for i in range(0,6):
    digit=X_test[index]
    label=np.argmax(y_test[index])
    score=model.predict_proba(digit.reshape(1,28,28,1)).max()
    predict=model.predict_classes(digit.reshape(1,28,28,1))
    ax[i].imshow(digit.reshape(28,28),plt.cm.binary)
    ax[i].set_title("label:{}\npredict:{},score:{}".format(label,predict,score))
    index+=5
x_pred=model.predict(test_images)
result=np.argmax(x_pred,axis=1)

result = pd.Series(result,name="Label")
df=pd.concat([pd.Series(range(1,28001),name = "ImageId"),result],axis = 1)
df.to_csv("predict_test.csv",index=False)
