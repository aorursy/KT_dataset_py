# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
train=pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
test=pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
train.head()
print(train.shape)
print(test.shape)
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
img_height,img_widht=28,28
input_shape=(img_height,img_widht,1)
X=np.array(train.iloc[:,1:])
y=to_categorical(train.iloc[:,0])
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=101)
X_test=np.array(test.iloc[:,1:])
y_test=to_categorical(test.iloc[:,0])

X_train=X_train.reshape(X_train.shape[0],img_height,img_widht,1)
X_test=X_test.reshape(X_test.shape[0],img_height,img_widht,1)
X_val=X_val.reshape(X_val.shape[0],img_height,img_widht,1)

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_val=X_val.astype('float32')

X_train=X_train/255
X_test=X_test/255
X_val=X_val/255
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.layers.normalization import BatchNormalization

batch_size=128
num_classes=10
epochs=50

model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation="relu",kernel_initializer="he_normal",input_shape=input_shape))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,kernel_size=(3,3),activation="relu",kernel_initializer="he_normal"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,kernel_size=(3,3),activation="relu",kernel_initializer="he_normal"))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(128,activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(num_classes,activation="softmax"))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.summary()
history=model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_val,y_val))
score=model.evaluate(X_test,y_test,verbose=1)
print("Test loss: ",score[0])
print("Test Accuracy: ",score[1])
accuracy=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(len(loss))

plt.plot(epochs,accuracy,label="Training accuracy")
plt.plot(epochs,val_acc,label="Validation accuracy")
plt.title("Training and Validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs,loss,label="Training loss")
plt.plot(epochs,val_loss,label="Validation loss")
plt.title("Training and Validation loss")
plt.legend()
pred=model.predict(X_test)
predicted_classes=model.predict_classes(X_test)
y_true=test.iloc[:,0]
correct = np.nonzero(predicted_classes==y_true)[0]
incorrect = np.nonzero(predicted_classes!=y_true)[0]
from sklearn.metrics import classification_report

print(classification_report(y_true,predicted_classes))
plt.imshow(X_test[1].reshape(28,28),interpolation=None)
predicted_classes[1]
