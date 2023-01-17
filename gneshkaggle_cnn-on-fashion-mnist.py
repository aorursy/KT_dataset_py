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
train_df=pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
test_df=pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')

train_df.shape,test_df.shape
train_df.head()
x_train=train_df.loc[:,train_df.columns != 'label']
y_train=train_df.loc[:,'label']

x_test=test_df.loc[:,test_df.columns != 'label']
y_test=test_df.loc[:,'label']
x_train=np.array(x_train)

x_test=np.array(x_test)
x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)
#rescale the values between 0 and 1
x_train=x_train/255.0
x_test=x_test/255.0
#number of unique labels for classification
print(set(y_train))
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D

model=Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(512,activation='relu'),
    Dense(10,activation='softmax')
    
    
])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
from tensorflow.keras.callbacks import Callback
class mycallback(Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy')>0.99):
            print('Reached 99% accuracy')
            self.model.stop_training=True

callback=mycallback()
            
#change the target variable to categorical
from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train)

y_test=to_categorical(y_test)
hist=model.fit(x_train,y_train,validation_split=0.2,epochs=10,verbose=1,callbacks=[callback])
acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
epochs=range(len(acc))

plt.plot(epochs,acc,'b',label='accuracy')
plt.plot(epochs,val_acc,'r',label='val_accuracy')
plt.legend()
plt.show()
loss=hist.history['loss']
val_loss=hist.history['val_loss']

plt.plot(epochs,loss,'b',label='loss')
plt.plot(epochs,val_loss,'r',label='val_loss')
plt.legend()
plt.show()
predictions=model.predict(x_test)
for i in range(5):
    print('predicted',np.argmax(predictions[i]))
    print('actual',np.argmax(y_test[i]))
    print('---------------')
prediction=[]
labels=[]
for i in range(len(y_test)):
    prediction.append(np.argmax(predictions[i]))
    labels.append(np.argmax(y_test[i]))



data=pd.DataFrame({'predictions':prediction,'labels':labels})
data.head(10)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(data['predictions'],data['labels'])
print(cm)

