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
### Setting up libraries
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

%matplotlib inline

print("Using ",keras.backend.backend()," backend")
### Setting seed for reproducibility
np.random.seed(123)
### Get Data
train=pd.read_csv("../input/digit-recognizer/train.csv")
test=pd.read_csv("../input/digit-recognizer/test.csv")

train.head()

X_train=train.drop(labels=['label'],axis=1)
y_train=train['label']

print("X_train:",X_train.shape)
print("y_train:",y_train.shape)
print("Test:", test.shape)
### Normalization
X_train_norm,test_norm=X_train/255.0,test/255.0

X_train_norm.head()
### Label encoding
num_classes=10

y_train=to_categorical(y_train,num_classes)

y_train.shape
### Split Training data to train and validation set
X_train,X_val,y_train,y_val=train_test_split(X_train_norm,y_train,test_size=0.2,random_state=123)

print(X_train.shape,X_val.shape,y_train.shape,y_val.shape)
input_cols=X_train.shape[1]
input_cols
### Modelling
# Model
from keras.models import Sequential
from keras.layers import Dense


def Keras_MNIST_model():
    model=Sequential()
    model.add(Dense(512,activation='relu',input_shape=(input_cols,)))
    model.add(Dense(512,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    
    model.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')
    return model
### Fitting data
model=Keras_MNIST_model()


epochs=50

fit=model.fit(X_train,y_train,epochs=epochs,verbose=1,
              batch_size=128,validation_data=(X_val,y_val))
### Plotting the results
plt.subplot(2,1,1)
plt.plot(fit.history['accuracy'])
plt.plot(fit.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train','test'],loc='upper right')

plt.subplot(2,1,2)
plt.plot(fit.history['loss'])
plt.plot(fit.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('Epochs')
plt.legend(['train','test'],loc='upper right')

plt.tight_layout()
### Predict on test data
preds=model.predict(test_norm)
prediction=np.argmax(preds,axis=1)

prediction=pd.Series(prediction,name='label')
prediction.head()

prediction.shape[0]
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),prediction],axis = 1)

submission.to_csv("keras_mnist_20200816.csv",index=False)
