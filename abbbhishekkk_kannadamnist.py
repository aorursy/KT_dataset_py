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

import seaborn as sns

import keras

%matplotlib inline
from keras.layers import Conv2D,pooling,Flatten

from keras.layers.pooling import MaxPool2D
data=pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
data.head()
from collections import Counter

Counter(data['label'])
data.shape
x_train=(data.iloc[:,1:].values).astype('float32')

y_train=(data.iloc[:,0].values).astype('int')

y_train=keras.utils.to_categorical(y_train)
x_train=x_train/255.0

import sklearn



from sklearn.model_selection import train_test_split

x_train=x_train.reshape(-1,28,28,1)
X_train,X_test,Y_train,y_test=train_test_split(x_train,y_train)
from keras.models import Sequential

import keras

from keras.layers import Dense

from keras.callbacks.callbacks import EarlyStopping
Modelnew=Sequential()

b=EarlyStopping(patience=3,monitor='val_loss')

from keras.layers import Dropout
Modelnew.add(Conv2D(filters=64,kernel_size=3,activation='relu',input_shape=(28,28,1)))

Modelnew.add(Conv2D(filters=32,kernel_size=3,activation='relu'))

Modelnew.add(Dropout(0.2))

Modelnew.add(Flatten())

Modelnew.add(Dense(10,activation='softmax'))
Modelnew.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
Modelnew.fit(X_train,Y_train,validation_data=(X_test,y_test),epochs=15,batch_size=32,callbacks=[b])
Modelnew.evaluate(X_test,y_test)
pred=Modelnew.predict(X_test)
Y_pred_classes = np.argmax(pred, axis = 1)

Y_Act_Classes=np.argmax(y_test,axis=1)
from sklearn.metrics import confusion_matrix,auc,f1_score,classification_report
confusion_matrix(Y_Act_Classes,Y_pred_classes)
test_data=pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

testdata=test_data.iloc[:,1:]
testdata
test_data=testdata.values.reshape(-1,28,28,1)
predicted_classes = Modelnew.predict_classes(test_data)

submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

submission['label'] = predicted_classes

submission.to_csv('submission.csv', index=False)
