# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('../input/face-mask-detection-dataset/train.csv')
data.head()
X=data.iloc[:,1:5].values
Y=data.iloc[:,5].values
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)
X
data.drop(['name'],axis=1)
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

e=LabelEncoder()
e.fit(Y)
e_Y=e.transform(Y)
enc_Y=np_utils.to_categorical(e_Y)
print(e_Y)
dict={}
for i in range(0,20) :
    for j in range(0,len(e_Y)) :
        if e_Y[j]==i :
            dict[i]=Y[j]
            break
dict
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#X=X.tolist()
Y=enc_Y.tolist()

X,Y=shuffle(X,enc_Y)
x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.90)
x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)
print(x_train.size/4,' ',x_test.size/4,' ',y_train.size/20,' ',y_test.size/20)
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(10, activation='sigmoid', kernel_initializer='random_normal', input_dim=4))
#model.add(BatchNormalization)
#model.add(Dense(12, activation='relu', kernel_initializer='random_normal'))
#model.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
model.add(Dense(20, activation='sigmoid', kernel_initializer='random_normal'))

opt=SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(x_train, y_train,validation_data = (x_test,y_test), epochs=200, batch_size=32)
from sklearn.metrics import accuracy_score
from keras.layers import Softmax

tmodel=Sequential([model,Softmax()])
y_pred=tmodel.predict(x_test)
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))
print(accuracy_score(pred,test))
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()
test_data=pd.read_csv('../input/face-mask-detection-dataset/submission.csv')
test_data.head()
