# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Convolution2D,BatchNormalization,MaxPooling2D,Flatten,Dense,Dropout

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df=pd.read_csv('../input/train.csv')
test_df=pd.read_csv('../input/test.csv')
X_train_arr=np.asarray(train_df[['pixel'+str(i) for i in range(len(train_df.columns)-1)]])
y_train_arr=np.asarray(train_df['label'])
X_train_arr=np.reshape(X_train_arr,(len(X_train_arr),28,28,1))
y_train_arr=to_categorical(num_classes=10,y=y_train_arr)
input_shape=(28,28,1)
np.shape(y_train_arr)
def createCNNModel():
    model=Sequential()
    model.add(Convolution2D(64,(3,3),input_shape=input_shape,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Convolution2D(32,(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(500,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10,activation='sigmoid'))
    
    return model
    
cnn_model=createCNNModel()
cnn_model.summary()
cnn_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
epochs=5
batch=128
cnn_model.fit(X_train_arr,y_train_arr,batch_size=batch,epochs=epochs,verbose=1)
test_X_arr=np.asarray(test_df)
test_X_arr=test_X_arr.reshape((len(test_X_arr),28,28))
np.shape(test_X_arr)
idx=np.random.randint(len(test_X_arr))
plt.imshow(test_X_arr[idx],cmap=plt.get_cmap('plasma'))
plt.annotate(xy=(0,10),s=str(cnn_model.predict_classes(test_X_arr[idx].reshape((1,28,28,1)))[0]))
plt.show()