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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train_image=pd.read_csv('/kaggle/input/ahcd1/csvTrainImages 13440x1024.csv')
test_image=pd.read_csv('/kaggle/input/ahcd1/csvTestImages 3360x1024.csv')
train_lable=pd.read_csv('/kaggle/input/ahcd1/csvTrainLabel 13440x1.csv')
test_lable=pd.read_csv('/kaggle/input/ahcd1/csvTestLabel 3360x1.csv')
print(train_image.head(1).shape)
print(train_image.head(1))
print(train_lable.iloc[100,:])
x_train=np.array(train_image)
x_test=np.array(test_image)
print(train_image.shape)
print(x_train.shape)
print(x_test.shape)
x_train=x_train/255.0
x_test=x_test/255.0
from keras.utils import to_categorical
y_train=to_categorical(train_lable)
y_test=to_categorical(test_lable)
plt.imshow(x_train[30].reshape(32,32))
print(y_train[30])
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Dropout, Input
from tensorflow.keras.activations import sigmoid, relu
input_dim,output_dim,encode_dim=1024,1024,128
input_layer=Input(shape=(1024),name='input')
layer1=Dense(512,activation='relu')(input_layer)
encode=Dense(256,activation='relu',name='encode')(layer1)
layer2=Dense(512,activation='relu')(encode)
output_layer=Dense(1024,activation='sigmoid',name='output')(layer2)
from tensorflow.keras.optimizers import Adam
model=Model(input_layer,output_layer)
model.compile(optimizer='adam',loss='binary_crossentropy')
model.summary()
model.fit(x_train,x_train,batch_size=128,epochs=20,validation_data=(x_test,x_test))
encode_model=Model(inputs=model.input,outputs=model.get_layer('encode').output)
new_xtrain=encode_model.predict(x_train)
new_xtest=encode_model.predict(x_test)
print(new_xtrain.shape)
print(x_train.shape)
plt.imshow(x_test[5].reshape(32,32))
plt.show()
plt.imshow(encode_model.predict(x_test)[5].reshape(16,16))
plt.show()
plt.imshow(model.predict(x_test)[5].reshape(32,32))
plt.show()
n_input = 256 
n_hidden_1 = 300
n_hidden_2 = 100
n_hidden_3 = 100
n_hidden_4 = 200
num_digits = 29
input_=Input(shape=(n_input,))
layer1=Dense(n_hidden_1,activation='relu')(input_)
layer2=Dense(n_hidden_2,activation='relu')(layer1)
layer3=Dense(n_hidden_3,activation='relu')(layer2)
layer4=Dense(n_hidden_4,activation='relu')(layer3)
Output=Dense(num_digits,activation='softmax')(layer4)
model_1=Model(input_,Output)
model_1.compile(optimizer='adam',loss='binary_crossentropy',metrics='accuracy')
model_1.summary()
history=model_1.fit(new_xtrain,y_train,batch_size=128,epochs=30)
y_pred=model_1.predict(new_xtest)
y_pred=np.argmax(y_pred,axis=1)
y_test=np.argmax(y_test,axis=1)
from sklearn.metrics import confusion_matrix,accuracy_score
print(accuracy_score(y_pred,y_test))