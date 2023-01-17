import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


from keras.layers import Flatten,Dense,Dropout
from keras.layers.convolutional import MaxPooling2D,Conv2D
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
epochs = 3
outputs = 10
columns = 28
rows = 28
data = pd.read_csv('../input/train.csv').values
X = data[:,1:]
Y = data[:,0]

X = X.reshape(X.shape[0],rows,columns,1)
X = X/255
Y = np_utils.to_categorical(Y)
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=.3)
import matplotlib.pyplot as plt
plt.imshow(Xtrain[8].reshape(28,28))
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),input_shape=(28,28,1),activation='relu'))
model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.2))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(.25))
model.add(Dense(outputs,activation='softmax'))

model.compile(optimizer= 'adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(Xtrain,Ytrain,epochs = epochs)
score = model.evaluate(Xtest,Ytest)
print('accuracy : ',score[1])
result = model.predict_classes(Xtest)
df_res = pd.DataFrame()
df_res['id'] = [i for i in range(1,Xtest.shape[0]+1)]
df_res['result'] = list(result)
df_res.to_csv('result.csv',index='False')