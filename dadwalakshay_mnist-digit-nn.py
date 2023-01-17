import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras import layers
import pandas as pd
import numpy as np
test_data = pd.read_csv('../input/test.csv',delimiter=',')
train_data = pd.read_csv('../input/train.csv',delimiter=',')
print('Shape of Test Dataset is {} and Train Dataset is{}.'.format(test_data.shape,train_data.shape))
train_data.head() #First columns label is the true output of image.
test_data.head()
xtrain = (train_data.iloc[:,1:].values).astype('float16')
ytrain = (train_data.iloc[:,0].values).astype('int32')
xtest = (test_data.values).astype('float16')
classify = 10
def one_hot(ip,classes): #convert label of train data into one-hot encoding
    one_hot = []
    for i in range(len(ip)):
        j = (np.eye(N=ip.shape[0],M=classes,k=ip[i]))[0]
        one_hot.append(j)
    return np.array(one_hot)
ytrain = one_hot(ytrain,classify)
xtrain = xtrain.reshape(xtrain.shape[0],28,28,1)
xtest = xtest.reshape(xtest.shape[0],28,28,1)
xtrain = xtrain / 255.0
xtest = xtest / 255.0
model = Sequential()
model.add(layers.InputLayer(input_shape=(28,28,1)))
model.add(Conv2D(16,5,padding='same',activation='relu'))
model.add(layers.MaxPooling2D(padding='same'))
model.add(layers.Flatten())
model.add(layers.Dense(416,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
from keras.optimizers import Adam
optimizer = Adam(lr = 0.01)
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
#model.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['accuracy'])
model.fit(x=xtrain,y=ytrain,batch_size=64,epochs=1)
predctions = model.predict(x=xtest)
pred_cls = np.argmax(predctions,axis=1)
submit = pd.DataFrame({'ImageID':list(range(1,xtest.shape[0]+1)),'Label':pred_cls})
submit.to_csv('kaggel_digit.csv',index=False,header=True)
pd.read_csv('kaggel_digit.csv').head()