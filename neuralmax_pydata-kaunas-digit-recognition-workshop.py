import tensorflow as tf
tf.test.gpu_device_name()
import os
pth=os.path.join('..','input','train.csv')

import pandas as pd
traindf=pd.read_csv(pth)
traindf.head()
pth=os.path.join('..','input','test.csv')
testdf=pd.read_csv(pth)
testdf.head()
trainYl=traindf['label']
trainYl.head()
from keras.utils import to_categorical
trainY=to_categorical(trainYl,10)
trainY
traindf.head()
#trainX=traindf.drop(columns=['label']).as_matrix()
trainX=traindf.drop(['label'],axis=1).as_matrix()
trainX
from sklearn.model_selection import train_test_split
trainX,valX,trainY,valY=train_test_split(trainX,trainY,test_size=.1)
trainX.shape
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=784,activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(trainX,trainY,epochs=8,validation_data=(valX,valY))
test=testdf.as_matrix()
preds=model.predict(test)
print(preds.shape)
preds

import numpy as np
preds=np.argmax(preds,axis=1).tolist()
preds
pth=os.path.join('..','input','sample_submission.csv')
#sample=pd.read_csv(pth)
#sample.head()
idx=[i for i in range(1,len(preds)+1)]
predsdf=pd.DataFrame(data={'ImageId':idx,'Label':preds})
predsdf.head()
predsdf.to_csv('sub.csv',index=False)
model.summary()
model.__dict__
model.layers[1].__dict__
model.layers[1].output_shape
model.layers[3].output_shape
