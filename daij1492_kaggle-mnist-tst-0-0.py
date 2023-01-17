import os

os.listdir('../input') # ['test.csv', 'train.csv']
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
trainDF = pd.read_csv('../input/train.csv')
trainDF.columns
Y,X=trainDF.iloc[:,0].values,trainDF.iloc[:,range(1,785)].values
Y.shape,X.shape
from keras.models import Sequential

from keras.layers import Dense

model = Sequential()

model.add(Dense(30,input_dim=784,activation='relu'))

model.add(Dense(10,activation='softmax'))

model.summary()
30*785+31*10
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
from keras.utils import to_categorical

Y_ = to_categorical(Y)

history = model.fit(X,Y_,epochs=2,batch_size=128,verbose=1)
history.history
history = model.fit(X,Y_,epochs=3,batch_size=128,verbose=1)
history.history
history = model.fit(X,Y_,epochs=3,batch_size=128,verbose=1,validation_split=.3)
history.history