# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

data=pd.read_csv('../input/train.csv')

X=data.values[:,3:-1]
Y=data.values[:,-1]

X.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,Y)
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import BatchNormalization,Dropout
model=Sequential()
model.add(Dense(128,input_dim=22,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(16,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(4,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1,activation='linear'))
from keras import optimizers

adam=optimizers.adam(lr=0.01)
model.compile(optimizer=adam,loss='mean_squared_error',metrics=['mae'])
saved_model=model.fit(X_train,y_train,batch_size=100000,epochs=300,validation_split=0.1)
plt.plot(saved_model.history['loss'])
plt.plot(saved_model.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation mae values
plt.plot(saved_model.history['mean_absolute_error'])
plt.plot(saved_model.history['val_mean_absolute_error'])
plt.title('Mean Abosulte Error')
plt.ylabel('Mean absolute error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
model.evaluate(X_test,y_test)
test_data_org=pd.read_csv('../input/test.csv')
test_data_org
test_data=test_data_org.values[:,3:]
y_pred=model.predict(test_data)
submission=pd.DataFrame()
submission['Id']=test_data_org['Id']
submission['winPlacePerc']=y_pred
submission.to_csv('submission.csv',index=False)