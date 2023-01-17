import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

train.head()
test.head()
train.shape
test.shape
train_nonzero=train.astype(bool).sum(axis=0)
print(train_nonzero)
modified_train=train.loc[:,train_nonzero>500]
modified_train.shape
Y=modified_train.loc[:,'target']
X=modified_train
X=X.drop(columns=['ID','target'])
X_hat= test[test.columns.intersection(X.columns)]
X_hat.shape
X
X_hat
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
model=Sequential()
model.add(Dense(128,input_dim=330,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(1,activation='relu'))
model.summary()
model.compile(optimizer='rmsprop',loss='mse')
model.fit(X,Y,epochs=64,batch_size=32)
Y_hat=model.predict(X_hat)
test['target']=Y_hat
submission_file=test.loc[:,['ID','target']]
pd.set_option('display.float_format', lambda x: '%.8f' % x) 
submission_file
submission_file.to_csv('ArpitTandon_submission.csv',index=False)