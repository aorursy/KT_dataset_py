import pandas as pd

import numpy as np

import seaborn as sea

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense,Dropout

from keras.losses import categorical_crossentropy

from keras.optimizers import RMSprop,SGD

from keras.metrics import categorical_accuracy

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
import os

os.listdir('../input')
train=pd.read_csv('../input/digit-recognizer/train.csv')

target_train=train['label']

data_train=train.drop('label',axis=1)/255
test=pd.read_csv('../input/digit-recognizer/test.csv')

data_test=test/255
X_train,X_test,Y_train,Y_test=train_test_split(data_train,target_train,test_size=0.1,random_state=47)
sea.countplot(Y_train)

plt.show()
X_train_processed=np.array(X_train).reshape(-1,28*28)

X_test_processed=np.array(X_test).reshape(-1,28*28)
Y_train_pro=to_categorical(Y_train)

Y_test_pro=to_categorical(Y_test)
network=Sequential()

network.add(Dense(512,activation='relu',input_shape=(28*28,)))

network.add(Dropout(0.005))

network.add(Dense(256,activation='relu'))

network.add(Dense(10,activation='softmax'))
network.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.09,momentum=0.9),metrics=['categorical_accuracy'])

network.fit(X_train_processed,Y_train_pro,epochs=20,batch_size=256)
network.evaluate(X_test_processed,Y_test_pro)