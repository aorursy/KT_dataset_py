import pandas as pd

import numpy as np

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten

from keras.losses import categorical_crossentropy

from keras.optimizers import RMSprop

from keras.metrics import categorical_accuracy

from keras.utils import to_categorical
df_train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

df_test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
df_train.head()
X=df_train.drop('label',axis=1)/255

Y=to_categorical(df_train['label'])
X=np.array(X).reshape(42000,28,28,1)

X_predict=np.array(df_test/255).reshape(28000,28,28,1)
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=101)
network=Sequential()

network.add(Conv2D(128,(3,3),activation='relu',input_shape=(28,28,1)))

network.add(MaxPool2D(2,2))

network.add(Conv2D(128,(3,3),activation='relu'))

network.add(MaxPool2D(2,2))

network.add(Conv2D(128,(3,3),activation='relu'))

#network.add(MaxPool2D(2,2))

network.add(Flatten())

network.add(Dropout(0.2))

network.add(Dense(256,activation='relu'))

network.add(Dense(512,activation='relu'))

network.add(Dense(256,activation='relu'))

network.add(Dense(10,activation='softmax'))
network.summary()
network.compile(loss='categorical_crossentropy',optimizer=RMSprop(lr=0.0001),metrics=['acc'])
network.fit(X_train,Y_train,epochs=30,batch_size=256)
network.evaluate(X_test,Y_test,batch_size=128)