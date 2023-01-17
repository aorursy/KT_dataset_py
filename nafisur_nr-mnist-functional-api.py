import keras
from keras.layers import Input,Dense,Dropout
from keras.models import Model
from keras.utils import to_categorical

import numpy as np
import pandas as pd
df_train=pd.read_csv('../input/train.csv')
df_test=pd.read_csv('../input/test.csv')
print(df_train.shape)
print(df_test.shape)
X_train=df_train.iloc[:,1:].values
y_train=df_train.iloc[:,0].values
X_test=df_test.values
X_train=X_train.astype(float)
X_test=X_test.astype(float)
X_train /=255
X_test /=255
y_train=to_categorical(y_train,10)
input1=Input(shape=(784,),name='input_layers')
d1=Dense(512,activation='tanh',name='hidden_dense_layer1')(input1)
d2=Dropout(0.3)(d1)
d3=Dense(512,activation='tanh',name='hidden_dense_layer2')(d2)
d4=Dropout(0.3)(d3)
output=Dense(10,activation='softmax',name='output_layer')(d4)
model=Model(inputs=input1,outputs=output)
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=0.05,momentum=0.9,decay=1e-6),metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=64,epochs=20)
