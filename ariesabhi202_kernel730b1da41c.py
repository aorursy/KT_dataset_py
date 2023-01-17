import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data = pd.read_csv('../input/musk-dataset/musk_csv.csv')

data.head()
data.count()
data.describe()
data.nunique()
new_data=data.copy()
new_data.head()
new_data.drop(['molecule_name','ID','conformation_name'],axis=1,inplace=True)
new_data.head()
new_data.isnull().values.any()
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(new_data, new_data['class'], test_size = 0.20,random_state=120)
import keras

from keras.models import Model

from keras.layers import *
print(X_train.shape)

print(X_test.shape)
Inp=Input(shape=(167,))

x=Dense(500,activation='sigmoid',name='Hidden_layer1')(Inp)

x=Dense(300,activation='relu',name='Hidden_layer2')(x)

x=Dense(155,activation='sigmoid',name='Hidden_layer3')(x)

x=Dense(80,activation='relu',name='Hidden_layer4')(x)

output=Dense(1,activation='sigmoid',name='Output_layer')(x)

model=Model(Inp,output)

model.summary()
from keras import optimizers
l_rate=0.00001

training_epoch=50

batch_size=700

adma=optimizers.adam(lr=l_rate)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
op=model.fit(X_train,Y_train,batch_size=batch_size,epochs=training_epoch,verbose=2,validation_data=(X_test,Y_test))
print(op.history.keys())
plt.plot(op.history['loss'],label='train')

plt.xlabel('epochs')

plt.plot(op.history['val_loss'],label='test')

plt.ylabel('loss')

plt.legend()

plt.show()
plt.plot(op.history['accuracy'],label='train')

plt.xlabel('epochs')

plt.plot(op.history['val_accuracy'],label='test')

plt.ylabel('accuracy')

plt.legend()

plt.show()
Y_pred=model.predict(X_test).astype('int').flatten()

print(Y_pred)
from sklearn.metrics import classification_report

cls = classification_report(Y_test,Y_pred)

print(cls)