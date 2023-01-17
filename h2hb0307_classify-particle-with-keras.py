import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import keras
data = pd.read_csv('../input/pid-5M.csv')
data.head()
train = data.iloc[0:int(len(data)*0.7),:]
train.tail()
train.head()
test = data.iloc[int(len(data)*0.7):,:]
test.head()
print(train.isnull().sum())
print('-'*40)
print(test.isnull().sum())
train.info()
print('-'*40)
test.info()
print(train["id"].unique())
print('-'*40)
print(test["id"].unique())
train['id'] = train['id'].map( {-11: 'positron', 211: 'pion',321:'kaon',2212:'proton'} ).astype(str)
print('-'*40)
test['id']=test['id'].map( {-11: 'positron', 211: 'pion',321:'kaon',2212:'proton'} ).astype(str)
train.head()
train['id'] =train['id'].map( { 'positron':0, 'pion':1,'kaon':2,'proton':3} ).astype(int)
test['id'] =test['id'].map( { 'positron':0, 'pion':1,'kaon':2,'proton':3} ).astype(int)
train.head()
positron = train[train['id']==0]['id'].value_counts()
pion = train[train['id']==1]['id'].value_counts()
kaon = train[train['id']==2]['id'].value_counts()
proton = train[train['id']==3]['id'].value_counts()

df = pd.DataFrame([positron, pion, kaon, proton])
df.index = ['positron','pion','kaon','proton']
df.plot(kind='bar',stacked=True, figsize=(10,5))
train.describe()
Y_train = train['id']
X_train = train.drop('id',axis=1)

Y_test = test['id']
X_test = test.drop('id',axis=1)

#one-hot Encoding
y_train = keras.utils.to_categorical(Y_train,4)
y_test = keras.utils.to_categorical(Y_test,4)
Nin = 6
Nh = 100
number_of_class = 4
Nout = number_of_class
model = keras.models.Sequential()
model.add(keras.layers.Dense(Nh, activation='relu', input_shape=(Nin,)))
model.add(keras.layers.Dense(4, activation='softmax'))
model.compile(optimizer='adam',
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train,batch_size=500, epochs=5)
score =model.evaluate(X_test, y_test)
print("Loss:",score[0])
print("Accuracy:",score[1])
model = keras.models.Sequential()
model.add(keras.layers.Dense(Nh, activation='relu', input_shape=(Nin,)))
model.add(keras.layers.Dense(Nh, activation='relu'))
model.add(keras.layers.Dense(Nh, activation='relu'))
model.add(keras.layers.Dense(4, activation='softmax'))

model.compile(optimizer='adam',
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train,batch_size=500, epochs=5)
score =model.evaluate(X_test, y_test)
print("Loss:",score[0])
print("Accuracy:",score[1])
