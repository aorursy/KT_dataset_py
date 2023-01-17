import numpy as np

import pandas as pd



from keras.layers import Dense, Dropout

from keras.models import Sequential



from sklearn.metrics import mean_absolute_error
data=pd.read_csv("~/Downloads/train.csv",encoding="utf-8",index_col=0)

data.head()
data.info()
data=data[(data.astype(str) != '?').all(axis=1)]
data.head()
df_onehot = data.copy()

df_onehot = pd.get_dummies(df_onehot, columns=['Size'], prefix = ['Size'])

df_onehot.head()
X = df_onehot.drop("Class", axis= 1)
y=df_onehot['Class']

from keras.utils import to_categorical

y= to_categorical(y)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(X,y)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
model = Sequential()

model.add(Dense(32,input_dim=13, activation='tanh'))

model.add(Dropout(rate=0.2))

model.add(Dense(16, activation='tanh'))

model.add(Dropout(rate=0.3))

model.add(Dense(8, activation='tanh'))

model.add(Dropout(rate=0.3))

model.add(Dense(6,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history=model.fit(x_train, y_train, validation_split=0.2, epochs=450,batch_size=75)
model.summary()

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
df_test=pd.read_csv("~/Downloads/test.csv", encoding = "utf-8", index_col=0)
df_test=pd.get_dummies(data=df_test,columns=['Size'])
pred=model.predict_classes(df_test)
df_test['Class']=pred
df=df_test['Class'].reset_index()
df.to_csv("~/Downloads/sub2.csv",index=False)