import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df.head()
# encoding Categorical data

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df.iloc[:,1] = le.fit_transform(df.iloc[:,1].values)
df.head()
df.info()
df = df.drop('Unnamed: 32',axis=1)
df.head()
df.describe()
sns.countplot(x='diagnosis',data=df)
df.corr()['diagnosis'][:-1].sort_values().plot(kind='bar')
plt.figure(figsize=(22,8))

sns.heatmap(df.corr(),annot=True, fmt = '.0%')
X = df.drop('diagnosis',axis=1).values

y = df['diagnosis'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
model = Sequential()

model.add(Dense(30,activation='relu'))

model.add(Dense(15,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam' )
model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test))
losses = pd.DataFrame(model.history.history)
losses
losses.plot()
model = Sequential()

model.add(Dense(30,activation='relu'))

model.add(Dense(15,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam' )
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)
model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test),

         callbacks=[early_stop])
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
model = Sequential()

model.add(Dense(30,activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(15,activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam' )
early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)
model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test),

         callbacks=[early_stop])
model_losses= pd.DataFrame(model.history.history)
model_losses.plot()
predictions= model.predict_classes(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))