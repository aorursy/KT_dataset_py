import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/employee-dataset/Dataset/Train.csv')
id_train = np.array(df['Employee_ID'])
y_train = np.array(df['Attrition_rate'])
x_train = np.array(df.drop(['Employee_ID','Attrition_rate'], axis=1))
df2 = pd.read_csv('../input/employee-dataset/Dataset/Test.csv')
id_test = np.array(df2['Employee_ID'])
x_test = np.array(df2.drop('Employee_ID', axis=1))
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(np.unique(x_train[:,0]))
x_train[:,0]=le.transform(x_train[:,0])
x_test[:,0]=le.transform(x_test[:,0])
for i in range(2,7):
    le.fit(np.unique(x_train[:,i]))
    x_train[:,i]=le.transform(x_train[:,i])
    x_test[:,i]=le.transform(x_test[:,i])
le.fit(np.unique(x_train[:,13]))
x_train[:,13]=le.transform(x_train[:,13])
x_test[:,13]=le.transform(x_test[:,13])
for i in range(15,22):
    x_test[:,i]=np.nan_to_num(x_test[:,i])
    x_train[:,i]=np.nan_to_num(x_train[:,i])
x_train = x_train[:,[2,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
x_test = x_test[:,[2,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
x_train=x_train.astype(float)
x_test=x_test.astype(float)
for i in range(0,17):
    x_test[:,i]=np.nan_to_num(x_test[:,i])
    x_train[:,i]=np.nan_to_num(x_train[:,i])
x_train[0:10,:]
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
ann = Sequential()

ann.add(Dense(17, activation = 'tanh', input_dim = 17))
ann.add(Dense(units = 256, activation = 'relu'))
ann.add(Dense(units = 256, activation = 'relu'))
ann.add(Dense(units = 256, activation = 'relu'))
ann.add(Dense(units = 256, activation = 'relu'))
ann.add(Dense(units = 256, activation = 'relu'))
ann.add(Dense(units = 1))
ann.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mean_squared_error'])
history = ann.fit(x_train, y_train, validation_split=0.33, batch_size = 100, epochs = 500, verbose=0)
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
y_test = ann.predict(x_test)
for i in range(0,3000):
    if y_test[i]<0:
        y_test[i]=0
    elif y_test[i]>1:
        y_test[i]=1
y_test
df3 = pd.DataFrame()
df3['Employee_ID'] = id_test.reshape(3000).tolist()
df3['Attrition_rate'] = y_test.reshape(3000).tolist()
df3.to_csv("./file.csv", sep=',',index=True)