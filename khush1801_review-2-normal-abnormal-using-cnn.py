import numpy as np 
import pandas as pd 
import keras
from keras.models import Sequential
from keras.layers import Reshape
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout,MaxPooling1D,GlobalAveragePooling1D

import os
print(os.listdir("../input"))
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


MI = pd.read_csv("../input/avanidataset/ptbdb_abnormal.csv") 
HC = pd.read_csv("../input/avanidataset/ptbdb_normal.csv") 

new_column_name = ['label']
for num in range(MI.shape[1]-1):
    tem = 'dim' + str(num)
    new_column_name.append(tem)
MI.columns = new_column_name    




column_name = ['label']
for num in range(HC.shape[1]-1):
    tem = 'dim' + str(num)
    column_name.append(tem)
HC.columns = column_name

train_MI=MI.iloc[0:7000]
test_MI=MI.iloc[7000:9000]
train_HC=HC.iloc[0:2500]
test_HC=HC.iloc[2500:3500]
train=[train_MI,train_HC]
train=pd.concat(train,sort=True)
test=[test_MI,test_HC]
test=pd.concat(test,sort=True)

ytrain=list(range(9500))
ytest=list(range(3000))

for i in range(9500):
    if i<=7000:
        ytrain[i]=1
    else:
        ytrain[i]=0

    
for j in range(3000):
    if j<=2000:
        ytest[j]=1
    else:
        ytest[j]=0     
        
        
        
ytrain=keras.utils.np_utils.to_categorical(ytrain)
ytest=keras.utils.np_utils.to_categorical(ytest)

ytrain
ytest



train=np.asarray(train)
train=train.reshape(9500, 188, 1)

test=np.asarray(test)
test=test.reshape(3000, 188, 1)
# For conv1d statement: 
#input_shape = (ncols, 1) 

model = Sequential()
model.add(Convolution1D(100, 5, activation='relu', input_shape=(188,1)))
model.add(Convolution1D(100, 10, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Convolution1D(100, 10, activation='relu'))
model.add(Convolution1D(160, 10, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
#model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
#print(model_m.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train, ytrain, validation_data=(test, ytest), epochs=5)
import seaborn as sns

Y_pred = model.predict(test)
print(Y_pred)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
print(Y_pred_classes)
print("Result from real time testing data:") 
print(Y_pred_classes[0])


_,accuracy = model.evaluate(test, ytest, verbose=0)
accuracy


























