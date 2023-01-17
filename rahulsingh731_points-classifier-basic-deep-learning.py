import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
df_x=pd.read_csv('../input/classify-points-using-deep-learning-beginner/Logistic_X_Train.csv')

df_y=pd.read_csv('../input/classify-points-using-deep-learning-beginner/Logistic_Y_Train.csv')
df_x.head()
df_y.head()
plt.scatter(df_x.iloc[:,0],df_x.iloc[:,1],c=df_y.iloc[:,0])
from keras import models

from keras.layers import Dense
#Length of X and Y should be shape

X=np.array(df_x) #Check shape of x

Y=np.array(df_y) #check shape of y

print(np.shape(X),np.shape(Y)) #print shape of Y,X
model =models.Sequential()

model.add(Dense(20,activation='relu',input_shape=(1575,2)))

model.add(Dense(50,activation='tanh'))

model.add(Dense(100,activation='tanh'))

model.add(Dense(50,activation='tanh'))

model.add(Dense(20,activation='relu'))

model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
from sklearn.model_selection import train_test_split 
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42) #make a random split of 30% testing and 70% training
print(X_train.shape,Y_train.shape) #Check shape of train data 
hist= model.fit(X_train,Y_train,epochs=40,batch_size=32,validation_data=(X_test,Y_test))
output_train = model.predict(X_train)

output_train.shape
output_test = model.predict(X_test)

output_test.shape
plt.scatter(X_train[:,0],X_train[:,1],c=output_train) #Let;s Visualize our Output on train-data
plt.scatter(X_test[:,0],X_test[:,1],c=output_test) #let's visualise it on test=data