import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



df=pd.read_csv('../input/diabetes.csv') #loading datasets

df.head() #loading first five data instances

df.groupby('Outcome').size()
df.hist(figsize=(10,8), bins =15)
X_train = df.iloc[:, 0:8] #selecting features columns

Y_train = df['Outcome'] # selecting output columns

X_train.head()

X_train.head()
y_train=df['Outcome'] #selecting output column
Y_train.head() #loading first 5 data instance
x_train.shape #checking the shape



from keras.models import Sequential #importing sequential model and dense layers

from keras.layers import Dense

model=Sequential() #instantiating the model
model.add(Dense(8, input_dim=8,activation='relu')) # 8 neurons for the first layers taking 8 input

model.add(Dense(4, activation='relu')) #4 neurons in the second layer

model.add(Dense(1,activation='sigmoid')) #1 neurons in the output layer

model.summary() #summary of the model

from keras.optimizers import SGD, RMSprop #importing optimizers

opt= SGD(lr=0.001) #lr=learning rate
opt= SGD(lr=0.001) #lr=learning rate
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['acc']) #compiling the model with metrics =accuracy
history=model.fit(X_train,Y_train,epochs=100,batch_size=8,validation_split=0.1) 

#saving the model and training on datasets with validation_set=10percent of trainning unit

                        
x_train.head()
test=[2,182,77,24,89,44.5,1.67,44]
test=np.expand_dims(test,axis=0) #fake data
test.shape
model.predict(test) #making prediction
import matplotlib.pyplot as plt

%matplotlib inline  

#for visualization
#visualization for training loss and validation loss
plt.plot(history.history['acc'], c='red')

plt.plot(history.history['val_acc'], c='green') 

plt.plot(history.history['loss'], color='red')

plt.plot(history.history["val_loss"], color='green')

#visualization for training loss and validation validation loss
model1=Sequential() #instantiating the model

model1.add(Dense(8, input_dim=8,activation='relu')) # 8 neurons for the first layers taking 8 input

model1.add(Dense(12, activation='relu'))#12eurons in the second layer

model1.add(Dense(6, activation = 'relu')) # 6 neurons in the third layer

model1.add(Dense(1, activation = 'sigmoid')) # 1 neuron in the final layer

opt = RMSprop(lr=0.001)

model1.compile(optimizer=opt, loss='binary_crossentropy',metrics = ['acc'])

history1=model1.fit(X_train,Y_train,epochs=150,batch_size=8,validation_split=0.1) 

plt.plot(history1.history['acc'], c='red')

plt.plot(history1.history['val_acc'], c='green') 

plt.plot(history1.history['loss'], color='red')

plt.plot(history1.history["val_loss"], color='green')

#visualization for training loss and validation validation loss