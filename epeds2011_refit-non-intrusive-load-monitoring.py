import numpy as np 

import pandas as pd 

import os

import sys



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error



from keras.models import Sequential,Model

from keras.layers import LSTM,Input, Dense, LSTM, MaxPooling1D, Conv1D, Conv2D ,Flatten ,Bidirectional

from keras import metrics

from keras.metrics import mean_absolute_percentage_error,mean_absolute_error,mean_squared_error



import matplotlib.pyplot as plt 
Windos_Size = 300

total_sample = 15000

df = pd.read_csv("../input/refit02/REFIT_House2.csv")



#Generate input and target arrays

X = np.array(df.iloc[:,1]).reshape(-1,1)

FG = np.array(df.iloc[:,2]).reshape(-1,1)



# calculate standard deviation to scale the data

X_std = np.std(X)

FG_std = np.std(FG)



#Generate required length of data

X=X[0:total_sample] #main power

FG=FG[0:total_sample] #fridg power



plt.figure(0)

FG_plot_1 = plt.plot(FG)

plt.show()



#Generate sliding window for data preprocessing

L = (len(X))

A= np.zeros((L-Windos_Size+1,Windos_Size))

B= np.zeros((L-Windos_Size+1,Windos_Size))



for i in range(L-Windos_Size+1):

    A[i] = np.array(X[i:i+Windos_Size]).ravel()     

    B[i] = np.array(FG[i:i+Windos_Size]).ravel()     

X = A

FG = B



#Subtract the mean of each row and divide it by standard deviation

X = (X - X.mean(axis=1).reshape(-1, 1))/X_std

FG = (FG - FG.mean(axis=1).reshape(-1, 1))/FG_std





# Split data

X_train, X_test, fg_train, fg_test = train_test_split(X,FG,test_size=0.25, random_state = 0, shuffle = False)



# reshape (N 10) to (N 10 1) for Conv1D don't make error in LSTMs

X_train = np.expand_dims(X_train, axis=2)

X_test = np.expand_dims(X_test, axis=2)



#Model definition

model = Sequential()

model.add(Conv1D(4, kernel_size=4, strides = 1, activation='linear', input_shape = (Windos_Size ,1)))

model.add(MaxPooling1D(pool_size=3))

model.add(Bidirectional(LSTM(128,activation = 'tanh', return_sequences=True)))

model.add(Bidirectional(LSTM(256,activation = 'tanh')))

model.add(Dense(128, activation='tanh'))

model.add(Dense(Windos_Size))

model.summary()

model.compile(loss= 'mae',optimizer = 'adam', metrics=[ 'mae' ])



model.fit(X_train,fg_train,epochs=10,verbose=2,batch_size=128)



#Predict the output of model

Predict_FG = model.predict(X_test)

#convert sequence to sequence back to single vector



LL = len(fg_test)+ Windos_Size

C = np.zeros((len(fg_test),1))

D = np.zeros((len(fg_test),1))



C[0:Windos_Size,0] = fg_test[0,:] 

D[0:Windos_Size,0] = Predict_FG[0,:] 



for i in range(len(fg_test)):

    C[i]= fg_test[i,-1]

    D[i]= Predict_FG[i,-1]



fg_test_1 = C

Predict_FG_1 = D



#convert them to original data

X = np.array(df.iloc[:,1]).reshape(-1,1)

FG = np.array(df.iloc[:,2]).reshape(-1,1)



X=X[0:total_sample] #main power

FG=FG[0:total_sample] #fridg power

L = (len(X))





A= np.zeros((L-Windos_Size+1,Windos_Size))

B= np.zeros((L-Windos_Size+1,Windos_Size))



for i in range(L-Windos_Size+1):

    A[i] = np.array(X[i:i+Windos_Size]).ravel()     

    B[i] = np.array(FG[i:i+Windos_Size]).ravel()     

X = A

FG = B



# split real data to train and test to find real fg_test

X_train_real, X_test_real, fg_train_real, fg_test_real = train_test_split(X,FG,test_size=0.25, random_state = 0, shuffle = False)



# multiply it by FG_std and add the mean for each row to generate the real output

fg_test_2 = (fg_test_1*FG_std) + fg_test_real.mean(axis=1).reshape(-1,1)

Predict_FG_2 = (Predict_FG_1*FG_std) + fg_test_real.mean(axis=1).reshape(-1,1)





plt.figure(1)

Test = plt.plot(fg_test_2)

Predict= plt.plot(Predict_FG_2)

plt.show()



#save the model

model.save('model_ws299.h5')

fg_df = pd.DataFrame(data=fg_test)

fg_df.to_csv("fg_test.csv",index=False)



np.set_printoptions(threshold=sys.maxsize)

print(fg_test_2)

print(Predict_FG_2)
