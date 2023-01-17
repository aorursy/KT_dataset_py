import pandas as pd

from sklearn.preprocessing import LabelEncoder

import numpy as np

import matplotlib.pyplot as plt



# read Kaggle datasets

X_train = pd.read_csv('/kaggle/input/career-con-2019/X_train.csv')

y_train = pd.read_csv('/kaggle/input/career-con-2019/y_train.csv')

# split X_train

samples = 20

time_series = 128

start_x = X_train.shape[0] - samples*time_series

X_train_new, X_test_new = X_train.iloc[:start_x], X_train.iloc[start_x:]

# split y_train

start_y = y_train.shape[0] - samples

y_train_new, y_test_new = y_train.iloc[:start_y], y_train.iloc[start_y:]

X_train_new.head(5)
X_train_new=X_train_new.drop(['row_id', 'series_id','measurement_number'], axis=1)

X_test_new=X_test_new.drop(['row_id', 'series_id','measurement_number'], axis=1)

y_train_new.head(5)

y_train_new=y_train_new.drop(['series_id', 'group_id'], axis=1)

y_test_new=y_test_new.drop(['series_id', 'group_id'], axis=1)
X_train_new=X_train_new.values

X_test_new=X_test_new.values

y_train_new=y_train_new.values

y_test_new=y_test_new.values

print('The size of X_train_new:', X_train_new.shape )

print('The size of X_test_new:', X_test_new.shape )



print('The size of y_train_new:', y_train_new.shape )

print('The size of y_test_new:', y_test_new.shape )
from keras.utils import to_categorical



labelencoder_y = LabelEncoder()



y=np.concatenate([y_train_new,y_test_new])

y=labelencoder_y.fit_transform(y)

y=to_categorical(y)





y_train_new = y[:-20]

y_test_new = y[-20:]

print('The shape of y_train_new:', y_train_new.shape)

print('The shape of y_test_new:', y_test_new.shape)

#FOR TRAIN DATASET

roll=np.zeros([X_train_new.shape[0],1])

pitch=np.zeros([X_train_new.shape[0],1])

yaw=np.zeros([X_train_new.shape[0],1])





for i in range(X_train_new.shape[0]):

  roll[i] = np.arctan2(2*(X_train_new[i,1]*X_train_new[i,2] + X_train_new[i,3]*X_train_new[i,0]),1 - 2*(X_train_new[i,2]*X_train_new[i,2] + X_train_new[i,3]*X_train_new[i,3]))

  pitch[i] = np.arcsin(2*(X_train_new[i,1]*X_train_new[i,3] - X_train_new[i,0]*X_train_new[i,2]))

  yaw[i] = np.arctan2(2*(X_train_new[i,1]*X_train_new[i,0] + X_train_new[i,2]*X_train_new[i,3]),1 - 2*(X_train_new[i,3]*X_train_new[i,3] + X_train_new[i,0]*X_train_new[i,0]))



X_train_new=np.delete(X_train_new,[0,1,2,3], 1)

X_train_new=np.concatenate((roll,pitch,yaw,X_train_new),axis=1)



#FOR TEST DATA SET

roll=np.zeros([X_test_new.shape[0],1])

pitch=np.zeros([X_test_new.shape[0],1])

yaw=np.zeros([X_test_new.shape[0],1])





for i in range(X_test_new.shape[0]):

  roll[i] = np.arctan2(2*(X_test_new[i,1]*X_test_new[i,2] + X_test_new[i,3]*X_test_new[i,0]),1 - 2*(X_test_new[i,2]*X_test_new[i,2] + X_test_new[i,3]*X_test_new[i,3]))

  pitch[i] = np.arcsin(2*(X_test_new[i,1]*X_test_new[i,3] - X_test_new[i,0]*X_test_new[i,2]))

  yaw[i] = np.arctan2(2*(X_test_new[i,1]*X_test_new[i,0] + X_test_new[i,2]*X_test_new[i,3]),1 - 2*(X_test_new[i,3]*X_test_new[i,3] + X_test_new[i,0]*X_test_new[i,0]))



X_test_new=np.delete(X_test_new,[0,1,2,3], 1)

X_test_new=np.concatenate((roll,pitch,yaw,X_test_new),axis=1)

nfeatures=X_train_new.shape[1]

ntimestamp=128

nsamples=3790

X_3D_train=X_train_new[:,0].reshape(nsamples,ntimestamp)

for i in range(nfeatures-1):

  i=i+1

  r=X_train_new[:,i].reshape(nsamples,ntimestamp)

  X_3D_train=np.dstack((X_3D_train,r))

print('The shape of X_train: ', X_3D_train.shape)



nfeatures=X_test_new.shape[1]

ntimestamp=128

nsamples=20

X_3D_test=X_test_new[:,0].reshape(nsamples,ntimestamp)

for i in range(nfeatures-1):

  i=i+1

  r=X_test_new[:,i].reshape(nsamples,ntimestamp)

  X_3D_test=np.dstack((X_3D_test,r))

print('The shape of X_test: ', X_3D_test.shape)

for i in range(2):

    rate = X_3D_train[:,:,i]

    rate_c = np.copy(rate)

    rate_c[:,1:] = rate_c[:,:-1]

    rate = rate - rate_c

    X_3D_train[:,:,i] = rate

    

for i in range(2):

    rate = X_3D_test[:,:,i]

    rate_c = np.copy(rate)

    rate_c[:,1:] = rate_c[:,:-1]

    rate = rate - rate_c

    X_3D_test[:,:,i] = rate
from scipy import fftpack

X_3Dfft = np.abs(np.fft.fft(X_3D_train,axis=1))

#freqs = fftpack.fftfreq(len(x)) * f_s

X_3D_train=np.dstack((X_3D_train,X_3Dfft))

print('The size of X_3D: ', X_3D_train.shape)



from scipy import fftpack

X_3Dfft = np.abs(np.fft.fft(X_3D_test,axis=1))

#freqs = fftpack.fftfreq(len(x)) * f_s

X_3D_test=np.dstack((X_3D_test,X_3Dfft))

print('The size of X_3D: ', X_3D_test.shape)
nfeatures=X_3D_train.shape[2]

ntimestamp=X_3D_train.shape[1]

nsamples=X_3D_train.shape[0]



print('Number of features in Xtrain: ', nfeatures)

print('Number of timestamps in Xtrain: ', ntimestamp)

print('Number of samples in Xtrain: ', nsamples)

for k in range(nfeatures):

    X_train_m = np.mean(X_3D_train[:,:,k])

    X_train_sd = np.std(X_3D_train[:,:,k])

    X_3D_train[:,:,k] = (X_3D_train[:,:,k]-X_train_m)/X_train_sd



for k in range(nfeatures):

    X_test_m = np.mean(X_3D_test[:,:,k])

    X_test_sd = np.std(X_3D_test[:,:,k])

    X_3D_test[:,:,k] = (X_3D_test[:,:,k]-X_test_m)/X_test_sd
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Dropout

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D

predict=np.zeros([y_test_new.shape[0],y_test_new.shape[1],3])

for k in range(1):

  verbose, epochs, batch_size = 1, 600, 512

  model = Sequential()

  model.add(Conv1D(filters=64, kernel_size=8, activation='relu', input_shape=(ntimestamp,nfeatures)))

  model.add(Dropout(0.5))

  model.add(Conv1D(filters=128, kernel_size=8, activation='relu'))

  model.add(Dropout(0.5))

  model.add(Conv1D(filters=256, kernel_size=8, activation='relu'))

  model.add(Dropout(0.5))

  model.add(MaxPooling1D(pool_size=1))

  model.add(Flatten())

  model.add(Dense(128, activation='relu')) 

  model.add(Dropout(0.5))

  model.add(Dense(y_train_new.shape[1], activation='softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit network

  model.fit(X_3D_train, y_train_new, epochs=epochs, batch_size=batch_size, verbose=verbose)

# evaluate model

predict=np.zeros([y_test_new.shape[0],y_test_new.shape[1],3])



yhat=model.predict(X_3D_test)

predict[:,:,k]=yhat



#_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)



yhat=np.mean(predict,axis=2)

yhat_final = np.array(list(np.argmax(yhat,axis=1)))

y_testt=np.argmax(y_test_new,axis=1)



accuracy=np.mean(yhat_final==y_testt)

print('The accuracy is:', accuracy)