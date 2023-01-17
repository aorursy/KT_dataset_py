import pandas as pd                   

from sklearn.neural_network import MLPRegressor   

from sklearn.neighbors import KNeighborsRegressor

import matplotlib.pyplot as plt

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.multioutput import MultiOutputRegressor

from sklearn.linear_model import Ridge

%pylab inline

pylab.rcParams['figure.figsize'] = (15, 6)

df = pd.read_csv("/kaggle/input/data2.csv")

df=df.drop("Unnamed: 0",1)

df.shape

X=df.drop(["R1","R2"],1)

y=df[["R1","R2"]]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

y=scaler.fit_transform(y)





#X = df.iloc[:,0:20]  #independent columns

#y = df.iloc[:,-2]



#from sklearn.preprocessing import MinMaxScaler



#X = MinMaxScaler().fit_transform(X)





from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

df.head()
df.isnull().values.any()
from keras.models import Sequential

from keras.layers import LSTM, Dense,Dropout

from keras.datasets import mnist

from keras.utils import np_utils
X_train = X_train.values.reshape(-1, 1, 20)

X_test  = X_test.values.reshape(-1, 1, 20)

#y_train = y_train.reshape(-1, 2)

#y_test = y_test.reshape(-1, 2)
import keras

model = Sequential()



model.add(LSTM(

         input_shape=(60,20),

         units=100,

         return_sequences=True))





model.add(LSTM(

          units=75,

          return_sequences=True))



model.add(LSTM(units=75))



model.add(Dense(units=2, activation='linear'))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

from keras.callbacks import EarlyStopping

history=model.fit(X_train, y_train, epochs=50, batch_size=1024, validation_data=(X_test,y_test), verbose=1,

          callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')])
from matplotlib import pyplot



pyplot.plot(history.history['loss'])

pyplot.plot(history.history['val_loss'])

pyplot.title('model train vs validation loss')

pyplot.ylabel('loss')

pyplot.xlabel('epoch')

pyplot.legend(['train', 'validation'], loc='upper right')

pyplot.show()





pyplot.plot(history.history['mean_absolute_error'])

pyplot.plot(history.history['val_mean_absolute_error'])

pyplot.title('model train vs validation mean_absolute_error')

pyplot.ylabel('mean_absolute_error')

pyplot.xlabel('epoch')

pyplot.legend(['train', 'validation'], loc='upper right')

pyplot.show()
ypred=model.predict(X_train[:100])
arr1=[]

for x in ypred:

    arr1.append(x[0])

    

arr2=[]

for y in y_train:

    arr2.append(y[0])


plt.plot(arr1[50:100],'b')



plt.plot(arr2[50:100],'r')

plt.show()
# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional

from keras.optimizers import SGD

import math

from sklearn.metrics import mean_squared_error
import keras

print('Build model...')

from keras.optimizers import Adam #maybe put this at the top of your file

opt = Adam(lr=0.00001)

regressorGRU = Sequential()



regressorGRU.add(GRU(100, input_shape=(30,20), return_sequences=True,activation="relu"))

regressorGRU.add(GRU(75,return_sequences=True))

regressorGRU.add(GRU(50))

regressorGRU.add(Dense(units=2,activation="linear"))

# Compiling the RNN

regressorGRU.compile(optimizer=opt,loss='mean_absolute_error',metrics=['mean_absolute_error'])

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history= regressorGRU.fit(X_train,y_train,epochs=75,batch_size=2048,validation_data=(X_test,y_test),callbacks=[early_stop])



from matplotlib import pyplot

pyplot.plot(history.history['mean_absolute_error']) 

pyplot.plot(history.history['val_mean_absolute_error']) 

pyplot.title('model train vs validation mean_absolute_error') 

pyplot.ylabel('mean_absolute_error') 

pyplot.xlabel('epoch') 

pyplot.legend(['train', 'validation'], loc='upper right') 

pyplot.show()

pyplot.plot(history.history['loss']) 

pyplot.plot(history.history['val_loss']) 

pyplot.title('model train vs validation loss') 

pyplot.ylabel('loss') 

pyplot.xlabel('epoch') 

pyplot.legend(['train', 'validation'], loc='upper right') 

pyplot.show()
ypred=regressorGRU.predict(X_test[:100])
arr1=[]

for x in ypred:

    arr1.append(x[0])

    

arr2=[]

for y in y_test:

    arr2.append(y[0])

import matplotlib.pyplot as plt

%pylab inline

pylab.rcParams['figure.figsize'] = (15, 6)

plt.plot(arr1[0:50],'b')#predicted



plt.plot(arr2[0:50],'r')#true

plt.show()




from keras.models import Sequential 





from keras.layers import Dense, Activation, Flatten, Dropout 

from keras import regularizers

from keras.optimizers import SGD, Adam

#output_dim = nb_classes = 15 

#input_dim = seq_length

model = Sequential() 

model.add(Flatten())

model.add(Dense(100, activation='relu',input_shape=(60,20)))

model.add(Dense(75))

model.add(Dense(50))



model.add(Dense(2, activation='linear'))

model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.0001), metrics=['mean_absolute_error'])

history = model.fit(X_train, y_train,batch_size=1024,validation_data=(X_test,y_test), epochs=50)
from matplotlib import pyplot

plt.figure()

plt.plot(history.history['mean_absolute_error'],color='blue')

plt.plot(history.history['val_mean_absolute_error'],color='orange')

#plt.title('Model loss',fontsize=12)

#plt.ylabel('loss',fontsize=12)

plt.xlabel('epoch',fontsize=12)

plt.legend(['train', 'validation'])

#plt.savefig('Write_up/model_loss.png')

plt.show()
ypred=model.predict(X_test)
arr1=[]

for x in ypred:

    arr1.append(x[0])

    

arr2=[]

for y in y_test:

    arr2.append(y[0])



plt.plot(arr1[:50],'b')



plt.plot(arr2[:50],'r')

plt.show()