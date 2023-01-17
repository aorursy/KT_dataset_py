# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

pd.set_option('max_columns', None)

import datetime

import math, time



from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Activation, LSTM, GRU

from tensorflow.keras.optimizers import Adam,RMSprop,SGD
stocks_df = pd.read_csv("/kaggle/input/nyse/prices-split-adjusted.csv",index_col=0)

firm_df = pd.read_csv("/kaggle/input/nyse/securities.csv",header=0)
stocks_df.head(10)
firm_df.head(10)
stocks_df.isnull().sum()
firm_df.isnull().sum()
symbols = list(set(stocks_df.symbol))

len(symbols)
symbols[:10]
firm_df.loc[firm_df.Security.str.startswith('Micro'),:]
stocks_msft = stocks_df[stocks_df.symbol == 'MSFT']
stocks_msft.head(10)
stocks_msft.shape
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

plt.plot(stocks_msft.open.values.astype('float32'),color='red',label=open)

plt.title('Opening Stock Prices for Microsoft')

plt.grid(True)

plt.xlabel('time(days)')

plt.ylabel('prices')



plt.subplot(1,2,2)

plt.plot(stocks_msft.close.values.astype('float32'),color='green',label=open)

plt.title('Closing Stock Prices for Microsoft')

plt.grid(True)

plt.xlabel('time(days)')

plt.ylabel('prices')

plt.figure(figsize=(25,10))

plt.plot(stocks_msft.volume.values,color='black')

plt.xlabel('time(days)')

plt.ylabel('Volume')
# fix random seed for reproducibility

np.random.seed(7)
# convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=1):

    dataX, dataY = [], []

    for i in range(len(dataset)-look_back-1):

        a = dataset[i:(i+look_back), 0]

        dataX.append(a)

        dataY.append(dataset[i + look_back, 0])

    return np.array(dataX), np.array(dataY)
msft = stocks_msft.copy()

#msft = msft.drop('symbol',axis=1)

closing = msft.close.values

closing = closing.reshape(len(closing),1)



#normalize the dataset

scaler = MinMaxScaler(feature_range=(0, 1))

closing = scaler.fit_transform(closing)
# split into train and test sets

train_size = int(len(closing) * 0.75)

test_size = len(closing) - train_size
print(train_size , test_size)
train, test = closing[0:train_size,:], closing[train_size:len(closing),:]
print(train)

print("\n ----------------------------")

print(test)
# reshape into X=t and Y=t+1

look_back = 1   # Taking 1 past values to predict a single value

trainX, trainY = create_dataset(train, look_back)

testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]

#Timestamp here is taken as 1 ie looking at the stock of time t we will predict at t+1

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
trainX.shape,trainY.shape,testX.shape,testY.shape
#Creating a call back functionn

class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self,epoch,logs={}):

        if(logs.get('accuracy')>0.95):

            print("\n Reaching 95% so stopping the training now")

            self.model.stop_training = True

            

callbacks = myCallback()
filepath = 'weights'

modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath,monitor = 'val_accuracy', verbose=0, save_freq='epoch',save_best_only=True)
lr_platue = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',factor=0.1,patience=5,verbose=0,mode='max')
def build_model():

    

    model = Sequential()

    

    model.add(LSTM(256, input_shape=(1, look_back), return_sequences=True))

    model.add(Dropout(0.4))

        

    model.add(LSTM(256))

    model.add(Dropout(0.4))

        

    model.add(Dense(64,kernel_initializer="uniform",activation='relu'))        

    model.add(Dense(1,kernel_initializer="uniform",activation='linear'))

    

        

    start = time.time()

    model.compile(loss='mse',optimizer=Adam(lr = 0.0005), metrics=['mean_squared_error'])

    print("Compilation Time : ", time.time() - start)

    return model
model = build_model()

history = model.fit(trainX,trainY,epochs=100,batch_size=128,

                   callbacks = [lr_platue,modelcheckpoint], validation_data = (testX,testY))
import matplotlib.pyplot as plt



plt.plot(history.history['mean_squared_error'])

plt.plot(history.history['val_mean_squared_error'])

plt.title('model mean squared error')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
def model_score(model, X_train, y_train, X_test, y_test):

    trainScore = model.evaluate(X_train, y_train, verbose=0)

    print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    testScore = model.evaluate(X_test, y_test, verbose=0)

    print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

    return trainScore[0], testScore[0]



model_score(model, trainX, trainY , testX, testY)
pred = model.predict(testX)

pred = scaler.inverse_transform(pred) #Performing Denormalization of Data
testY = testY.reshape(testY.shape[0] , 1)

testY = scaler.inverse_transform(testY)
plt.figure(figsize=(15,10))

plt.plot(pred,color='red', label='Prediction')

plt.plot(testY,color='blue', label='Actual')

plt.xlabel('Time')

plt.ylabel('Stock Prices')

plt.title('Model Accuracy with time')

plt.legend(loc='best')

plt.show()