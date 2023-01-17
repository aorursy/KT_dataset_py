#installing yfinance for fetching past stock datas

!pip install yfinance
#importing necessary libraries

import yfinance as yf

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Dropout



from sklearn.preprocessing import MinMaxScaler
#Building helper funtion for fetcching stock data

def hist(tick):

    ticker = yf.Ticker(tick + '.NS')  #NS is used for fetching NSE(indian) stock datas. For US stocks use without NS. Refer to yfinance documentation for usage

    

    data = ticker.history(period = 'max') #Period used here is max i.e. fetching data from the date of listing to present

    

    return data





#Checking if the function is working

hist('SBIN').tail(10)
#Fetching ITC data

ITC = hist('ITC')

ITC.head()
#Plotting close price of ITC



plt.figure(figsize=(16,8))

plt.plot(ITC['Close'])

plt.title('Historical Closing Price of ITC',fontsize = 15)

plt.xlabel('Time',fontsize = 15)

plt.ylabel('Price',fontsize = 15)

plt.show()
#Extracting closing price and storing it in a dataframe 'df'

df = pd.DataFrame(ITC['Close'])

df = df.reset_index()

df = df.drop(['Date'],axis = 1)

df
#Scaling the values of closing price using MinMax Scaler since LSTM is sensitive to ranges of data

scaler = MinMaxScaler()

scaled_data  = scaler.fit_transform(np.array(df).reshape(-1,1))

scaled_data
#Splitting the dataset into train and test

training_size = int(len(scaled_data) * 0.7) #taking 70% of total data to be training set and rest to be test set

test_size = int(len(scaled_data) - training_size)

train,test = scaled_data[0:training_size],scaled_data[training_size:len(scaled_data)]
#Generating dataset to train

X_train = [] #independent

y_train = [] #dependent



for i in range(100,len(train)):

    X_train.append(train[i-100:i,0]) #100 days historical data

    y_train.append(train[i,0]) #corresponding next day's output

    

    



X_train,y_train = np.array(X_train),np.array(y_train)

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1) 
#Lets see the dimensions of both X_train,y_train

print('The shape of X_train is {0} and The shape of y_train is {1}'.format(X_train.shape,y_train.shape))
model=Sequential()

model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))

model.add(Dropout(0.2))

model.add(LSTM(50,return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(50))

model.add(Dense(1))

#model summary

model.summary()
model.compile(loss = 'mean_squared_error',optimizer='adam')

model.fit(X_train,y_train,epochs = 5,batch_size=64)
#Since test set will be used to predict,lets preprocess it as similar to train set

X_test = []

y_test = []



for i in range(100,len(test)):

    X_test.append(test[i - 100 : i,0])

    y_test.append(test[i,0])



    

X_test,y_test = np.array(X_test),np.array(y_test)

X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
#Lets see the dimensions of both X_test,y_test

print('The shape of X_test is {0} and The shape of y_test is {1}'.format(X_test.shape,y_test.shape))
#Making prediction on train and test set

predicted_train = model.predict(X_train)

predicted_test = model.predict(X_test)



#Since the predicted values will be in scaled form so we have to inverse the values inorder to be in original form

predicted_train = scaler.inverse_transform(predicted_train)

predicted_test = scaler.inverse_transform(predicted_test)
print('The shape of predicted train is {0} and shape of predicted test values is {1}'.format(predicted_train.shape,predicted_test.shape))
#importing the MSE module from sklearn 

from sklearn.metrics import mean_squared_error



def RMSE(val,pred):

    return np.sqrt(mean_squared_error(val,pred))
print('RMSE of training data is ',RMSE(scaler.inverse_transform(y_train.reshape(-1,1)),predicted_train))
print('RMSE of test data is ',RMSE(scaler.inverse_transform(y_test.reshape(-1,1)),predicted_test))
#Plotting test data

plt.figure(figsize = (16,8))

plt.plot(scaler.inverse_transform(y_test.reshape(-1,1)),label = 'true value')

plt.plot(predicted_test,label = 'predicted value')

plt.legend()

plt.show()