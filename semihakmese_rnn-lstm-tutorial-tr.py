#Importing the Libraries

import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns 



import warnings 

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

dataset_train = pd.read_csv("/kaggle/input/Stock_Price_Train.csv")
dataset_train.head()
dataset_train.info() #1258 verimiz var
train = dataset_train.loc[:,["Open"]].values #Tüm satırlarda Open column seçtik

#Values ile Numpy library geçişi ve reshape ile de 1D array'i 2D array haline getirdik.

train
# Feature Scaling (Normalization) işlemini yapalım 

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,1)) # Değerleri 0 ile 1 arasında scale ettik. 

train_scaled = scaler.fit_transform(train) #Fit ile datamızı 0 ile 1 arasına fitledik. Transform ile ise uygun şekilde dönüştürdük.

train_scaled
plt.plot(train_scaled)

plt.xlabel("Sample")

plt.ylabel("Values")

plt.show()
X_train = []

y_train = []

timesteps = 50

for i in range(timesteps, 1258):

    X_train.append(train_scaled[i-timesteps:i, 0])

    #Burda kaydırarak yapıyoruz yani 50 100 diye timestepler olmayacak 

    #Onun yerine 1 50 arası 2 51 arası gibi olacak ve bir sonrakini tahmini edecek. 

    y_train.append(train_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
#Reshape işlemini yapalm 

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_train
y_train
# #Kütüphanelerin import edilmesi 

# from keras.models import Sequential #Tüm layerları içinde bulunduran modül

# from keras.layers import Dense #Layer yapıları

# from keras.layers import SimpleRNN #RNN için özelleşmiş Kütüphanemiz

# from keras.layers import Dropout #Overfitting engellemek için kullandığımız yapı

#                                 #Bir regularization methodudur.

# #Modelin oluşturulması

# regressor = Sequential()

# 

# #Modele RNN Layer ve Dropout Eklenmesi 

# regressor.add(SimpleRNN(units = 264, activation ="relu",

#                         return_sequences = True, 

#                         input_shape = (X_train.shape[1],1)))

# regressor.add(Dropout(0.2))

# 

# #2. RNN Layer Eklenmesi - Input shape sadece ilk layerda belirtilir.

# regressor.add(SimpleRNN(units = 264, activation ="relu",

#                         return_sequences = True)) 

# regressor.add(Dropout(0.2))

# 

# #2. RNN Layer Eklenmesi - Input shape sadece ilk layerda belirtilir.

# regressor.add(SimpleRNN(units = 128, activation ="relu",

#                         return_sequences = True)) 

# regressor.add(Dropout(0.2))

# 

# 

# #3. RNN Layer Eklenmesi

# regressor.add(SimpleRNN(units = 128, activation ="relu",

#                         return_sequences = True)) 

# regressor.add(Dropout(0.2))

# 

# #5. RNN Layer Eklenmesi

# regressor.add(SimpleRNN(units = 64))

# regressor.add(Dropout(0.15))

# 

# #Output Layer Eklenmesi 

# regressor.add(Dense(units = 1)) # 1 Node'a sahip Layer ekledik = Output

# 

# #Modelin Compile (Derleme) Edilmesi

# regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# 

# #Modelin Fit edilmesi ve eğitilmesi 

# regressor.fit(X_train,y_train, epochs = 200, batch_size = 64) #Burada verileri 32şerli olarak alacak ve her batch için 100 iterasyon yapacaktır. 

#Kütüphanelerin import edilmesi 

from keras.models import Sequential #Tüm layerları içinde bulunduran modül

from keras.layers import Dense #Layer yapıları

from keras.layers import SimpleRNN #RNN için özelleşmiş Kütüphanemiz

from keras.layers import Dropout #Overfitting engellemek için kullandığımız yapı

                                #Bir regularization methodudur.

#Modelin oluşturulması

regressor = Sequential()



regressor.add(SimpleRNN(units = 50,activation='relu', return_sequences = True, input_shape = (X_train.shape[1], 1)))



regressor.add(SimpleRNN(units = 50,activation='relu', return_sequences = True))



regressor.add(SimpleRNN(units = 50,activation='relu', return_sequences = True))



regressor.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True))



regressor.add(SimpleRNN(units = 50,activation='relu', return_sequences = True))



regressor.add(SimpleRNN(units = 50))



regressor.add(Dense(units = 1))



regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')



regressor.fit(X_train, y_train, epochs = 150, batch_size = 100)
#Test datamızın çağrılması

dataset_test = pd.read_csv('/kaggle/input/Stock_Price_Test.csv')

dataset_test.head()
dataset_test.info()
real_stock_price = dataset_test.loc[:, ["Open"]].values

real_stock_price
# Getting the predicted stock price of 2017

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - timesteps:].values.reshape(-1,1)

inputs = scaler.transform(inputs)  # min max scaler

inputs
X_test = []

for i in range(timesteps, 70):

    X_test.append(inputs[i-timesteps:i, 0])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)

predicted_stock_price = scaler.inverse_transform(predicted_stock_price)



# Visualising the results

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')

plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')

plt.title('Google Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('Google Stock Price')

plt.legend()

plt.show()

# epoch = 250 daha güzel sonuç veriyor.
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd 

import math 

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error



import warnings 

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/international-airline-passengers.csv",skipfooter = 5) 

data.head(10)
data.info()
#Görselleştirme

dataset = data.iloc[:,1].values #Numpy array'a cevirdik

plt.plot(dataset)

plt.xlabel("Time")

plt.ylabel("Number of Passengers")

plt.title("International Airline Passengers")

plt.show()
#Reshaping

dataset = dataset.reshape(-1,1) #Bu işlemi yapmazsak (142, ) formunda olur ancak keras hata verebildiği için (142,1) formatına çevirdik

dataset = dataset.astype("float32") #int formatını float haline getirdik.

dataset.shape
#Scaling

scaler = MinMaxScaler(feature_range = (0,1))

dataset = scaler.fit_transform(dataset) # Datamızı 0 ile 1 arasına normalize ettik

#Tüm Neural Networklerde yapılmalıdır. - Hız artar, - Sonuç iyileşir
#Train Test Split

train_size = int(len(dataset)*0.5) #Verilerimizin yarısını train için ayırdık

test_size = len(dataset)- train_size #Kalan yarısını da test için ayırdık

train = dataset[0:train_size,:] # İlk satır ile datanın yarısı arası kadar train verisi

test = dataset[train_size:len(dataset),:] #Diğer yarısını da Test datasına atadık 

print("Train Size : {}, Test Size: {}".format(len(train),len(test)))
time_stemp = 10

dataX = []

dataY = []

for i in range(len(train)-time_stemp-1):

    a = train[i:(i+time_stemp), 0]

    dataX.append(a)

    dataY.append(train[i + time_stemp, 0])

trainX = np.array(dataX)

trainY = np.array(dataY)  
dataX = []

dataY = []

for i in range(len(test)-time_stemp-1):

    a = test[i:(i+time_stemp), 0]

    dataX.append(a)

    dataY.append(test[i + time_stemp, 0])

testX = np.array(dataX)

testY = np.array(dataY)  
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# model

model = Sequential()

model.add(LSTM(10, input_shape=(1, time_stemp))) # 10 lstm neuron(block)

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX, trainY, epochs=50, batch_size=1)
trainPredict = model.predict(trainX)

testPredict = model.predict(testX)

# invert predictions

trainPredict = scaler.inverse_transform(trainPredict)

trainY = scaler.inverse_transform([trainY])

testPredict = scaler.inverse_transform(testPredict)

testY = scaler.inverse_transform([testY])

# calculate root mean squared error

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))

print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))

print('Test Score: %.2f RMSE' % (testScore))
# shifting train

trainPredictPlot = np.empty_like(dataset)

trainPredictPlot[:, :] = np.nan

trainPredictPlot[time_stemp:len(trainPredict)+time_stemp, :] = trainPredict

# shifting test predictions for plotting

testPredictPlot = np.empty_like(dataset)

testPredictPlot[:, :] = np.nan

testPredictPlot[len(trainPredict)+(time_stemp*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions

plt.plot(scaler.inverse_transform(dataset))

plt.plot(trainPredictPlot)

plt.plot(testPredictPlot)

plt.show()
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd 

import math 

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error



import warnings 

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
stocklstm_train = pd.read_csv("/kaggle/input/Stock_Price_Train.csv")

stocklstm_test = pd.read_csv("/kaggle/input/Stock_Price_Test.csv")
stocklstm_train.head(5)
stock_train = stocklstm_train.loc[:,["Open"]].values #Numpy array'e çevirdik.

stock_train
stock_test = stocklstm_test.loc[:,["Open"]].values #Numpy array'e çevirdik.

stock_test
#Reshape 

stock_train = stock_train.reshape(-1,1)

stock_train = stock_train.astype("float32")

stock_train.shape
stock_test = stock_test.reshape(-1,1)

stock_test = stock_test.astype("float32")

stock_test.shape
#Scaling

scaler2 = MinMaxScaler(feature_range = (0,1))

stock_train = scaler2.fit_transform(stock_train) # Datamızı 0 ile 1 arasına normalize ettik

stock_test = scaler2.fit_transform(stock_test)

#Tüm Neural Networklerde yapılmalıdır. - Hız artar, - Sonuç iyileşir

dataset_top = np.concatenate((stock_train, stock_test), axis = 0)
plt.plot(stock_train)

plt.xlabel("Sample")

plt.ylabel("Values")

plt.show()
time_stemp = 10

dataX2 = []

dataY2 = []

for i in range(len(stock_train)-time_stemp-1):

    b = stock_train[i:(i+time_stemp), 0]

    dataX2.append(b)

    dataY2.append(stock_train[i + time_stemp, 0])

trainX2 = np.array(dataX2)

trainY2 = np.array(dataY2)  
dataX2 = []

dataY2 = []

for i in range(len(stock_test)-time_stemp-1):

    b = stock_test[i:(i+time_stemp), 0]

    dataX2.append(b)

    dataY2.append(stock_test[i + time_stemp, 0])

testX2 = np.array(dataX2)

testY2 = np.array(dataY2)  
trainX2 = np.reshape(trainX2, (trainX2.shape[0], 1, trainX2.shape[1]))

testX2 = np.reshape(testX2, (testX2.shape[0], 1, testX2.shape[1]))
model = Sequential()

model.add(LSTM(10, input_shape=(1, time_stemp))) # 10 lstm neuron(block)

model.add(Dense(8))

model.add(Dense(4))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX2, trainY2, epochs=50, batch_size=1)
trainPredict2 = model.predict(trainX2)

testPredict2 = model.predict(testX2)

# invert predictions

trainPredict2 = scaler2.inverse_transform(trainPredict2)

trainY2 = scaler2.inverse_transform([trainY2])

testPredict2 = scaler2.inverse_transform(testPredict2)

testY2 = scaler2.inverse_transform([testY2])

# calculate root mean squared error

trainScore2 = math.sqrt(mean_squared_error(trainY2[0], trainPredict2[:,0]))

print('Train Score: %.2f RMSE' % (trainScore2))

testScore2 = math.sqrt(mean_squared_error(testY2[0], testPredict2[:,0]))

print('Test Score: %.2f RMSE' % (testScore2))
# shifting train

trainPredictPlot2 = np.empty_like(dataset_top)

trainPredictPlot2[:, :] = np.nan

trainPredictPlot2[time_stemp:len(trainPredict2)+time_stemp, :] = trainPredict2

# shifting test predictions for plotting

testPredictPlot2 = np.empty_like(dataset_top)

testPredictPlot2[:, :] = np.nan

testPredictPlot2[len(trainPredict2)+(time_stemp*2)+1:len(dataset_top)-1, :] = testPredict2

# plot baseline and predictions

plt.plot(scaler2.inverse_transform(dataset_top))

plt.plot(trainPredictPlot2)

plt.plot(testPredictPlot2)

plt.show()