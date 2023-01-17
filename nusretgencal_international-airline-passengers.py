# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings

warnings.filterwarnings("ignore") 



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.models import Sequential # model oluşturuyoruz

from keras.layers import Dense, LSTM # layer için , lstm ve dense output için

from sklearn.preprocessing import MinMaxScaler # scale etmemize yarıyor

from sklearn.metrics import mean_squared_error # modeli karşılaştırdığımız yöntem
data = pd.read_csv('/kaggle/input/international-airline-passengers/international-airline-passengers.csv',skipfooter=5)

data.head()
data.shape

data.iloc[:,1].values
dataset = data.iloc[:,1].values

plt.plot(dataset)

plt.xlabel("time")

plt.ylabel("passenger")

plt.show()
dataset.shape
dataset = dataset.reshape(-1,1) # dataset'in shape'ini (142,1) olarak tanımlıyoruz bazen sorun çıkabilir

dataset = dataset.astype("float32")

dataset.shape
#scaling

scaler = MinMaxScaler(feature_range = (0,1))

dataset = scaler.fit_transform(dataset)

dataset
train_size = int(len(dataset) * 0.50) # dataset'in boyututun yarısıyla 71 71

test_size = len(dataset) - train_size 

train = dataset[0:train_size,:] # train'e dataset'imdeki verileri al

test = dataset[train_size:len(dataset),:] # test'e dataset'imin boyutundan sonrasını al

print("train size {}, teest size {}".format(train_size, test_size))

print("train size {}, teest size {}".format(len(train), len(test)))





timestep = 10

datax = []

datay = []

for i in range(len(train)-timestep-1): ## dataları timestep = 10 step olarak 11. output yaparak train ettirmeye çalışıyoruz. 10 tanesini datax'e at 11. datay'e at gibi..

    a = train[i:(i+timestep), 0]

    datax.append(a)

    datay.append(train[i + timestep, 0])

x_train = np.array(datax)

y_train = np.array(datay) 
dataX = []

dataY = []

for i in range(len(test)-timestep-1):

    a = test[i:(i+timestep), 0]

    dataX.append(a)

    dataY.append(test[i + timestep, 0])

x_test = np.array(dataX)

y_test = np.array(dataY)  
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))

x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1])) # 3 boyut yapıyoruz diyebiliriz...
print(x_train.shape)

print(x_test.shape) # 71 tane input vardı 10 step ile y'e aldığımızı düşünelim  

print(y_train.shape)



# model

model = Sequential()

model.add(LSTM(10, activation = 'relu', input_shape=(1, timestep))) # 10 lstm neuron(block) 1 layerda 10 tane lstm olsun

model.summary()

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='Adam')

model.fit(x_train, y_train, epochs=100, batch_size=1)
trainPredict = model.predict(x_train)

testPredict = model.predict(x_test)

# invert predictions

trainPredict = scaler.inverse_transform(trainPredict)

y_train = scaler.inverse_transform([y_train])

testPredict = scaler.inverse_transform(testPredict)

y_test = scaler.inverse_transform([y_test])

# calculate root mean squared error

trainScore = math.sqrt(mean_squared_error(y_train[0], trainPredict[:,0]))

print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(y_test[0], testPredict[:,0]))

print('Test Score: %.2f RMSE' % (testScore))
# shifting train

trainPredictPlot = np.empty_like(dataset)

trainPredictPlot[:, :] = np.nan

trainPredictPlot[timestep:len(trainPredict)+timestep, :] = trainPredict

# shifting test predictions for plotting

testPredictPlot = np.empty_like(dataset)

testPredictPlot[:, :] = np.nan

testPredictPlot[len(trainPredict)+(timestep*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions

plt.plot(scaler.inverse_transform(dataset))

plt.plot(trainPredictPlot)

plt.plot(testPredictPlot)

plt.legend()

plt.show()