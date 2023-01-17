#Let's start with adding libraries I will use

import numpy as np 
import pandas as pd 
import seaborn as sns

# plotly
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_dark"
from plotly.subplots import make_subplots

# word cloud library
from wordcloud import WordCloud

# matplotlib
import matplotlib.pyplot as plt

from collections import Counter

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv("../input/covidtr/Covid_vaka.csv")

# I want to see total hospital usage so I will add new column as "Hastane_Kullanımı".
df["Hastane_Kullanımı"] = df.Toplan_Entübe + df.Toplam_Yogun_Bakim
df.tail()
df.describe()


# Then lets start with show change of "Vaka","Test_Sayisi","Vaka_Test" and "Vaka","Vefat","Olum_Oran" date by date 
# import graph objects as "go"
import plotly.graph_objs as go

# Creating graph1
graph1 = go.Scatter(
                    x = df.Tarih,
                    y = df.Vaka_Test,
                    mode = "lines+markers",
                    name = "Tarihe Göre Vaka Test Oranı ",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= df.Vaka)


data = [graph1]
layout = dict(title = 'Tarihe Göre Vaka Analizi',
              xaxis= dict(title= 'COVID',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)

import plotly.graph_objs as go

# Creating graph2
graph2 = go.Scatter(
                    x = df.Tarih,
                    y = df.Aktif_Vaka,
                    mode = "lines+markers",
                    name = "Aktif Vaka Durumu",
                    marker = dict(color = 'rgba(139 ,69, 19, 1.0)'),
                    text= df.Vaka)

data = [graph2]
layout = dict(title = 'Aktif Vaka Durumu',
              xaxis= dict(title= 'COVID',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
# Creating graph3
graph3 = go.Scatter(
                    x = df.Tarih,
                    y = df.Toplan_Entübe,
                    mode = "lines+markers",
                    name = "Entübe Hasta",
                    marker = dict(color = 'rgba(99 ,69, 72, 1.0)'),
                    text= df.Toplan_Entübe)
graph4 = go.Scatter(
                    x = df.Tarih,
                    y = df.Toplam_Yogun_Bakim,
                    mode = "lines+markers",
                    name = "Yoğun Bakım Hasta",
                    marker = dict(color = 'rgba(21 ,98, 120, 1.0)'),
                    text= df.Toplam_Yogun_Bakim)
graph5 = go.Scatter(
                    x = df.Tarih,
                    y = df.Hastane_Kullanımı,
                    mode = "lines+markers",
                    name = "Hastane Kullanımı",
                    marker = dict(color = 'rgba(21 ,98, 19, 1.0)'),
                    text= df.Hastane_Kullanımı)

data = [graph3,graph4,graph5]
layout = dict(title = 'Hastane Kullanım Değişimi',
              xaxis= dict(title= 'COVID',ticklen= 10,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
# Train datamızı yüklüyoruz.
dataset_train = pd.read_csv("../input/covid-train/Covid_train.csv")
dataset_train.head()
train = dataset_train.loc[:, ["Aktif_Vaka"]].values
train
# Feature Scaling(normalizasyon)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
train_scaled = scaler.fit_transform(train)
train_scaled
plt.plot(train_scaled)
plt.show()
# 50 timesteps ve 1 output ile data structure
X_train = []
y_train = []
timesteps = 2
for i in range(timesteps, 58):
    X_train.append(train_scaled[i-timesteps:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train
y_train
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 250, batch_size = 32)
# Test datamızı yüklüyoruz.
dataset_test = pd.read_csv("../input/covidtest/Covid_test.csv")
dataset_test.head()
real_vaka = dataset_test.loc[:, ["Aktif_Vaka"]].values
real_vaka
# Tahmin edilen Aktif Vaka 
dataset_total = pd.concat((dataset_train['Aktif_Vaka'], dataset_test['Aktif_Vaka']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - timesteps:].values.reshape(-1,1)
inputs = scaler.transform(inputs)  # min max scaler
inputs
X_test = []
for i in range(timesteps, 27):
    X_test.append(inputs[i-timesteps:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_vaka = regressor.predict(X_test)
predicted_vaka = scaler.inverse_transform(predicted_vaka) #scale yapma diyorum.

# Sonucu Görselleştirme
plt.plot(real_vaka, color = 'red', label = 'Gerçek Vaka Durumu')
plt.plot(predicted_vaka, color = 'blue', label = 'Tahmin Edilen Vaka Durumu')
plt.title('Aktif Vaka')
plt.xlabel('Zaman')
plt.ylabel('Aktif Vaka')
plt.legend()
plt.show()
import numpy
import pandas as pd 
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
dataset = df.iloc[:,12].values
plt.plot(dataset)
plt.xlabel("Tarih")
plt.ylabel("Aktif Vaka")
plt.title("Aktif Vaka Değişimi")
plt.show()
dataset = dataset.reshape(-1,1)
dataset = dataset.astype("float32")
dataset.shape
# Değerlerimi normalize ediyorum çünkü 0 ile 80000 arasında kaybolacak verilerin önüne geçip veri kaybetmek istemiyorum.
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# Verisetimi train ve test olarak iki bölüme ayırıyorum. Datasetin %30 unu kuracağım modeli test etmek için %70 ini de modelimi eğitmek için kullancam.
train_size = int(len(dataset) * 0.70)
test_size = len(dataset) - train_size
train = dataset[0:train_size,:]
test = dataset[train_size:len(dataset),:]
print("train size: {}, test size: {} ".format(len(train), len(test)))
time_stemp = 10
dataX = []
dataY = []
for i in range(len(train)-time_stemp-1):
    a = train[i:(i+time_stemp), 0]
    dataX.append(a)
    dataY.append(train[i + time_stemp, 0])
trainX = numpy.array(dataX)
trainY = numpy.array(dataY) 
dataX = []
dataY = []
for i in range(len(test)-time_stemp-1):
    a = test[i:(i+time_stemp), 0]
    dataX.append(a)
    dataY.append(test[i + time_stemp, 0])
testX = numpy.array(dataX)
testY = numpy.array(dataY)

trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
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
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[time_stemp:len(trainPredict)+time_stemp, :] = trainPredict
# shifting test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(time_stemp*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.legend()
plt.show()
