import pandas as pd
link ='https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_recovered_global.csv&filename=time_series_covid19_recovered_global.csv'
data = pd.read_csv(link, header=0)
data.head()
import datetime
date = [i for i in data.columns[4:]]
sum_of_global = [sum(data[i]) for i in data.columns[4:]]
series = pd.DataFrame(sum_of_global, columns=['Number'], index=date)
series.index = pd.to_datetime(series.index.str.split().str[0], format='%m/%d/%y')
print('Chuỗi thời gian về tổng số các người bệnh Covid-19 được phục hồi bệnh trên toàn thế giới')
series
series.Number=series-series.shift()
series = series.dropna()
print('Chuỗi thời gian về số các người bệnh Covid-19 được phục hồi bệnh trên toàn thế giới dao động mỗi ngày')
series
import numpy as np
import matplotlib.pyplot as plt
series.plot(figsize = (10,8), style = 'o-', label = 'Y')
plt.title('Chuỗi thời gian về số người bệnh Covid-19 được phục hồi bệnh trên toàn thế giới dao động mỗi ngày', fontsize=20)
legend = plt.legend(loc = 'upper center', shadow = True, fontsize = 'x-large')
legend.get_frame().set_facecolor('C')
series.hist()
print(series.describe())
def create_data(data, look_back):
    data_frame = data
    cols = ['X']
    for i in range(look_back):
        data_frame = pd.merge(data_frame, data.shift(i+1), how = 'left', left_index = True, right_index = True)
        cols.append('X%s'%(i+1))
    data_frame.columns = cols
    data_frame=data_frame.rename(columns = {'X':'y'})
    return data_frame.iloc[look_back:, :]
look_back =1
data_series = create_data(series, 1)
data_series
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

scaler = MinMaxScaler(feature_range = (0, 1))
data_series_scaler = scaler.fit_transform(data_series)
train_size = int(data_series.shape[0]*0.7)
test_size = data_series.shape[0] - train_size
train, test = data_series_scaler[0:train_size, :], data_series_scaler[train_size:data_series.shape[0], :]
print('Kích thước Train:',train.shape)
print('Kích thước Test:',test.shape)
trainX, trainY = train[:, 1:], train[:, 0]
testX, testY = test[:, 1:], test[:, 0]
# reshape input to be [samples, time steps, features]
trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
testX = testX.reshape(testX.shape[0], 1, testX.shape[1])
from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(5, input_shape = (1, look_back)))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.fit(trainX, trainY, epochs = 500, batch_size = 30, verbose = 0, shuffle = False)
# Make prediction
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
import math
# Invert prediction
trainPredictInv = scaler.inverse_transform(np.concatenate((trainPredict, trainX.reshape(trainX.shape[0], -1)), axis = 1))[:, 0]
trainYInv = data_series.iloc[:train_size, 0].values
testPredictInv = scaler.inverse_transform(np.concatenate((testPredict, testX.reshape(testX.shape[0], -1)), axis = 1))[:, 0]
testYInv = data_series.iloc[train_size:, 0].values
#Mse
mse_train = math.sqrt(mean_squared_error(trainYInv[:-1], trainPredictInv[1:]))
mse_test = math.sqrt(mean_squared_error(testYInv[:-1], testPredictInv[1:]))
print('Train MSE: %s' %mse_train)
print('Test MSE: %s' %mse_test)
plt.figure(figsize = (12, 8))
plt.plot(trainYInv[:-1], label = 'Thực tế')
plt.plot(trainPredictInv[1:], 'k--', label = 'Dự đoán')
plt.title('Dữ liệu Train')
legend = plt.legend(loc = 'upper center', shadow = True, fontsize = 'x-large')
legend.get_frame().set_facecolor('C')
plt.figure(figsize = (12, 5))
plt.plot(testYInv[:-1], label = 'Thực tế')
plt.plot(testPredictInv[1:], 'k--', label = 'Dự đoán')
plt.title('Dữ liệu Test')
legend = plt.legend(loc = 'upper center', shadow = True, fontsize = 'x-large')
legend.get_frame().set_facecolor('C')
import matplotlib.collections as ml
n_predict=31
look_back=1
for i in range(n_predict):
    forecastY = testY.reshape(-1, 1)
    Ypredict = model.predict(forecastY[-look_back:].reshape(1,1,look_back))
    forecastY = np.concatenate((forecastY, Ypredict), axis = 0)
forecastY = forecastY[-(n_pred+look_back+1):]
inversePredict = [forecastY[-(n_predict+i):] if i == 0 else forecastY[-(n_predict+i):-i] for i in range(look_back+1)]
inversePredict = np.concatenate(inversePredict, axis = 1)
Ypredict = scaler.inverse_transform(inversePredict)[:, 0]

date_predict=[i for i in range(data_series[train_size:].index[-1].day+1,data_series[train_size:].index[-1].day+n_predict+1)]
series = pd.DataFrame(data={'Ngày':date_predict,'Giá trị dự đoán':Ypredict})
print(series)

Ymerge = np.concatenate((testPredictInv, Ypredict))
Ygraph = np.concatenate((np.arange(Ymerge.shape[0]).reshape(Ymerge.shape[0], -1), Ymerge.reshape(Ymerge.shape[0], -1)), axis = 1)
fig, ax = plt.subplots(figsize = (24, 8))
line_segments = ml.LineCollection([Ygraph[:47], Ygraph[47:]], colors = ['b', 'r'], linestyle = ['solid', 'dashdot'], linewidth = 2)
ax.add_collection(line_segments)
ax.autoscale()
ax.set_ylabel('Giá trị')
ax.set_xlabel('Ngày')

plt.title('Dự đoán số người phục hồi bệnh Covid19 trên toàn thế giới dao động trong %d ngày kế tiếp'%Ypredict.shape[0], fontsize = 20)    
import pandas as pd
import datetime
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#create data
link = 'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv&filename=time_series_covid19_confirmed_global.csv'
data_confirmed = pd.read_csv(link, header=0)
print('Chuỗi thời gian về tổng số các người bệnh Covid-19 trên toàn thế giới')
print(data_confirmed )
date = [i for i in data_confirmed.columns[4:]]
sum_of_global = [sum(data_confirmed[i]) for i in data_confirmed.columns[4:]]
series_confirmed = pd.DataFrame(sum_of_global, columns=['Number'], index=date)
series_confirmed.index = pd.to_datetime(series_confirmed.index.str.split().str[0], format='%m/%d/%y')
series_confirmed.Number=series_confirmed-series_confirmed.shift()
series_confirmed = series_confirmed.dropna()
print('Chuỗi thời gian về số người bệnh Covid-19 trên toàn thế giới dao động mỗi ngày')
print(series_confirmed)
#data_description
series_confirmed.plot(figsize = (15,8), style = 'o-', label = 'Y')
plt.title('Chuỗi thời gian về số người bệnh Covid-19 trên toàn thế giới dao động mỗi ngày', fontsize=20)
legend = plt.legend(loc = 'upper center', shadow = True, fontsize = 'x-large')
legend.get_frame().set_facecolor('C')
print('Thống kê mô tả')
print(series_confirmed.describe())
#creat_series_shift
look_back = 1
data_series_confirmed = create_data(series_confirmed, 1)
scaler = MinMaxScaler(feature_range = (0, 1))
data_series_scaler_confirmed = scaler.fit_transform(data_series_confirmed)
#split into train and test sets
train_size_confirmed = int(data_series_confirmed.shape[0]*0.7)
test_size_confirmed = data_series_confirmed.shape[0] - train_size_confirmed
train_confirmed, test_confirmed = data_series_scaler_confirmed[0:train_size_confirmed, :], data_series_scaler_confirmed[train_size_confirmed:data_series_confirmed.shape[0], :]
#print('Kích thước Train:',train_confirmed.shape)
#print('Kích thước Test:',test_confirmed.shape)
trainX_confirmed, trainY_confirmed = train_confirmed[:, 1:], train_confirmed[:, 0]
testX_confirmed, testY_confirmed = test_confirmed[:, 1:], test_confirmed[:, 0]
# reshape input to be [samples, time steps, features]
trainX_confirmed = trainX_confirmed.reshape(trainX_confirmed.shape[0], 1, trainX_confirmed.shape[1])
testX_confirmed = testX_confirmed.reshape(testX_confirmed.shape[0], 1, testX_confirmed.shape[1])
from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(5, input_shape = (1, look_back)))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.fit(trainX_confirmed, trainY_confirmed, epochs = 500, batch_size = 25, verbose = 0, shuffle = False)
# Make prediction
trainPredict_confirmed = model.predict(trainX_confirmed)
testPredict_confirmed = model.predict(testX_confirmed)
# Invert prediction
trainPredictInv_confirmed = scaler.inverse_transform(np.concatenate((trainPredict_confirmed, trainX_confirmed.reshape(trainX_confirmed.shape[0], -1)), axis = 1))[:, 0]
trainYInv_confirmed = data_series_confirmed.iloc[:train_size_confirmed, 0].values
testPredictInv_confirmed = scaler.inverse_transform(np.concatenate((testPredict_confirmed, testX_confirmed.reshape(testX_confirmed.shape[0], -1)), axis = 1))[:, 0]
testYInv_confirmed = data_series_confirmed.iloc[train_size_confirmed:, 0].values
#Calculate MSE
mse_train_confirmed = math.sqrt(mean_squared_error(trainYInv_confirmed[:-1], trainPredictInv_confirmed[1:]))
mse_test_confirmed = math.sqrt(mean_squared_error(testYInv_confirmed[:-1], testPredictInv_confirmed[1:]))
print('Train MSE: %s' %mse_train_confirmed)
print('Test MSE: %s' %mse_test_confirmed)
#Draw train
print('Kích thước Train:',train_size_confirmed)
plt.figure(figsize = (12, 8))
plt.plot(trainYInv_confirmed[:-1], label = 'Thực tế')
plt.plot(trainPredictInv_confirmed[1:], 'k--', label = 'Dự đoán')
plt.title('Dữ liệu Train')
legend = plt.legend(loc = 'upper center', shadow = True, fontsize = 'x-large')
legend.get_frame().set_facecolor('C')
#Draw test
print('Kích thước Test:',train_size_confirmed)
plt.figure(figsize = (12, 8))
plt.plot(testYInv_confirmed[:-1], label = 'Thực tế')
plt.plot(testPredictInv_confirmed[1:], 'k--', label = 'Dự đoán')
plt.title('Dữ liệu Test')
legend = plt.legend(loc = 'upper center', shadow = True, fontsize = 'x-large')
legend.get_frame().set_facecolor('C')
import matplotlib.collections as ml
n_predict=31
look_back=1
for i in range(n_predict):
    forecastY_confirmed = testY_confirmed.reshape(-1, 1)
    Ypredict_confirmed = model.predict(forecastY_confirmed[-look_back:].reshape(1,1,look_back))
    forecastY_confirmed = np.concatenate((forecastY_confirmed, Ypredict_confirmed), axis = 0)
forecastY_confirmed = forecastY_confirmed[-(n_predict+look_back+1):]
inversePredict_confirmed = [forecastY_confirmed[-(n_predict+i):] if i == 0 else forecastY_confirmed[-(n_predict+i):-i] for i in range(look_back+1)]
inversePredict_confirmed = np.concatenate(inversePredict_confirmed, axis = 1)
Ypredict_confirmed = scaler.inverse_transform(inversePredict_confirmed)[:, 0]
date_predict=[i for i in range(data_series_confirmed[train_size_confirmed:].index[-1].day+1,data_series_confirmed[train_size_confirmed:].index[-1].day+n_predict+1)]
series_confirmed = pd.DataFrame(data={'Ngày':date_predict,'Giá trị dự đoán':Ypredict_confirmed})
print(series_confirmed)
Ymerge_confirmed = np.concatenate((testPredictInv_confirmed, Ypredict_confirmed))
Ygraph_confirmed = np.concatenate((np.arange(Ymerge_confirmed.shape[0]).reshape(Ymerge_confirmed.shape[0], -1), Ymerge_confirmed.reshape(Ymerge_confirmed.shape[0], -1)), axis = 1)
fig, ax = plt.subplots(figsize = (24, 8))
line_segments_confirmed = ml.LineCollection([Ygraph_confirmed[:47], Ygraph_confirmed[47:]], colors = ['b', 'r'], linestyle = ['solid', 'dashdot'], linewidth = 2)
ax.add_collection(line_segments_confirmed)
ax.autoscale()
ax.set_ylabel('Values')
ax.set_xlabel('Day')
plt.title('Dự đoán số người mắc bệnh Covid19 trên toàn thế giới dao động trong %d ngày kế tiếp'%Ypredict_confirmed.shape[0], fontsize = 20) 