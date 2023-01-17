from math import sqrt
from numpy import concatenate
from numpy import datetime64
from numpy import timedelta64
import numpy as np
import pandas as pd
from pandas import read_csv
from pandas import merge
from pandas import DataFrame
from pandas import concat
from pandas import to_datetime
from pandas import DateOffset
from datetime import datetime
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.preprocessing.sequence import TimeseriesGenerator

import os
print(os.listdir("../input"))


dataset = read_csv('../input/watershed-water-quality-data.csv')
dataset.head()




twelve_am_data_set = dataset[['Date','Turbidity(NTU) at 12AM','Coliform, Fecal(fc/100mL)']]
four_am_data_set = dataset[['Date','Turbidity(NTU) at 4AM','Coliform, Fecal(fc/100mL)']]
eight_am_data_set = dataset[['Date','Turbidity(NTU) at 8AM','Coliform, Fecal(fc/100mL)']]
twelve_pm_data_set = dataset[['Date','Turbidity(NTU) at 12PM','Coliform, Fecal(fc/100mL)']]
four_pm_data_set = dataset[['Date','Turbidity(NTU) at 4PM','Coliform, Fecal(fc/100mL)']]
eight_pm_data_set = dataset[['Date','Turbidity(NTU) at 8PM','Coliform, Fecal(fc/100mL)']]

#Change Date by adding hour of measurement

twelve_am_data_set['Date'] = to_datetime(twelve_am_data_set['Date'])

four_am_data_set['Date'] = to_datetime(four_am_data_set['Date'])
four_am_data_set['Date'] = four_am_data_set['Date'] + DateOffset(hours=4)

eight_am_data_set['Date'] = to_datetime(eight_am_data_set['Date'])
eight_am_data_set['Date'] = eight_am_data_set['Date'] + DateOffset(hours=8)

twelve_pm_data_set['Date'] = to_datetime(twelve_pm_data_set['Date'])
twelve_pm_data_set['Date'] = twelve_pm_data_set['Date'] + DateOffset(hours=12)

four_pm_data_set['Date'] = to_datetime(four_pm_data_set['Date'])
four_pm_data_set['Date'] = four_pm_data_set['Date'] + DateOffset(hours=16)

eight_pm_data_set['Date'] = to_datetime(eight_pm_data_set['Date'])
eight_pm_data_set['Date'] = eight_pm_data_set['Date'] + DateOffset(hours=20)

twelve_am_data_set.columns = ["Date", "Turbidity",'Fecal']
four_am_data_set.columns = ["Date", "Turbidity",'Fecal']
eight_am_data_set.columns = ["Date", "Turbidity",'Fecal']
twelve_pm_data_set.columns = ["Date", "Turbidity",'Fecal']
four_pm_data_set.columns = ["Date", "Turbidity",'Fecal']
eight_pm_data_set.columns = ["Date", "Turbidity",'Fecal']

complete_data_set= concat([twelve_am_data_set,four_am_data_set, eight_am_data_set,twelve_pm_data_set,four_pm_data_set,eight_pm_data_set], axis=0, join='outer', ignore_index=False)

complete_data_set = complete_data_set.sort_values(by=['Date'])
complete_data_set.head(10)

complete_data_set.Turbidity.unique()
complete_data_set.Fecal.unique()
pd.value_counts(complete_data_set['Fecal']).plot.bar(figsize=(10,5))

complete_data_set = complete_data_set.drop(['Fecal'], axis=1)
complete_data_set.info()
complete_data_set['Turbidity'].plot.hist(bins=50,figsize = (10,10))
df = complete_data_set.set_index('Date')
df['Turbidity'].plot(figsize = (20,10))
df[:180]['Turbidity'].plot(figsize = (20,10))

nan_rows = df[df['Turbidity'].isnull()]
nan_rows
df= df.fillna(method='ffill')
df.isna().sum()

#remove first row which is also null
df = df.iloc[1:]
df.isna().sum()
values = df.values
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(values)

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
train_data_gen = TimeseriesGenerator(train, train,
	length=2, sampling_rate=1,stride=1,
    batch_size=3)
test_data_gen = TimeseriesGenerator(test, test,
	length=2, sampling_rate=1,stride=1,
	batch_size=1)
model = Sequential()
model.add(LSTM(4, input_shape=(2, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit_generator(train_data_gen, epochs=50).history
model.evaluate_generator(test_data_gen)
trainPredict = model.predict_generator(train_data_gen)
trainPredict.shape
testPredict = model.predict_generator(test_data_gen)
testPredict.shape

#return values to their pre-normalized form
inv_trainPredict = scaler.inverse_transform(trainPredict)
inv_testPredict= scaler.inverse_transform(testPredict)


trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[2:len(trainPredict)+2, :] = inv_trainPredict


testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(2*2):len(dataset), :] = inv_testPredict

pyplot.figure(figsize = (20, 10))
pyplot.plot(trainPredictPlot, label="trainPredict")
pyplot.plot(testPredictPlot,label="testPredict")
pyplot.plot(df['Turbidity'].values, label="real")
pyplot.xlabel("Date")
pyplot.ylabel("Turbidity")
pyplot.title("Comparison ")
pyplot.legend()
pyplot.show()




pyplot.figure(figsize = (20, 10))
pyplot.plot(trainPredictPlot[6600:6630], label="trainPredict")
pyplot.plot(testPredictPlot[6600:6630],label="testPredict")
pyplot.plot(df[6600:6630]['Turbidity'].values, label="real")
pyplot.xlabel("4-hour measurements")
pyplot.ylabel("Turbidity")
pyplot.title("Comparison ")
pyplot.legend()
pyplot.grid(True,which='both')
pyplot.show()

