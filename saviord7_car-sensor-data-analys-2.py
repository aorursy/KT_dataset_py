import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout   
from keras import optimizers

from sklearn.model_selection import train_test_split
from sklearn import metrics

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df1=pd.read_csv('../input/carsensorsdata/Opel1.csv')
df2=pd.read_csv('../input/carsensorsdata/Opel2.csv')
df3=pd.read_csv('../input/carsensorsdata/Peugeot1.csv')
df4=pd.read_csv('../input/carsensorsdata/Peugeot2.csv')

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#df1[['AltitudeVariation', 'VehicleSpeedInstantaneous', 'VehicleSpeedAverage', 'VehicleSpeedVariance','VehicleSpeedVariation', 'LongitudinalAcceleration', 'EngineLoad', 'EngineCoolantTemperature', 'ManifoldAbsolutePressure', 'EngineRPM','MassAirFlow','IntakeAirTemperature','VerticalAcceleration','FuelConsumptionAverage']] = sc.fit_transform(df1[['AltitudeVariation', 'VehicleSpeedInstantaneous', 'VehicleSpeedAverage', 'VehicleSpeedVariance','VehicleSpeedVariation', 'LongitudinalAcceleration', 'EngineLoad', 'EngineCoolantTemperature', 'ManifoldAbsolutePressure', 'EngineRPM','MassAirFlow','IntakeAirTemperature','VerticalAcceleration','FuelConsumptionAverage']])
#df2[['AltitudeVariation', 'VehicleSpeedInstantaneous', 'VehicleSpeedAverage', 'VehicleSpeedVariance','VehicleSpeedVariation', 'LongitudinalAcceleration', 'EngineLoad', 'EngineCoolantTemperature', 'ManifoldAbsolutePressure', 'EngineRPM','MassAirFlow','IntakeAirTemperature','VerticalAcceleration','FuelConsumptionAverage']] = sc.fit_transform(df2[['AltitudeVariation', 'VehicleSpeedInstantaneous', 'VehicleSpeedAverage', 'VehicleSpeedVariance','VehicleSpeedVariation', 'LongitudinalAcceleration', 'EngineLoad', 'EngineCoolantTemperature', 'ManifoldAbsolutePressure', 'EngineRPM','MassAirFlow','IntakeAirTemperature','VerticalAcceleration','FuelConsumptionAverage']])
#df3[['AltitudeVariation', 'VehicleSpeedInstantaneous', 'VehicleSpeedAverage', 'VehicleSpeedVariance','VehicleSpeedVariation', 'LongitudinalAcceleration', 'EngineLoad', 'EngineCoolantTemperature', 'ManifoldAbsolutePressure', 'EngineRPM','MassAirFlow','IntakeAirTemperature','VerticalAcceleration','FuelConsumptionAverage']] = sc.fit_transform(df3[['AltitudeVariation', 'VehicleSpeedInstantaneous', 'VehicleSpeedAverage', 'VehicleSpeedVariance','VehicleSpeedVariation', 'LongitudinalAcceleration', 'EngineLoad', 'EngineCoolantTemperature', 'ManifoldAbsolutePressure', 'EngineRPM','MassAirFlow','IntakeAirTemperature','VerticalAcceleration','FuelConsumptionAverage']])
#df4[['AltitudeVariation', 'VehicleSpeedInstantaneous', 'VehicleSpeedAverage', 'VehicleSpeedVariance','VehicleSpeedVariation', 'LongitudinalAcceleration', 'EngineLoad', 'EngineCoolantTemperature', 'ManifoldAbsolutePressure', 'EngineRPM','MassAirFlow','IntakeAirTemperature','VerticalAcceleration','FuelConsumptionAverage']] = sc.fit_transform(df4[['AltitudeVariation', 'VehicleSpeedInstantaneous', 'VehicleSpeedAverage', 'VehicleSpeedVariance','VehicleSpeedVariation', 'LongitudinalAcceleration', 'EngineLoad', 'EngineCoolantTemperature', 'ManifoldAbsolutePressure', 'EngineRPM','MassAirFlow','IntakeAirTemperature','VerticalAcceleration','FuelConsumptionAverage']])

df_Dataset=pd.concat([df1,df2,df3,df4],axis=0)
df_Dataset.head(400)
if (df_Dataset.isnull().sum().sum()!=0):
    df_Dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
df_Dataset.head()
y = df_Dataset['drivingStyle']
X = df_Dataset.drop(['drivingStyle','traffic','roadSurface'], axis=1)


train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
X

test_X
from sklearn.preprocessing import MinMaxScaler

feature_scaler = MinMaxScaler()
train_X = feature_scaler.fit_transform(train_X)
test_X = feature_scaler.transform(test_X)
Xtrain, Xval, Ytrain, Yval = train_test_split(train_X, train_y, test_size=0.2, random_state=5) #
model = Sequential()

model.add(Dense(train_X.shape[1] * 128, input_dim=train_X.shape[1], activation='relu'))   

model.add(Dense(1, activation='sigmoid'))
#model.add(Dense(1, activation='softmax'))

model.summary()
opt = optimizers.SGD(learning_rate=0.1)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

history = model.fit(train_X, train_y, epochs=10000, validation_data=(Xval, Yval), batch_size=10)
test_loss, test_acc = model.evaluate(test_X,  test_y, verbose=2)

print('\nAccuracy on test data:', test_acc)
plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.ylim((0, 1))
plt.legend()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.ylim((0, 1))
plt.legend()
prediction=model.predict(test_X)
np.around(prediction, decimals=4, out=None)
df = pd.DataFrame(data=prediction,  columns=["DrivingRating"])
np.around(df, decimals=3, out=None)


prediction = model.predict(test_X) > 0.5
prediction = (prediction > 0.5) * 1
accuracy_nn = metrics.accuracy_score(test_y, prediction) * 100
print(accuracy_nn)