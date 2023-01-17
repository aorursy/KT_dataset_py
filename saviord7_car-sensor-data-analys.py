import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout   
from keras import optimizers

from sklearn.model_selection import train_test_split
from sklearn import metrics


df1=pd.read_csv('../input/carsensorsdata/Opel1.csv')
df2=pd.read_csv('../input/carsensorsdata/Opel2.csv')
df3=pd.read_csv('../input/carsensorsdata/Peugeot1.csv')
df4=pd.read_csv('../input/carsensorsdata/Peugeot2.csv')

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df1[['AltitudeVariation', 'VehicleSpeedInstantaneous', 'VehicleSpeedAverage', 'VehicleSpeedVariance','VehicleSpeedVariation', 'LongitudinalAcceleration', 'EngineLoad', 'EngineCoolantTemperature', 'ManifoldAbsolutePressure', 'EngineRPM','MassAirFlow','IntakeAirTemperature','VerticalAcceleration','FuelConsumptionAverage']] = sc.fit_transform(df1[['AltitudeVariation', 'VehicleSpeedInstantaneous', 'VehicleSpeedAverage', 'VehicleSpeedVariance','VehicleSpeedVariation', 'LongitudinalAcceleration', 'EngineLoad', 'EngineCoolantTemperature', 'ManifoldAbsolutePressure', 'EngineRPM','MassAirFlow','IntakeAirTemperature','VerticalAcceleration','FuelConsumptionAverage']])
df2[['AltitudeVariation', 'VehicleSpeedInstantaneous', 'VehicleSpeedAverage', 'VehicleSpeedVariance','VehicleSpeedVariation', 'LongitudinalAcceleration', 'EngineLoad', 'EngineCoolantTemperature', 'ManifoldAbsolutePressure', 'EngineRPM','MassAirFlow','IntakeAirTemperature','VerticalAcceleration','FuelConsumptionAverage']] = sc.fit_transform(df2[['AltitudeVariation', 'VehicleSpeedInstantaneous', 'VehicleSpeedAverage', 'VehicleSpeedVariance','VehicleSpeedVariation', 'LongitudinalAcceleration', 'EngineLoad', 'EngineCoolantTemperature', 'ManifoldAbsolutePressure', 'EngineRPM','MassAirFlow','IntakeAirTemperature','VerticalAcceleration','FuelConsumptionAverage']])
df3[['AltitudeVariation', 'VehicleSpeedInstantaneous', 'VehicleSpeedAverage', 'VehicleSpeedVariance','VehicleSpeedVariation', 'LongitudinalAcceleration', 'EngineLoad', 'EngineCoolantTemperature', 'ManifoldAbsolutePressure', 'EngineRPM','MassAirFlow','IntakeAirTemperature','VerticalAcceleration','FuelConsumptionAverage']] = sc.fit_transform(df3[['AltitudeVariation', 'VehicleSpeedInstantaneous', 'VehicleSpeedAverage', 'VehicleSpeedVariance','VehicleSpeedVariation', 'LongitudinalAcceleration', 'EngineLoad', 'EngineCoolantTemperature', 'ManifoldAbsolutePressure', 'EngineRPM','MassAirFlow','IntakeAirTemperature','VerticalAcceleration','FuelConsumptionAverage']])
df4[['AltitudeVariation', 'VehicleSpeedInstantaneous', 'VehicleSpeedAverage', 'VehicleSpeedVariance','VehicleSpeedVariation', 'LongitudinalAcceleration', 'EngineLoad', 'EngineCoolantTemperature', 'ManifoldAbsolutePressure', 'EngineRPM','MassAirFlow','IntakeAirTemperature','VerticalAcceleration','FuelConsumptionAverage']] = sc.fit_transform(df4[['AltitudeVariation', 'VehicleSpeedInstantaneous', 'VehicleSpeedAverage', 'VehicleSpeedVariance','VehicleSpeedVariation', 'LongitudinalAcceleration', 'EngineLoad', 'EngineCoolantTemperature', 'ManifoldAbsolutePressure', 'EngineRPM','MassAirFlow','IntakeAirTemperature','VerticalAcceleration','FuelConsumptionAverage']])

df_Dataset=pd.concat([df1,df2,df3,df4],axis=0)
df_Dataset.index
df_Dataset.head(20)
df1.index
if (df_Dataset.isnull().sum().sum()!=0):
    df_Dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
df_Dataset.head()
y = df_Dataset['drivingStyle']
X = df_Dataset.drop(['drivingStyle','traffic','roadSurface'], axis=1)

y.index
X.index
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)
train_X.index
test_X.index
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lr = LinearRegression()
lr.fit(train_X, train_y)
predictions = lr.predict(test_X)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


score = lr.score(test_X, test_y)
print("Score: %.3f" % score)
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(hidden_layer_sizes=(5,), max_iter=5000)
mlp.fit(train_X, train_y.values.ravel())
predictions=mlp.predict(test_X)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(classification_report(test_y,mlp.predict(test_X)))