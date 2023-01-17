import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
import io
import tensorflow as tf

from keras import initializers
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score 
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Input, Dropout
from tensorflow.keras import layers
from sklearn.feature_selection import SelectKBest, f_regression, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
cityAttributes = pd.read_csv('../input/historical-hourly-weather-data/city_attributes.csv')
cityAttributes.iloc[:5]
humidity = pd.read_csv('../input/historical-hourly-weather-data/humidity.csv')
humidity.iloc[:5]
pressure = pd.read_csv('../input/historical-hourly-weather-data/pressure.csv')
pressure.iloc[:5]
temperature = pd.read_csv('../input/historical-hourly-weather-data/temperature.csv')
temperature.iloc[:5]
weather = pd.read_csv('../input/historical-hourly-weather-data/weather_description.csv')
BsmtQual = {'heavy shower snow': 0, 'heavy snow': 0, 'light shower snow': 1, 'light snow': 1, 'shower snow': 2, 'sleet': 2, 'light rain and snow': 2, ' light shower sleet': 2, 
            'fog': 3, 'haze':3, 'mist': 3, 'thunderstorm with heavy rain': 5, 'heavy intensity shower rain': 5, 'thunderstorm with rain': 5, 'very heavy rain': 5, 
            'ragged thunderstorm': 6, 'proximity thunderstorm': 6, 'smoke': 6, 'moderate rain': 6, 'heavy intensity rain': 6, 'thunderstorm with light rain': 6, 
            'shower rain': 7, 'thunderstorm': 7, 'proximity shower rain': 7, 'light intensity drizzle rain': 8, 'light intensity drizzle': 8, ' volcanic ash': 8, 
            'dust': 8, 'overcast clouds': 10, 'light intensity shower rain': 6, 'light rain': 7, 'broken clouds': 11, 'scattered clouds': 11,'few clouds': 11, 'sky is clear': 13}
for i in weather.columns.tolist():
    weather[i] = weather[i].map(BsmtQual)
weather
weather.iloc[:5]
windSpeed = pd.read_csv('../input/historical-hourly-weather-data/wind_speed.csv')
windSpeed.iloc[:5]
# Размерности исходных таблиц
print(cityAttributes.shape)
print(humidity.shape)
print(pressure.shape)
print(temperature.shape)
print(weather.shape)
print(windSpeed.shape)
# Добавление индексов значений к названию городов и удалить колонки "datetime"
del humidity['datetime']
humidity = humidity.rename(columns = lambda x: x.replace(x, x + 'H'))
del pressure['datetime']
pressure = pressure.rename(columns = lambda x: x.replace(x, x + 'P'))
del temperature['datetime']
temperature = temperature.rename(columns = lambda x: x.replace(x, x + 'T'))
del weather['datetime']
weather = weather.rename(columns = lambda x: x.replace(x, x + 'W'))
del windSpeed['datetime']
windSpeed = windSpeed.rename(columns = lambda x: x.replace(x, x + 'Wd'))
# Смещение значений температур
temperature = temperature - 273
temperature
# Сокращение наборов данных и объедение
allData = pd.concat([humidity, pressure, temperature, weather], axis = 1)
allData = allData.iloc[37000:42000].reset_index()
del allData['index']
allData
# Проверка на отсутствующие значения
allDataNa = allData.isnull().sum()
allDataNA = allDataNa.drop(allDataNa[allDataNa == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :allDataNa})
missing_data.iloc[0:5]
# Редактирование недостающих значений
allData = allData.fillna(method='ffill')
allData = allData.fillna(method='bfill')
allDataNa = allData.isnull().sum()
allDataNA = allDataNa.drop(allDataNa[allDataNa == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :allDataNa})
missing_data.iloc[0:5]
# Разеление данных на тренировочные и тестовые
trainData, testData = train_test_split(allData, test_size = 0.2, random_state = 42, shuffle = False)
# Переиндексация
trainData = trainData.reset_index()
del trainData['index']
testData = testData.reset_index()
del testData['index']
# Разбивка данных на параметры и цель
trainDataX = trainData.drop(['VancouverT'], axis = 1)
trainDataY = trainData['VancouverT']
testDataX = testData.drop(['VancouverT'], axis = 1)
testDataY = testData['VancouverT']
# Среднее значение и стандартное отклонение
mean = trainDataX.mean(axis=0)
std = trainDataX.std(axis=0)
# Нормализация данных
trainDataXN = (trainDataX - mean)/std
testDataXN = (testDataX - mean)/std
# Применение метода главных компонент
pcaX = PCA(n_components = 15)
pcaX.fit(trainDataXScal)
trainDataXP = pd.DataFrame(pcaX.transform(trainDataX))
testDataXP = pd.DataFrame(pcaX.transform(testDataX))
# Вывод величины дисперсии компонентов
sum((pcaX.explained_variance_ratio_[i] for i in range(0, len(pcaX.explained_variance_ratio_))))
fSelect = SelectPercentile(f_regression, percentile = 15)
fSelect.fit(trainDataXScal, trainDataY)
trainFData = pd.DataFrame(fSelect.transform(trainDataXN))
testFData = pd.DataFrame(fSelect.transform(testDataXN))
trainDataXCor = pd.concat([trainDataXP, trainFData], axis = 1)
testDataXCor = pd.concat([testDataXP, testFData], axis = 1)
# Просмотр корреляции
correlation = trainDataXCor.corr()
plt.figure(figsize = (20,20))
sns.heatmap(correlation, vmax = 1, square = True, annot = True, cmap = 'Blues')
# Класс отображения хода обучения сети
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 30 == 0: print('')
        print('.', end = '')
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.layers.advanced_activations import *
from numpy.random import seed
seed(1)
tf.compat.v1.set_random_seed(1)
# Создание функции для кросс валидации
def crossValdation(modelTF, dataSet, arg, printD):
    trainDataX = trainData.drop([arg], axis = 1)
    trainDataY = trainData[arg]
    epohs = 200
    batch = 100
    early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5)
    histMean = pd.DataFrame({'loss':[], 'mse':[], 'mae':[], 'val_loss':[], 'val_mse':[], 'val_mae':[], 'epoch':[]})
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 1)
    for trainIndex, validIndex in skf.split(trainDataX, trainDataY.astype(int)):
        trainX = trainDataX.iloc[trainIndex]
        trainY = trainDataY.iloc[trainIndex]
        validX = trainDataX.iloc[validIndex]
        validY = trainDataY.iloc[validIndex]
        modelG = modelTF(trainX.shape[1])
        history = modelG.fit(trainX, trainY, validation_data = (validX, validY), batch_size = batch, epochs = epohs, verbose = 0, callbacks = [early_stop, printD])
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        histMean = histMean.append(hist.iloc[[-1]], ignore_index = True)
    
    del histMean['epoch']    
    tir = pd.DataFrame({'loss':['---'], 'mse':['---'], 'mae':['---'], 'val_loss':['---'], 'val_mse':['---'], 'val_mae':['---']})
    histM = pd.DataFrame(histMean.mean()).T
    histMean = histMean.append(tir)
    histMean = histMean.append(histM)
    return histMean, modelG, hist
def createModelG(inputShape):
    model = Sequential()
    model.add(Dense(4096, input_dim = inputShape, 
                    kernel_initializer = initializers.glorot_uniform(seed = 1), 
                    kernel_regularizer = keras.regularizers.l2(0.01), activation = "relu")) 
    model.add(Dense(2048, 
                    kernel_initializer = initializers.glorot_uniform(seed = 1), activation = "relu"))
    model.add(Dense(2048, 
                    kernel_initializer = initializers.glorot_uniform(seed = 1), activation = "relu"))
    model.add(Dense(1024, 
                    kernel_initializer = initializers.glorot_uniform(seed = 1), activation = "relu"))
    model.add(Dense(1024, 
                    kernel_initializer = initializers.glorot_uniform(seed = 1), activation = "relu"))
    model.add(layers.Dropout(0.05))
    model.add(Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.000001)
    model.compile(loss = 'mse', optimizer = optimizer, metrics = ["mse", "mae"])
    return model
startModelTest = crossValdation(createModelG, trainDataXCor, 'VancouverT', PrintDot())
modelTest = startModelTest[1]
hist = startModelTest[2]
startModelTest[0]
pred = modelTest.predict(testDataX).flatten()
difference = pred - testDataY.values
print(sum(abs(difference))/len(difference))
def plot_history():
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label = 'Val Error')
    plt.legend()
    plt.ylim([0,20])

plot_history()
for i in range(0, 1000, 50):
    print("Предсказанная температура:", round(pred[i], 3), ", правильная температура:", round(testDataY.values[i], 3))
# Прогноз на общих данных ненормированных
reg = LinearRegression().fit(trainDataXN, trainDataY)
predL = reg.predict(testDataXN)
for i in range(0, 1000, 100):
    print("Предсказанная температура:", round(predL[i], 3), ", правильная температура:", round(testDataY.values[i], 3))
# Оценка погрешности (в градусах)
difference = predL - testDataY.values
print(sum(abs(difference))/len(difference))
# Прогноз на общих данных масштабированных
reg = LinearRegression().fit(trainDataXScal, trainDataY)
predS = reg.predict(testDataXScal)
for i in range(0, 1000, 100):
    print("Предсказанная температура:", round(predS[i], 3), ", правильная температура:", round(testDataY.values[i], 3))
# Оценка погрешности (в градусах)
differenceS = predS - testDataY.values
print(sum(abs(differenceS))/len(differenceS))
# Прогноз на данных отобранных F регрессией
reg = LinearRegression().fit(trainFData, trainDataY)
predF = reg.predict(testFData)
for i in range(0, 1000, 100):
    print("Предсказанная температура:", round(predF[i], 3), ", правильная температура:", round(testDataY.values[i], 3))
# Оценка погрешности (в градусах)
differenceF = predF - testDataY.values
print(sum(abs(differenceF))/len(differenceF))
# Прогноз на данных отобранных PCA
reg = LinearRegression().fit(trainDataXScalPCA, trainDataY)
predP = reg.predict(testDataXScalPCA)
for i in range(0, 1000, 100):
    print("Предсказанная температура:", round(predP[i], 3), ", правильная температура:", round(testDataY.values[i], 3))
# Оценка погрешности (в градусах)
differenceP = predP - testDataY.values
print(sum(abs(differenceP))/len(differenceP))
# Прогноз на данных отобранных PCA с F
reg = LinearRegression().fit(correlationDataX, trainDataY)
predPF = reg.predict(correlationDataY)
for i in range(0, 1000, 100):
    print("Предсказанная температура:", round(predPF[i], 3), ", правильная температура:", round(testDataY.values[i], 3))
# Оценка погрешности (в градусах)
differencePF = predPF - testDataY.values
print(sum(abs(differencePF))/len(differencePF))
rfr = RandomForestRegressor(n_estimators = 5, criterion = 'mae', max_depth = 10).fit(trainDataX, trainDataY)
predRFR = rfr.predict(testDataX)
for i in range(0, 1000, 100):
    print("Предсказанная температура:", round(predRFR[i], 3), ", правильная температура:", round(testDataY.values[i], 3))
# Оценка погрешности (в градусах)
differenceRFR = predRFR - testDataY.values
print(sum(abs(differenceRFR))/len(differencePF))





