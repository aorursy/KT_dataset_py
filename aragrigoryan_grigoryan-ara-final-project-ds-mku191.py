import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))

#Во-первых, мы должны импортировать данные

data = pd.read_csv("../input/did-it-rain-in-seattle-19482017/seattleWeather_1948-2017.csv")
data.head()

#Показывает нам информацию о первых 5 погодных условиях.
data.tail()

# Показывает нам информацию о последних 5 погодных условиях
data.columns

# См. Столбцы данных
data.describe()

data['DATE'] = pd.to_datetime(data['DATE'])
data['YEAR'] = data['DATE'].dt.year

data['MONTH'] = data['DATE'].dt.month

data['DAY'] = data['DATE'].dt.day
data.columns 
data.info()
data=data.dropna() #Мы очищаем все пропущенные значения

data.info()
data.RAIN = [1 if each == True  else 0 for each in data.RAIN.values]
data.info()

# Мы можем легко увидеть, что значение RAIN переобразовалось в чилсо
y = data.RAIN.values

#Ось y определяется значениями дождя, потому что мы хотим узнать, какая погода? 

#Таким образом,понять: погода дождливая или нет?
x_data = data.drop(["RAIN"],axis=1)

#x_data - это все функции, кроме RAIN в таблице.

x_data
x_data = data.drop(["DATE"],axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data)).values

x
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 42)

#test_size=0.2 means %20 test datas, %80 train datas
x_train.head()
x_train = data.drop(["RAIN", "DATE"],axis=1)
x_train.head()
x_train.tail()
y_test
from sklearn import linear_model

logistic_reg = linear_model.LogisticRegression(random_state=50,max_iter=210)

#max_iter является необязательным параметром. Вы можете написать 10 или 3000, если хотите.
print("Test accuracy {}".format(logistic_reg.fit(x_train,y_train).score(x_test,y_test)))

print("Train accuracy {}".format(logistic_reg.fit(x_train, y_train).score(x_train, y_train)))