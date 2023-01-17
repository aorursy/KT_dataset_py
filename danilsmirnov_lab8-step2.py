 #import pandas

import pandas as pd

import os

#print plot

import matplotlib.pyplot as plt

import seaborn as sns

 
#ЗАДАНИЕ 1 ПОДКЛЮЧИЛ ДАТАСЕТ

print(os.listdir('../input/nevije'))

#filename 

my_filepath = "../input/nevije/Nedvijimost.csv"

#read this file

df = pd.read_csv(my_filepath)

 

nRow, nCol = df.shape

print(f'В таблице {nRow} строк и {nCol} колонок')



#ЗАДАНИЕ 2 ПРОАНАЛИЗИРОВАЛ ДАННЫЕ

df.describe()
# Set up code checking

 

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex2 import *

from sklearn.model_selection import train_test_split

print("Setup Complete")
def setAvgByColumn(column):

    avg = 0

    length = 0

    for index, row in df.iterrows():

        if not row[column].isnull().any():  

            avg+=row[cols_with_missing]

            length+=1

    return avg/length

 
#ЗАДАНИЕ 3 СМОТРИМ ПОТЕРЯННЫЕ ДАННЫЕ

missing_val_count_by_column = (df.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])

cols_with_missing = [col for col in df.columns

                     if df[col].isnull().any()]

#ЗАМЕНЯЕМ ПОТЕРЯННЫЕ ДАННЫЕ (РЕШИЛ НАПИСАТЬ САМ, ФУНКЦИЯ ПОЛУЧИЛСЬ НЕ ОЧЕНЬ, НО ОНА РАБОТАЕТ)

for index, row in df.iterrows():

    if row[cols_with_missing].isnull().any():

        print('Replace empty value for',row[cols_with_missing])

        df.at[index,cols_with_missing] = setAvgByColumn(cols_with_missing)

        print('Replaced value',row[cols_with_missing])

missing_val_count_by_column = (df.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])



 
#ЗАДАНИЕ 4. ОПРЕДЕЛЯЕМ КАТЕГОРИАЛЬНЫЕ СТОЛБЦЫ

from sklearn.preprocessing import OneHotEncoder



object_cols = [col for col in df.columns if df[col].dtype == "object"]

print(object_cols)



low_cardinality_cols = [col for col in object_cols if df[col].nunique() < 10]

high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))



OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(df[low_cardinality_cols]))

OH_cols_valid = pd.DataFrame(OH_encoder.transform(df[low_cardinality_cols]))

 

OH_cols_train.index = df.index

OH_cols_valid.index = df.index

 

num_X_train = df.drop(object_cols, axis=1)

num_X_valid = df.drop(object_cols, axis=1)



OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
#ЗАДАНИЕ 5  Разделить данные на обучающее и тестовое множества и построить

# несколько деревьев решений, определить наилучшее количество

# узлов в дереве по ошибке обучения модели

X_full = df

import random

randomNum = random.randint(100, 150)

#тестовых данных нет берем просто рандомные строки из текущего файла

X_test_full = df.iloc[randomNum:randomNum*2, :]

#print(X_test_full.columns)

print(df.columns)

y = X_full.iloc[:,10:11] 

 

features = object_cols

features = ['Количество комнат','Общая площадь (м2)', 'Жилая площадь (м2)']

X = X_full[features].copy()

X_test = X_test_full[features].copy()



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)





from sklearn.ensemble import RandomForestRegressor



# Define the models

model_1 = RandomForestRegressor(n_estimators=50, random_state=0)

model_2 = RandomForestRegressor(n_estimators=100, random_state=0)

model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)

model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)

model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)



models = [model_1, model_2, model_3, model_4, model_5]
from sklearn.metrics import mean_absolute_error



# сравниваем омдели по ошибке

def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):

    model.fit(X_t, y_t)

    preds = model.predict(X_v)

    return mean_absolute_error(y_v, preds)



for i in range(0, len(models)):

    mae = score_model(models[i])

    print("Model %d MAE: %d" % (i+1, mae))

#присваиваем лучшую модель

best_model = model_3

best_model