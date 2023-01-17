# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import matplotlib.pyplot as plt
#import seaborn as sns 
#%matplotlib inline
# Загружаем специальный удобный инструмент для разделения датасета:
#from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under 
# the input directory
import os
for dirname, _, filenames in os.walk('data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.
# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:
!pip freeze > requirements.txt
import pandas as pd

# ТЕПЛОВАЯ КАРТА КОРРЕЛЯЦИОННОГО АНАЛИЗА МЕЖДУ ВЕЛИЧИНАМИ.
# степень связи (`коэффициент корреляции`) может меняться `от -1 до +1`: чем ближе абсолютное значение 
# коэффициента к единице, тем сильнее связь между признаками.
import seaborn as sns
#import warnings; warnings.simplefilter('ignore')
import matplotlib.pyplot as plt

# СПЕЦИАЛЬНЫЙ ИНСТРУМЕНТ ДЛЯ РАЗБИВКИ ДАТАФРЕЙМА НА ЧАСТИ, необходимые для обучения и тестирования модели
from sklearn.model_selection import train_test_split

# БИБЛИОТЕКИ, НЕОБХОДИМЫЕ ДЛЯ СОЗДАНИЯ, ОБУЧЕНИЯ И ТЕСТИРОВАНИЯ МОДЕЛИ
from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
from sklearn import metrics # инструменты для оценки точности модели

import numpy as np

import math as math

from collections import Counter

from datetime import datetime, date

import json
from pprint import pprint
RANDOM_SEED = 42
def TrainModel(displaydf=1, displayIMP=1, displayBARH=0, BARHsize=4, delete=[], unite=[[]]):
    global df_all
    
# УБИРАЕМ ВСЕ ЛИШНИЕ СТОЛБЦЫ (нечисловые данные) ИЗ ДАТАФРЕЙМА
    #['Restaurant_id', 'City', 'Cuisine Style', 'Price Range', 'Reviews', 'URL_TA', 'ID_TA']
    DropCols = [column for column in df_all.columns if df_all[column].dtypes=='object']
    DropCols = DropCols+delete # если есть еще столбцы для удаления из датафрейма
    df = df_all.drop(DropCols, axis=1) # df - столбцы с числовыми данными
    df = df.fillna(0) # заполняем пропуски 0 (нулями)
    if displaydf==1: display(df.drop(['sample'], axis=1).sample(1))
    
# РАЗБИВАЕМ ДАТАФРЕЙМ НА ЧАСТИ, НЕОБХОДИМЫЕ ДЛЯ ОБУЧЕНИЯ И ТЕСТИРОВАНИЯ МОДЕЛИ:
    train_data = df.query('sample == 1').drop(['sample'], axis=1)
    test_data = df.query('sample == 0').drop(['sample'], axis=1)
    X = train_data.drop(['Rating'], axis=1) # Х - данные с информацией о ресторанах
    y = train_data['Rating'] # у - целевая переменная (рейтинги ресторанов)
    # нужный инструмент для разбивки уже загружен:
    #from sklearn.model_selection import train_test_split
    # Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.
    # Для тестирования мы будем использовать 25% от исходного датасета.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED)
    # проверяем
    #print(test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape)
    
# СОЗДАЁМ, ОБУЧАЕМ И ТЕСТИРУЕМ МОДЕЛЬ:
    # необходимые библиотеки уже импортированы:
    #from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
    #from sklearn import metrics # инструменты для оценки точности модели
    # создаём модель
    regr = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED) # , verbose=1, n_jobs=-1
    # обучаем модель на тестовом наборе данных
    regr.fit(X_train, y_train)
    # используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
    # предсказанные значения записываем в переменную y_pred
    y_pred = np.round(regr.predict(X_test), 1)

# MAE - СРАВНИВАЕМ ПРЕДСКАЗАННЫЕ ЗНАЧЕНИЯ (y_pred) С РЕАЛЬНЫМИ (y_test):
    # насколько в среднем отличаются предсказанные значения (y_pred) и реальные (y_test)
    # Метрика - Mean Absolute Error (MAE) - показывает среднее отклонение предсказанных значений от фактических
    MAE = metrics.mean_absolute_error(y_test, y_pred) # = abs(y_pred - y_test)
    print('\nMAE:', round(MAE, 2), end='\t')

# ACCURACY - ОПРЕДЕЛЯЕМ ПОКАЗАТЕЛИ ЭФФЕКТИВНОСТИ:
    errors = abs(y_pred - y_test) # считаем абсолютые ошибки
    mape = 100 * (errors / y_test) # считаем средний абсолютный процент ошибки (MAPE)
    accuracy = 100 - np.mean(mape) # считаем и выводим точность
    print('Точность:', round(accuracy, 2), '%')
    
# ОКРУГЛЯЕМ ПРЕДСКАЗАННЫЕ ЗНАЧЕНИЯ В СООТВЕТСТВИИ С ШАГОМ ЗНАЧЕНИЙ РЕЙТИНГА
# (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
    for i in range(0,len(y_pred)): y_pred[i]=(round(y_pred[i]*2,0)/2)
    MAE = metrics.mean_absolute_error(y_test, y_pred) # = abs(y_pred - y_test)
    print('    ', round(MAE, 2), end='\t')
    errors = abs(y_pred - y_test) # считаем абсолютые ошибки
    mape = 100 * (errors / y_test) # считаем средний абсолютный процент ошибки (MAPE)
    accuracy = 100 - np.mean(mape) # считаем и выводим точность
    print('         ', round(accuracy, 2), '%\t(после округления предсказанных значений)')    

# ЗНАЧИМОСТЬ ПЕРЕМЕННЫХ:
    # относительные значения переменных показывают, насколько включение определенной переменной 
    # может улучшить прогноз - в RandomForestRegressor есть возможность вывести самые важные 
    # признаки для модели
    importances = list(regr.feature_importances_)
    # cписок кортежей из переменной и ее относительного значения
    X_importances = [(feature, importance) for feature, importance in zip(list(X.columns), importances)]
    
    if displayIMP==1:
        # проверка, задан ли необязательный аргумент unite - список названий переменных, относительное 
        # значение которых нужно воспринимать суммарно (например, при прямом кодировании городов получим N 
        # столбцов в датафрейме и, следовательно, столько же переменных в X_importances каждая со своим 
        # относительным значением)
        if len(unite[0])>0:
            for item in unite:
                UnitedImportance=0
                for i, j in X_importances: # перебор переменных модели и их относительных значений
                    if item.count(i)>0: # нужная переменная
                        UnitedImportance+=j # считаем суммарную значимость
                X_importances.append((item[0]+' SUM', UnitedImportance)) # добавить к X_importances
        # cортировка переменных по убыванию относительной значимости
        # возьмем за пороговую значимость 0.1 (остальные переменные "слабые") - тогда минимальное значение, 
        # округляемое до 0.1, равно 0.05 вывод на экран переменных с относительной значимостью больше 0.05
        LowImportanceVariables=0
        print('-------------------------------------------------')
        for i, j in X_importances:
            if j>=0.05: # значения, большие или равные минимальной пороговой значимости
                if len(unite[0])==0 or (len(unite[0])>0 and str(unite).find(i)==-1): # не в unite
                    print('Переменная: {:20} Значимость: {}'.format(i, round(j, 2)))
                elif (str(unite).find(i) and str(i).find('Cols')==-1): # переменная в unite, не название
                    LowImportanceVariables+=1
            else: # меньше минимальной пороговой значимости
                if str(i).find('Cols')==-1: # переменная не название в unite (потому что такого столбца нет)
                    LowImportanceVariables+=1    
        if LowImportanceVariables>0:
            print()
            temp=[]
            for i, j in X_importances:
                if j<0.05:
                    if len(unite[0])>0: # если задан аргумент unite
                        if str(unite).find(i)==-1: # если i не из unite
                            print('Переменная: {:20} Значимость: {}'.format(i, round(j, 2)))
                        else: # если i из unite
                            for item in unite:
                                if i==item[0]+' SUM': # если это название из unite
                                    print('Переменная: {:20} Значимость: {}'.format(i, round(j, 2)))
                                elif item.count(i)>0: # если это переменная из unite, но не название
                                    if temp.count(item[0]+' SUM')==0: temp.append(item[0]+' SUM')
                    else: # unite не задан
                        print('Переменная: {:20} Значимость: {}'.format(i, round(j, 2)))
            if len(temp)>0:
                print('Остальные переменные - часть {}'.format(temp))
                print('Их индивидуальная значимость не учитывается')
        print('-------------------------------------------------')
    
    if displayBARH==1:
        plt.rcParams['figure.figsize'] = (BARHsize,BARHsize)
        if len(unite[0])>0:
            for item in unite:
                for i, j in X_importances:
                    if i==(item[0]+' SUM'): X_importances.remove((i, j))
        pd.Series([x[1] for x in X_importances], [x[0] for x in X_importances]).nlargest(15).plot(kind='barh')
def Heatmap(size=5, delete=[], drawmap=1, porog=0.5, displayCORR=1):
    global df_all
    
    DropCols = [column for column in df_all.columns if df_all[column].dtypes=='object']
    DropCols = DropCols+delete
    DropCols = DropCols+['sample']
    # Коэффициенты корреляции между количественными признаками, содержащимися в датафрейме - метод corr().
    correlation = df_all.query('sample == 1').drop(DropCols, axis=1).corr()
    
    if drawmap==1:
        # Тепловая карта значений коэффициентов корреляции
        plt.rcParams['figure.figsize'] = (size,size) # размер области отображения
        sns.heatmap(correlation, cmap='Accent', center=.2, 
                    robust=False, fmt='.1g', linewidths=0.01, linecolor='white', square=True)
    
    # Наибольшие коэффициенты корреляции
    # матрица симметрична, поэтому нужен верхний треугольник матрицы без диагонали
    corrpairs = (correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(np.bool))
                 .stack().sort_values(ascending=False))
    
    if displayCORR==1:
        for i,j in corrpairs.items():
            if abs(round(j, 1))>=porog and i.count('Rating')==0:
                if j<0: space=''
                else: space=' '
                print('[{}{} ] {}'.format(space, round(j, 1), 
                                          str(i).replace("('", "").replace("')", "").replace("', '", " - ")))
def Pairplot(delete=[]):
    global df_all
        
    DropCols = [column for column in df_all.columns if df_all[column].dtypes=='object']
    DropCols = DropCols+delete
    DropCols = DropCols+['sample']
    
    sns.pairplot(df_all.query('sample == 1').drop(DropCols, axis=1))
DIR = '/kaggle/input/sf-dst-restaurant-rating/'
filename_Train = 'main_task.csv'
filename_Test = 'kaggle_task.csv'
filename_Submission = 'sample_submission.csv'
sample_submission = pd.read_csv(DIR+filename_Submission)
df_train = pd.read_csv(DIR+filename_Train)
df_train.sample(2)
df_test = pd.read_csv(DIR+filename_Test)
df_test.sample(2)
# Помечаем датафреймы в едином датафрейме с помощью столбца 'sample'
df_train['sample'] = 1 # тренировочный датафрейм
df_test['sample'] = 0 # тестовый датафрейм
# создаем недостающий столбец 'Rating' (должны будем его предсказать) в тестовом датафрейме df_test 
# и заполняем его нулями
df_test['Rating'] = 0
print(sorted(list(df_test.columns))==sorted(list(df_train.columns)))
# объединяем оба датафрейма в один для обучения модели
df_all = df_test.append(df_train, sort=False).reset_index(drop=True)
TrainModel()
display(df_all[df_all['Restaurant_id']=='id_1717'][['Ranking', 'Cuisine Style']].nunique())
display(df_all[df_all['Ranking']==1719][['Restaurant_id', 'Cuisine Style']].nunique())
# Значения = инфо из столбца 'Restaurant_id' без 'id_'
df_all['Restaurant ID'] = df_all['Restaurant_id'].apply(lambda x: int(x[3::]))
print(df_all['ID_TA'].count(), ' | ', df_all['ID_TA'].nunique())
# 'URL_TA' > 'URL_TA gID'
df_all[['URL_TA', 'ID_TA']].head(1)
# Есть 2 цифровых значения: g1068497 и d12160475;
# значения типа d12160475 полностью совпадает со значением в 'ID_TA'
# Извлечем цифровые данные из значений типа g1068497
df_all['URL_TA gID'] = df_all['URL_TA'].apply(lambda x: int((str(x).split('-'))[1][1::]))
df_all.groupby('City')['URL_TA gID'].nunique().sort_values().plot(kind='barh', figsize=(5, 6))
# цены даны в 3 категориях от самых дешевых ($) до самых дорогих ($$$$) - кодировка цен числами 
# 0 (NaN), 1 ($), 2 ($$ - $$$) и 3 ($$$$) логически сохраняет эту иерархию
PriceCategories = { '$': { 'code': 1 }, '$$ - $$$': { 'code': 2 }, '$$$$': { 'code': 3 } }
# столбцу 'Price Category' присваивается значение ключа из словаря PriceCategories если ключ совпадает со
# значением в 'Price Range', или 0 - если 'Price Range' NaN
df_all['Price category'] = df_all['Price Range'].apply(lambda x: PriceCategories[x]['code'] 
                                                       if PriceCategories.get(x) else x)
df_all.isna().sum()
TrainModel()
Heatmap()
TrainModel(displaydf=0, delete=['Restaurant ID'])
df_all.drop('Restaurant ID', axis=1, inplace=True)
# 'Number of Reviews'
df_all['Number of Reviews'].isna().sum()
display(round(df_all[df_all['Number of Reviews'].isna()]['Cuisine Style'].
              value_counts(dropna=False, normalize=True).head(1)*100,0))

display(round(df_all[df_all['Number of Reviews'].isna()]['Price Range'].
              value_counts(dropna=False, normalize=True).head(1)*100,0))

# у 'Reviews' 2 значения NaN - заменим сначала все NaN значения 'Reviews' на '[[], []]'
df_all.Reviews = df_all.Reviews.apply(lambda x: '[[], []]' if x!=x else x)
display(round(df_all[df_all['Number of Reviews'].isna()]['Reviews'].
              value_counts(dropna=False, normalize=True).head(1)*100,0))
df_all.loc[(df_all['Cuisine Style'].isna()) & (df_all['Price Range'].isna()) & 
           (df_all['Reviews']=='[[], []]')]['Number of Reviews'].value_counts(dropna=False).head(1)
fig, axes = plt.subplots(1, 2, figsize=[20,6])
fig.autofmt_xdate(rotation=85)

axes[0].plot(df_all.groupby('City')['Number of Reviews'].agg(lambda x: pd.Series.value_counts(x).idxmax()), 
             color='blue', linestyle='--', label="Наиболее частое 'Number of Reviews'")
axes[0].plot(round(df_all.groupby('City')['Number of Reviews'].mean(), 0), color='red', linestyle='--', 
             label="Среднее 'Number of Reviews'")
axes[0].plot(df_all.groupby('City')['Number of Reviews'].min(), color='green', linestyle='--', 
             label="Минимум 'Number of Reviews'")
axes[0].legend(loc=1)

axes[1].plot(df_all.groupby('City')['Number of Reviews'].agg(lambda x: pd.Series.value_counts(x).idxmax()), 
             color='blue', linestyle='--', label="Наиболее частое 'Number of Reviews'")
axes[1].plot(df_all.groupby('City')['Number of Reviews'].min(), color='green', linestyle='--', 
             label="Минимум 'Number of Reviews'")
axes[1].plot(round(df_all[df_all['Price Range'].isna()].groupby('City')['Number of Reviews'].mean(),0), 
             color='violet', linestyle=':', label="Среднее 'Number of Reviews' при NaN 'Price Range'")
axes[1].plot(df_all[df_all['Price Range'].isna()].groupby('City')
             ['Number of Reviews'].agg(lambda x: pd.Series.value_counts(x).idxmax()), 
             color='violet', linestyle='-.', label="Наиболее частое 'Number of Reviews' при NaN 'Price Range'")
axes[1].plot(round(df_all.loc[(df_all['Cuisine Style'].isna()) & (df_all['Price Range'].isna()) & 
                              (df_all['Reviews']=='[[], []]')].groupby('City')['Number of Reviews'].mean(),0), 
             color='purple', linestyle=':', 
             label="Среднее 'Number of Reviews' при NaN 'Price Range', 'Cuisine Style' & '[[], []]' 'Reviews'")
axes[1].legend(loc=1)
print()
# среднее значение 'Number of Reviews' по городу
display(round(df_all.groupby('City')['Number of Reviews'].mean().sort_values(ascending=False).head(5),0))
# минимальное значение 'Number of Reviews' по городу
display(round(df_all.groupby('City')['Number of Reviews'].min().sort_values(ascending=False).head(5),0))
fig, axes = plt.subplots(1, 3, figsize=[20,6])
fig.autofmt_xdate(rotation=85)

axes[0].plot(df_all.groupby('City')['Number of Reviews'].max(), color='black', linestyle='--',label="Максимум")
axes[0].plot(df_all.groupby('City')['Number of Reviews'].agg(lambda x: pd.Series.value_counts(x).idxmax()), 
             color='blue', linestyle=':', label="Наиболее частое")
axes[0].plot(round(df_all.groupby('City')['Number of Reviews'].mean(), 0), color='red', linestyle=':', 
             label="Среднее")
axes[0].plot(df_all.groupby('City')['Number of Reviews'].min(), color='green', linestyle=':', label="Минимум")
axes[0].legend(loc=1)
axes[0].set_title(
    "Максимальные, средние, наиболее частые\nи минимальные значения\n'Number of Reviews' по городам")

axes[1].plot(df_all.groupby('City')['Number of Reviews'].agg(lambda x: pd.Series.value_counts(x).idxmax()), 
             color='blue', linestyle=':', label="Наиболее частое")
axes[1].plot(round(df_all.groupby('City')['Number of Reviews'].mean(), 0), color='red', linestyle=':', 
             label="Среднее")
axes[1].plot(df_all.groupby('City')['Number of Reviews'].min(), color='green', linestyle=':', label="Минимум")
axes[1].plot([round(np.mean(y), 0) for y in 
              zip(df_all.groupby('City')['Number of Reviews'].agg(
                  lambda x: pd.Series.value_counts(x).idxmax()),
                  df_all.groupby('City')['Number of Reviews'].min(), 
                  round(df_all.groupby('City')['Number of Reviews'].mean(), 0))], color='purple', 
             linestyle='-', label="Среднее между Средним, Минимальным и Наиболее частым")
axes[1].legend(loc=1)
axes[1].set_title(
    "Cредние, наиболее частые, минимальные\nи средние между ними значения\n'Number of Reviews' по городам")

axes[2].plot(df_all.groupby('City')['Number of Reviews'].agg(lambda x: pd.Series.value_counts(x).idxmax()), 
             color='blue', linestyle=':', label="Наиболее частое")
axes[2].plot(df_all.groupby('City')['Number of Reviews'].min(), color='green', linestyle=':', label="Минимум")
axes[2].legend(loc=1)
axes[2].set_title(
    "Наиболее частые и минимальные между ними\nзначения 'Number of Reviews' по городам")
print()
NaN = df_all.groupby('City')['Number of Reviews'].agg(lambda x: pd.Series.value_counts(x).idxmax())
for city in df_all.City.unique():
    df_all['Number of Reviews']=df_all.apply(lambda x: NaN[city] if x['City']==city 
                                             and x['Number of Reviews']!=x['Number of Reviews'] 
                                             else x['Number of Reviews'], axis=1)
# цены даны в 3 категориях от самых дешевых ($) до самых дорогих ($$$$) - кодировка цен числами 
# 0 (NaN), 1 ($), 2 ($$ - $$$) и 3 ($$$$) логически сохраняет эту иерархию
PriceCategories = { '$': { 'code': 1 }, '$$ - $$$': { 'code': 2 }, '$$$$': { 'code': 3 } }
# столбцу 'Price Category' присваивается значение ключа из словаря PriceCategories если ключ совпадает со
# значением в 'Price Range', или 0 - если 'Price Range' NaN
df_all['Price category'] = df_all['Price Range'].apply(lambda x: PriceCategories[x]['code'] 
                                                       if PriceCategories.get(x) else x)
df_all['Price category'].isna().sum()
fig = plt.figure()
axes = fig.add_axes([0, 0, 1.5, 0.75])
fig.autofmt_xdate(rotation=80)

axes.plot(df_all.groupby('City')['Price category'].max(), color='black', linestyle='--', 
          label="Максимальное")
axes.plot(df_all.groupby('City')['Price category'].agg(lambda x: pd.Series.value_counts(x).idxmax()), 
          color='blue', linestyle=':', label="Наиболее частое")
axes.plot(df_all.groupby('City')['Price category'].mean(), color='red', linestyle=':', 
          label="Среднее")
axes.plot(round(df_all.groupby('City')['Price category'].mean(),0), color='pink', linestyle='-.', 
          label="Среднее округленное")
axes.set_title("'Price category' ПО ГОРОДАМ")
axes.legend(loc=1)
NaN = df_all.groupby('City')['Price category'].agg(lambda x: pd.Series.value_counts(x).idxmax())
for city in df_all.City.unique():
    df_all['Price category']=df_all.apply(lambda x: NaN[city] if x['City']==city 
                                          and x['Price category']!=x['Price category'] 
                                          else x['Price category'], axis=1)
TrainModel()
Heatmap()
Pairplot()
# СЛОВАРЬ СТРАН (https://www.worldometers.info/world-population/portugal-population/)
CountriesInfo = {
    'Austria': {'cap': 'Vienna', 'pop': 8984912, 'cuis': 'Austrian', 'reg': ['European', 'Central European']},
    'Belgium': {'cap': 'Brussels', 'pop': 11568565, 'cuis': 'Belgian', 'reg': ['European']},
    'Czech Republic': {'cap': 'Prague', 'pop': 10700709, 'cuis': 'Czech', 
                       'reg': ['European', 'Central European']},
    'Denmark': {'cap': 'Copenhagen', 'pop': 5783694, 'cuis': 'Danish', 'reg': ['European', 'Scandinavian']},
    'England': {'cap': 'London', 'pop': 56000000, 'cuis': 'British', 'reg': ['European']},
    'Finland': { 'cap': 'Helsinki', 'pop': 5537137, 'cuis': 'Finnish', 'reg': ['European']},
    'France': {'cap': 'Paris', 'pop': 65213347, 'cuis': 'French', 'reg': ['European', 'Mediterranean']},
    'Germany': {'cap': 'Berlin', 'pop': 83672230, 'cuis': 'German', 'reg': ['European', 'Central European']},
    'Greece': { 'cap': 'Athens', 'pop': 10444100, 'cuis': 'Greek', 'reg': ['European', 'Mediterranean']},
    'Hungary': {'cap': 'Budapest', 'pop': 9670516, 'cuis': 'Hungarian', 
                'reg': ['European', 'Central European']},
    'Ireland': {'cap': 'Dublin', 'pop': 4914590, 'cuis': 'Irish', 'reg': ['European']},
    'Italy': {'cap': 'Rome', 'pop': 60498712, 'cuis': 'Italian', 'reg': ['European', 'Mediterranean']},
    'Luxembourg': { 'cap': 'Luxembourg', 'pop': 621672, 'cuis': 'Luxembourgish', 'reg': ['European']},
    'Netherlands': {'cap': 'Amsterdam', 'pop': 17119079, 'cuis': 'Dutch', 'reg': ['European']},
    'Norway': {'cap': 'Oslo', 'pop': 5403478, 'cuis': 'Norwegian', 'reg': ['European', 'Scandinavian']},
    'Poland': {'cap': 'Warsaw', 'pop': 37863816, 'cuis': 'Polish', 'reg': ['European', 'Central European']},
    'Portugal': {'cap': 'Lisbon', 'pop': 10209025, 'cuis': 'Portuguese', 'reg': ['European']},
    'Scotland': { 'cap': 'Edinburgh', 'pop': 5400000, 'cuis': 'Scottish', 'reg': ['European']},
    'Slovakia': {'cap': 'Bratislava', 'pop': 5458543, 'cuis': 'Slovakian', 
                 'reg': ['European', 'Eastern European']},
    'Slovenia': { 'cap': 'Ljubljana', 'pop': 2078819, 'cuis': 'Slovenian', 
                 'reg': ['European', 'Central European', 'Mediterranean']},
    'Spain': {'cap': 'Madrid', 'pop': 46747250, 'cuis': 'Spanish', 'reg': ['European', 'Mediterranean']},
    'Sweden': {'cap': 'Stockholm', 'pop': 10072922, 'cuis': 'Swedish', 'reg': ['European', 'Scandinavian']},
    'Switzerland': { 'cap': 'Bern', 'pop': 8628114, 'cuis': 'Swiss', 'reg': ['European', 'Central European']},
}
# СЛОВАРЬ ГОРОДОВ
CitiesInfo = {'Amsterdam': {'country': 'Netherlands', 'pop': 1140000},
              'Athens': {'country': 'Greece', 'pop': 3154000 }, 
              'Barcelona': {'country': 'Spain', 'pop': 5541000}, 
              'Berlin': {'country': 'Germany', 'pop': 3557000}, 
              'Bratislava': {'country': 'Slovakia', 'pop': 434926}, 
              'Brussels': {'country': 'Belgium', 'pop': 2065284}, 
              'Budapest': {'country': 'Hungary', 'pop': 1764000}, 
              'Copenhagen': {'country': 'Denmark', 'pop': 1334000}, 
              'Dublin': {'country': 'Ireland', 'pop': 1215000}, 
              'Edinburgh': {'country': 'Scotland', 'pop': 531000}, 
              'Geneva': {'country': 'Switzerland', 'pop': 201818}, 
              'Hamburg': {'country': 'Germany', 'pop': 1789954 }, 
              'Helsinki': {'country': 'Finland', 'pop': 1292000}, 
              'Krakow': {'country': 'Poland', 'pop': 760000}, 
              'Lisbon': {'country': 'Portugal', 'pop': 2942000}, 
              'Ljubljana': {'country': 'Slovenia', 'pop': 292988}, 
              'London': {'country': 'England', 'pop': 8787892}, 
              'Luxembourg': {'country': 'Luxembourg', 'pop': 613894}, 
              'Lyon': {'country': 'France', 'pop': 1705000}, 
              'Madrid': {'country': 'Spain', 'pop': 6559000}, 
              'Milan': {'country': 'Italy', 'pop': 3136000}, 
              'Munich': {'country': 'Germany', 'pop': 1521000}, 
              'Oporto': {'country': 'Portugal', 'pop': 1312947}, 
              'Oslo': {'country': 'Norway', 'pop': 1027000}, 
              'Paris': {'country': 'France', 'pop': 2141000}, 
              'Prague': {'country': 'Czech Republic', 'pop': 1319000}, 
              'Rome': {'country': 'Italy', 'pop': 4234000}, 
              'Stockholm': {'country': 'Sweden', 'pop': 1608000}, 
              'Vienna': {'country': 'Austria', 'pop': 1915000}, 
              'Warsaw': {'country': 'Poland', 'pop': 1776000}, 
              'Zurich': {'country': 'Switzerland', 'pop': 1383000}}
# df_all['City'].value_counts() = ('London': 5757, 'Paris': 4897, ...) - 'Город': число встречаемоти
# Проверим, уникально ли число встречаемости каждого из городов в датафрейме - уникальность гарантируется, 
# если сумма количества появления каждого из 31 чисел встречаемости (для каждого города) совпадает 
# с количеством уникальных городов
print(len(df_all['City'].value_counts())==sum(Counter(df_all['City'].value_counts().values).values()))
for city, value in df_all['City'].value_counts().items():
    CitiesInfo[city]['code'] = value
TrainModel(displaydf=0, displayIMP=0) # предыдущий шаг
# в столбце 'City Code' присвоить городу нужное значение из словаря CityCodes
df_all['City code'] = df_all.City.apply(lambda x: CitiesInfo[x]['code'])
TrainModel()
TrainModel(displaydf=0, delete=['URL_TA gID'])
Heatmap(drawmap=0)
Pairplot()
TrainModel(displaydf=0, displayIMP=0) # предыдущий шаг
# для каждого города свой столбец со значением 0 или 1
CitiesDummiesCols = pd.get_dummies(df_all['City'], prefix='city') # dummy_na=True
# добавляем полученные столбцы в основной датафрейм
df_all = df_all.join(CitiesDummiesCols)
TrainModel(unite=[['CitiesDummiesCols']+list(CitiesDummiesCols.columns)])
TrainModel(displaydf=0, displayIMP=0, delete=['City code'])
TrainModel(displaydf=0, displayIMP=0, delete=['URL_TA gID'])
df_all.drop(['URL_TA gID'], axis=1, inplace=True)
Heatmap(drawmap=0)
# если столица страны (словарь CountriesInfo) города (словарь CitiesInfo) и город совпадают, ставим 1, нет - 0
df_all['City capital'] = df_all.City.apply(lambda x: 1 
                                           if CountriesInfo[CitiesInfo[x]['country']]['cap']==x else 0)
TrainModel(displaydf=0, displayIMP=0)
Pairplot(delete=list(CitiesDummiesCols.columns))
df_all.drop(['City capital'], axis=1, inplace=True)
TrainModel(displaydf=0, displayIMP=0) # предыдущий шаг
# 10 - если это столица (столица страны (словарь CountriesInfo) города (словарь CitiesInfo) и город совпадают),
# 5 - если не столица,
# 0 - не город ресторана
for city, info in CitiesInfo.items():
    df_all['city_'+city] = df_all.City.apply(lambda x: 10 
                                     if x==city and CountriesInfo[CitiesInfo[x]['country']]['cap']==x else 
                                     (5 if x==city else 0))
TrainModel(displayIMP=0)
df_all.drop(list(CitiesDummiesCols.columns), axis=1, inplace=True)
df_all = df_all.join(CitiesDummiesCols)
# для каждого города в столбце нужной страны поставить 1
for country in CountriesInfo.keys():
    df_all['country_'+country] = df_all.City.apply(lambda x: 1 
                                                   if country==CitiesInfo[x]['country'] else 0)
TrainModel(displaydf=0, displayIMP=0)
Heatmap(displayCORR=0, size=20)
for country in CountriesInfo.keys():
    df_all.drop(['country_'+country], axis=1, inplace=True)
TrainModel(displaydf=0, displayIMP=0) # предыдущий шаг
# Популяция города
df_all['Population'] = df_all.City.apply(lambda x: CitiesInfo[x]['pop'])
# Корреляция популяции города и количества отзывов
df_all['Population-Reviews corr'] = df_all.apply(lambda x: 
                                                 round(x['Number of Reviews']*100/x['Population'], 2), axis=1)
TrainModel(displaydf=0, displayIMP=0)
Heatmap(size=10)
Pairplot(delete=list(CitiesDummiesCols.columns))
df_all.drop(['Population', 'Population-Reviews corr'], axis=1, inplace=True)
TrainModel(displaydf=0, displayIMP=0) # предыдущий шаг
# значение 'Cuisines Number' равно длине списка из столбца 'Cuisine Style' или 0, если этот список пустой
df_all['Cuisines number']=df_all['Cuisine Style'].apply(lambda x: 
                                                        len(str(x).replace("['", "").
                                                            replace("']", "").
                                                            replace("', '", ",").
                                                            split(',')) 
                                                        if str(x).replace("['", "").
                                                        replace("']", "").
                                                        replace("', '", ",").split(',')[0]!='nan' else 0)
TrainModel(displaydf=0, displayIMP=0)
Heatmap(drawmap=0)
CuisinesDict = {} # словарь уникальных кухонь и их количества соответственно { кухня1: кол-во, ... }

# функция заполнения списка CuisinesList и словаря CuisinesDict
def CuisinesDictFILL(cuisines):
    global CuisinesDict
    # из входящей строки вида "['German', 'Central European', 'Vegetarian Friendly']"
    cuisines = str(cuisines).replace("['", "").replace("']", "").replace("', '", ",").split(',')
    # получили список отдельных значений вида ['German', 'Central European', 'Vegetarian Friendly']
    if cuisines[0]!='nan': # если список кухонь не пустой
        for i in cuisines: # перебор полученного списка
            if CuisinesDict.get(i)==None: # нет такой кухни в словаре CuisinesDict
                CuisinesDict[i] = 1 # добавить кухню (ключ) в CuisinesDict со значением 1
            else: CuisinesDict[i] += 1 # увеличить значение для этой кухни (ключа) в словаре CuisinesDict

df_all['Cuisine Style'].apply(CuisinesDictFILL)
print(sorted(CuisinesDict))
TrainModel(displaydf=0, displayIMP=0) # предыдущий шаг
Place = [ 'Bar', 'Brew Pub', 'Cafe', 'Cajun & Creole', 'Delicatessen', 'Diner', 'Gastropub', 
         'Pub', 'Steakhouse', 'Wine Bar' ]
Food = [ 'Barbecue', 'Fast Food', 'Grill', 'Halal', 'Kosher', 'Pizza',  'Seafood', 'Soups', 
        'Steakhouse', 'Street Food', 'Sushi' ]
Health = [ 'Healthy', 'Vegetarian Friendly', 'Vegan Options', 'Gluten Free Options' ]
    
Geo = []
for i in CuisinesDict.keys():
    if Place.count(i)==0 and Food.count(i)==0 and Health.count(i)==0: Geo.append(i)
df_all['Geo number']=0
for i in Geo: df_all['Geo number']+=df_all['Cuisine Style'].apply(lambda x: 1 if str(x).count(i)>0 else 0)
df_all['Place number']=0
for i in Place: df_all['Place number']+=df_all['Cuisine Style'].apply(lambda x: 1 if str(x).count(i)>0 else 0)
df_all['Food number']=0
for i in Food: df_all['Food number']+=df_all['Cuisine Style'].apply(lambda x: 1 if str(x).count(i)>0 else 0)
df_all['Health number']=0
for i in Health:
    df_all['Health number']+=df_all['Cuisine Style'].apply(lambda x: 1 if str(x).count(i)>0 else 0)
df_all[['Cuisine Style', 'Cuisines number', 
        'Geo number', 'Place number', 'Food number', 'Health number']].tail(5)
TrainModel(displaydf=0, displayIMP=0)
TrainModel(displaydf=0, displayIMP=0, delete=['Cuisines number'])
TrainModel(displaydf=0, displayIMP=0) # предыдущий шаг
df_all['Geo'] = df_all['Geo number'].apply(lambda x: 1 if x>0 else 0)
df_all['Place'] = df_all['Place number'].apply(lambda x: 1 if x>0 else 0)
df_all['Food'] = df_all['Food number'].apply(lambda x: 1 if x>0 else 0)
df_all['Health'] = df_all['Health number'].apply(lambda x: 1 if x>0 else 0)
df_all[['Cuisine Style', 'Cuisines number', 'Geo number', 'Place number', 'Food number', 'Health number',
        'Geo', 'Place', 'Food', 'Health']].tail(5)
# Cuisines number + Geo number, Place number, Food number, Health number + Geo, Place, Food, Health
TrainModel(displaydf=0, displayIMP=0)
# Cuisines number + Geo, Place, Food, Health
TrainModel(displaydf=0, displayIMP=0, 
           delete=['Geo number', 'Place number', 'Food number', 'Health number'])
# Geo number, Place number, Food number, Health number + Geo, Place, Food, Health
TrainModel(displaydf=0, displayIMP=0, delete=['Cuisines number'])
df_all.drop(['Geo number', 'Place number', 'Food number', 'Health number'], axis=1, inplace=True)
TrainModel(displaydf=0, displayIMP=0) # предыдущий шаг
for i in CuisinesDict.keys(): # список всех уникальных кухонь
    df_all[i]=df_all['Cuisine Style'].apply(lambda x: 1 if str(x).count(i)>0 else 0)
TrainModel(displayIMP=0)
Heatmap(drawmap=0)
TrainModel(displaydf=0, displayIMP=0) # предыдущий шаг
print(CountriesInfo[CitiesInfo['Lyon']['country']]['cuis'], 
      CountriesInfo[CitiesInfo['Lyon']['country']]['reg'])
df_all['CuisineStyle Local']=df_all.apply(lambda x: 1 
                                          if list(CuisinesDict.keys()).count(
                                              CountriesInfo[CitiesInfo[x['City']]['country']]['cuis'])>0 
                                          and x[CountriesInfo[CitiesInfo[x['City']]['country']]['cuis']]==1
                                          else 0, axis=1)
df_all['CuisineStyle Regional']=0
for city, data in CitiesInfo.items():
    for region in CountriesInfo[data['country']]['reg']:
        df_all['CuisineStyle Regional']=df_all.apply(lambda x: 1 if x[region]==1 and x['City']==city 
                                                     else x['CuisineStyle Regional'], axis=1)
df_all[df_all['City']=='Paris'][['City', 'Cuisine Style', 'French', 'European', 'Mediterranean', 
                                 'Cuisines number', 'Geo', 'Place', 'Food', 'Health', 
                                 'CuisineStyle Local', 'CuisineStyle Regional']].tail(5)
TrainModel(displaydf=0, displayIMP=0)
# сегодняшняя дата
DateNow = datetime.date(datetime.now())
def ReviewsDates(reviews):
    global datetime, date, DateNow, DateCounter
    
    ReviewDate = date(1900,1,1)
    # "[['Good food at your doorstep', 'A good hotel restaurant'], ['12/31/2017', '11/20/2017']]"
    # "[['Pastries have never tasted so good!'], ['06/16/2017']]"
    # "[[], []]"
    if reviews!=reviews: reviews="[[], []]"
    reviewsList = reviews[reviews.find("], [")+4:reviews.rfind("]]")].replace("'","").split(", ")
    # оставили только даты и превратили в список
    if reviewsList[0]!='': # если есть данные
        if DateCounter==0 or (DateCounter==1 and len(reviewsList)>1): # DateCounter - внешний счетчик
            ReviewDate = date(int(reviewsList[DateCounter][6::]), 
                              int(reviewsList[DateCounter][0:2:]), 
                              int(reviewsList[DateCounter][3:5:])) # год, месяц, день
    return ReviewDate

for DateCounter in range(0,2): # счетчик нужен, чтобы заполнить инфо для обеих дат (если они есть)
    df_all['Review'+str(DateCounter+1)+' date'] = df_all.Reviews.apply(ReviewsDates)
TrainModel(displaydf=0, displayIMP=0, delete=['Review1 date', 'Review2 date']) # предыдущий шаг
# возраст отзывов, выраженный в днях; 
# вычисляется вычитанием сегодняшней даты из даты отзыва - отрицательный результат сохраняет приоритетность
# более новых отзывов относительно более старых; если дата не указана - используется дата 1900-01-01 
for i in range(1, 3):
    temp='Review'+str(i)+' '
    df_all[temp+'days']=df_all[temp+'date'].apply(lambda x: 
                                                  int(str(x-DateNow)[:str(x-DateNow).find(" days"):]))
    df_all[temp+'years']=df_all[temp+'days'].apply(lambda x: round(x/365, 1))
df_all[['Reviews', 'Review1 date', 'Review1 days', 'Review1 years', 
        'Review2 date', 'Review2 days', 'Review2 years']].head(3)
TrainModel(displaydf=0, delete=['Review1 date', 'Review2 date'], 
           unite=[['CitiesCols']+list(CitiesDummiesCols.columns), 
                  ['CuisinesCols']+list(CuisinesDict.keys())])
df_all['ReviewsDIFF days']=-abs(abs(df_all['Review1 days'])-abs(df_all['Review2 days']))
df_all['ReviewsDIFF years']=df_all['ReviewsDIFF days'].apply(lambda x: round(x/365, 1))
TrainModel(displaydf=0, displayIMP=0, delete=['Review1 date', 'Review2 date'])
df_all.drop(['ReviewsDIFF days', 'ReviewsDIFF years'], axis=1, inplace=True)
days = [x for x in range(1, 32)]
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
          'September', 'October', 'November', 'December'] # = значения 1 - 12
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] # значения 0 - 6
TrainModel(displaydf=0, displayIMP=0, delete=['Review1 date', 'Review2 date']) # предыдущий шаг
for i in range(1,3):
    A='Review'+str(i)
    B='Review'+str(i)+' date'
    df_all[A+' month'] = df_all[B].apply(lambda x: x.month if x.year!=1900 else -1)
    df_all[A+' weekday'] = df_all[B].apply(lambda x: x.weekday() if x.year!=1900 else -1)
df_all[['Review1 date', 'Review1 days', 'Review1 years', 'Review1 month', 'Review1 weekday', 
        'Review2 date', 'Review2 days', 'Review2 years', 'Review2 month', 'Review2 weekday']].sample(5)
TrainModel(displaydf=0, displayIMP=0, delete=['Review1 date', 'Review2 date'])
df_all.drop(['Review1 month', 'Review2 month', 'Review1 weekday', 'Review2 weekday'], axis=1, inplace=True)
for mnth in months:
    df_all['RM_'+mnth]=0
    df_all['RM_'+mnth]+=df_all['Review1 date'].apply(lambda x: 1 if months[x.month-1]==mnth 
                                                     and x.year!=1900 else 0)
    df_all['RM_'+mnth]+=df_all['Review2 date'].apply(lambda x: 1 if months[x.month-1]==mnth 
                                                     and x.year!=1900 else 0)
for wkdy in weekdays:
    df_all['RW_'+wkdy]=0
    df_all['RW_'+wkdy]+=df_all['Review1 date'].apply(lambda x: 1 if weekdays[x.weekday()]==wkdy 
                                                     and x.year!=1900 else 0)
    df_all['RW_'+wkdy]+=df_all['Review2 date'].apply(lambda x: 1 if weekdays[x.weekday()]==wkdy 
                                                     and x.year!=1900 else 0)
ShowMonthsWeekdaysCols=[]
for mnth in months:
    ShowMonthsWeekdaysCols.append('RM_'+mnth)
for wkdy in weekdays:
    ShowMonthsWeekdaysCols.append('RW_'+wkdy)
df_all[['Review1 date', 'Review2 date']+ShowMonthsWeekdaysCols].sample(5)
TrainModel(displaydf=0, displayIMP=0, delete=['Review1 date', 'Review2 date']) # предыдущий шаг
df_all.drop(ShowMonthsWeekdaysCols, axis=1, inplace=True)
TrainModel(displaydf=0, delete=['Review1 date', 'Review2 date'], 
           unite=[['CitiesCols']+list(CitiesDummiesCols.columns), ['CuisinesCols']+list(CuisinesDict.keys())])
Heatmap(size=30, displayCORR=0)
# УБИРАЕМ ВСЕ ЛИШНИЕ СТОЛБЦЫ (нечисловые данные) ИЗ ДАТАФРЕЙМА
DropCols = [column for column in df_all.columns if df_all[column].dtypes=='object']
df = df_all.drop(DropCols, axis=1) # df - столбцы с числовыми данными
df = df.fillna(0) # заполняем пропуски 0 (нулями) 
# РАЗБИВАЕМ ДАТАФРЕЙМ НА ЧАСТИ, НЕОБХОДИМЫЕ ДЛЯ ОБУЧЕНИЯ И ТЕСТИРОВАНИЯ МОДЕЛИ:
train_data = df.query('sample == 1').drop(['sample'], axis=1)
X = train_data.drop(['Rating'], axis=1) # Х - данные с информацией о ресторанах
y = train_data['Rating'] # у - целевая переменная (рейтинги ресторанов)
# нужный инструмент для разбивки уже загружен:
#from sklearn.model_selection import train_test_split
# Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.
# Для тестирования мы будем использовать 25% от исходного датасета.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED)
# СОЗДАЁМ, ОБУЧАЕМ И ТЕСТИРУЕМ МОДЕЛЬ:
# необходимые библиотеки уже импортированы:
#from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
#from sklearn import metrics # инструменты для оценки точности модели
# создаём модель
regr = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED) # , verbose=1, n_jobs=-1
# обучаем модель на тестовом наборе данных
regr.fit(X_train, y_train)
# используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
# предсказанные значения записываем в переменную y_pred
y_pred = np.round(regr.predict(X_test), 1)
# MAE - СРАВНИВАЕМ ПРЕДСКАЗАННЫЕ ЗНАЧЕНИЯ (y_pred) С РЕАЛЬНЫМИ (y_test):
MAE = metrics.mean_absolute_error(y_test, y_pred)
print('\nMAE:', round(MAE, 2), end='\t')

# ACCURACY - ОПРЕДЕЛЯЕМ ПОКАЗАТЕЛИ ЭФФЕКТИВНОСТИ:
errors = abs(y_pred - y_test) # считаем абсолютые ошибки
mape = 100 * (errors / y_test) # считаем средний абсолютный процент ошибки (MAPE)
accuracy = 100 - np.mean(mape) # считаем и выводим точность
print('Точность:', round(accuracy, 2), '%')

# ОКРУГЛЯЕМ ПРЕДСКАЗАННЫЕ ЗНАЧЕНИЯ В СООТВЕТСТВИИ С ШАГОМ ЗНАЧЕНИЙ РЕЙТИНГА
# (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
for i in range(0,len(y_pred)): y_pred[i]=(round(y_pred[i]*2,0)/2)
MAE = metrics.mean_absolute_error(y_test, y_pred) # = abs(y_pred - y_test)
print('    ', round(MAE, 2), end='\t')
errors = abs(y_pred - y_test) # считаем абсолютые ошибки
mape = 100 * (errors / y_test) # считаем средний абсолютный процент ошибки (MAPE)
accuracy = 100 - np.mean(mape) # считаем и выводим точность
print('         ', round(accuracy, 2), '%\t(после округления предсказанных значений)')
# в RandomForestRegressor есть возможность вывести самые важные признаки для модели
plt.rcParams['figure.figsize'] = (10, 5)
feat_importances = pd.Series(regr.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
test_data = df.query('sample == 0').drop(['sample'], axis=1)
test_data.sample(10)
# test_data - данные с информацией о ресторанах, тестовая выборка
test_data.drop(['Rating'], axis=1, inplace=True)
# используем обученную на тестовом наборе (X_train, y_train) данных модель regr для предсказания рейтинга 
# ресторанов в тестовой выборке test_data
# предсказанные значения записываем в переменную predict_submission
predict_submission = np.round(regr.predict(test_data), 1)
predict_submission[:15]
# ОКРУГЛЯЕМ ПРЕДСКАЗАННЫЕ ЗНАЧЕНИЯ В СООТВЕТСТВИИ С ШАГОМ ЗНАЧЕНИЙ РЕЙТИНГА
# (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
for i in range(0,len(predict_submission)): predict_submission[i]=(round(predict_submission[i]*2,0)/2)
predict_submission[:15]
sample_submission['Rating'] = predict_submission
sample_submission.to_csv('/kaggle/working/submission.csv', index=False)
sample_submission.head(10)
