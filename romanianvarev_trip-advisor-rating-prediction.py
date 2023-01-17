# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline

# Загружаем модули для работы с датой и временем
from datetime import datetime
from datetime import timedelta

# Загружаем модуль для работы с регулярными выражениями
import re

# Загружаем специальный удобный инструмент для разделения датасета:
from sklearn.model_selection import train_test_split

import math # для расчета натуральных логарифмов

import requests 
from bs4 import BeautifulSoup

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

RANDOM_SEED = 42
!pip freeze > requirements.txt
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'
df_train = pd.read_csv(DATA_DIR+'/main_task.csv')
df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')
sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
df_train.head()
data[data.City == 'Luxembourg']
"""Посмотрим на пропуски и на то, какие данные могут быть категориальными"""
df_train.info()
df_test.head()
df_test.info()
# Для того, чтобы обработать признаки в обеих частях (и в тренировочной и тестовой) объединяем эти части в один датасет
df_train['sample'] = 1 # помечаем трейн
df_test['sample'] = 0 # помечаем наш тест
df_test['Rating'] = 0 # заполняем нулями рейтинг в тестовой части, где его пока нет
data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
data.nunique(dropna=False)
"""Посмотрим на пропуски в данных"""
plt.figure(figsize = (5,5))
sns.heatmap(data = data.isnull())
print(data.Reviews[5], type(data.Reviews[5]))
print(data['Cuisine Style'][5], type(data['Cuisine Style'][5]))
#данные грязные, например ревью и типы кухни спарсили списком, но по формату данные строковые
"""Заполним пустые значения количества ревью нулями, но создадим отдельный признак, говорящий о пустом значении ревью."""
data['rev_isna'] = pd.isna(data['Number of Reviews']).astype('uint8')
data['Number of Reviews'].fillna(0, inplace=True)
"""Ещё раз посмотрим на признаки"""
data.nunique(dropna = False)
"""Создадим признак rest_ratio, на основании количества ресторанов в городе и популяции"""
restaurants = data.City.value_counts()
population = pd.Series ({
    'London':8.982, 'Paris':2.148,'Madrid':6.642,'Barcelona':5.575,
    'Berlin':3.769,'Milan':1.352,'Rome':2.873, 'Prague':1.309,
    'Lisbon':0.548, 'Vienna':1.897, 'Amsterdam':0.822, 'Brussels':0.174,
    'Hamburg':1.899, 'Munich':1.472, 'Lyon':0.513, 'Stockholm':0.976,
    'Budapest':1.752, 'Warsaw':1.708, 'Dublin':1.388, 'Copenhagen':0.602,
    'Athens':0.664, 'Edinburgh':0.482, 'Zurich':0.403, 'Oporto':0.214,
    'Geneva':0.499, 'Krakow':0.760, 'Oslo':0.681, 'Helsinki':0.631,
    'Bratislava':0.424, 'Luxembourg':0.614, 'Ljubljana':0.293})
rest_ratio = restaurants / 1000 / population
data['rest_ratio'] = data.apply(lambda x: rest_ratio[x.City], axis = 1)
data['dCity'] = data['City']
data = pd.get_dummies(data, columns=[ 'dCity'])
capitals = ['Mariehamn', 'Tirana', 'Andorra la Vella', 'Vienna',
            'Minsk', 'Brussels', 'Sarajevo', 'Sofia',
            'Zagreb', 'Nicosia', 'Prague', 'Copenhagen',
            'Tallinn', 'Tórshavn', 'Helsinki', 'Paris',
            'Berlin', 'Gibraltar', 'Athens', 'St. Peter Port',
            'Budapest', 'Reykjavik', 'Dublin', 'Douglas',
            'Rome', 'Saint Helier', 'Pristina', 'Riga',
            'Vaduz', 'Vilnius', 'Luxembourg', 'Skopje',
            'Valletta', 'Chișinău', 'Monaco', 'Podgorica',
            'Amsterdam', 'Oslo', 'Warsaw', 'Lisbon',
            'Bucharest', 'Moscow', 'City of San Marino', 'Belgrade',
            'Bratislava', 'Ljubljana', 'Madrid', 'Longyearbyen',
            'Stockholm', 'Bern', 'Kiev', 'London', 'Vatican City']
data['is_capital'] = data.City.apply(lambda x: 1 if x in capitals else 0)
data['Price Range'].value_counts()
data['price_isna'] = pd.isna(data['Price Range']).astype('uint8') #создадим признак отсутствия ценового диапазона
data['Price Range'] = data.apply(lambda x: x['Price Range'].replace('$$ - $$$', '2').replace('$$$$', '3').replace('$', '1')
                                 if type(x['Price Range']) == str else 2,axis = 1)
data['Price Range'] = data.apply(lambda x: float(x['Price Range']), axis = 1)
data.sample(5)
"""Определим несколько полезных функций для работы с датами публикации ревью"""
# def d_time (cell): #функция которая возвращает разницу по времени в днях между двумя последними ревью
#     try:
#         x = cell.split(',')[2:]
#         x = pd.Series(x).apply(lambda x: x.replace('[','').replace(']','').replace("'",'').replace(' ',''))
#         x = x.apply(lambda x: datetime.strptime(x,'%d/%m/%Y'))
#         delta_x=x[1]-x[0]
#         return abs(delta_x.days)
#     except:
#         return -1 # чтобы значения NaN не мешали и выделялись кардинально 
def fresh_date (cell): #Функция, которая вычисляет дату самого свежего отображенного отзыва для каждого ресторана"""
    try:
        x = cell.split(',')[2:]
        x = pd.Series(x).apply(lambda x: x.replace('[','').replace(']','').replace("'",'').replace(' ',''))
        x = x.apply(lambda x: datetime.strptime(x,'%d/%m/%Y'))
        fresh_x = max(x[0],x[1])
        return fresh_x
    except:
        return None
"""Создадим синтетический признак на основании даты самого свежего отзыва о ресторане"""
data['rev_date'] = data.apply(lambda x: fresh_date(x['Reviews']), axis = 1)
data['rev_date'] = data['rev_date'].apply(lambda x: (x - datetime.now()).days)
data['rev_date'].fillna(data['rev_date'].mean(), inplace = True)
data['rev_date'] = (data['rev_date'] - data['rev_date'].mean())
data['rev_date'] = (data['rev_date'] / np.linalg.norm(data['rev_date']))
"""Создадим признак для ресторанов где тип кухни не упомянут"""
data['cuisine_isna'] = pd.isna(data['Cuisine Style']).astype('uint8')
"""Очистим строки от всего лишнего, включая пробелы и оставим только разделители-запятые"""
data['Cuisine Style'] = data.apply(lambda x: x['Cuisine Style'].replace('[','').replace(']','').replace("'",'').replace(' ','') 
                                   if type(x['Cuisine Style']) != float else x['Cuisine Style'], axis = 1)

"""Разберем признак который представлен строкой, на дамми-переменные по разделителю"""
styles = data['Cuisine Style'].str.get_dummies(',').sum().sort_values(ascending = False)
styles_drop = [x for x in styles.index if styles[x] < 100] # изначально ограничимся только признаками которые имеют больше 1000 ресторанов

"""Присоединим получившийся датафрейм новых признаков """
data = data.join(data['Cuisine Style'].str.get_dummies(',').drop(styles_drop, axis = 1), how = 'left')

styles[:50]
data.VegetarianFriendly.value_counts() #для проверки посчитаем количество вегетарианских заведений до обработки
pattern = re.compile('[A-Z][a-z]*') #регулярное выражение которое вытащит из признаков теги
def fill_styles (row):
    for style in styles.drop(styles_drop).index:
        x = pattern.match(style)[0]
        try:
            if x.lower() in row.Reviews.lower(): #ищем теги в отзывах
                row[style] = 1
        except:
            continue
    return row
data = data.apply(lambda x: fill_styles(x),axis = 1)
data.VegetarianFriendly.value_counts() # и после
local = pd.Series ({
    'London':'British', 'Paris':'French','Madrid':'Spanish','Barcelona':'Spanish',
    'Berlin':'German','Milan':'Italian','Rome':'Italian', 'Prague':'Czech',
    'Lisbon':'Portuguese', 'Vienna':'Austrian', 'Amsterdam':'Dutch', 'Brussels':'Belgian',
    'Hamburg':'German', 'Munich':'German', 'Lyon':'French', 'Stockholm':'Scandinavian',
    'Budapest':'Hungarian', 'Warsaw':'Polish', 'Dublin':'British', 'Copenhagen':'Scandinavian',
    'Athens':'Greek', 'Edinburgh':'British', 'Zurich':'CentralEuropean', 'Oporto':'Portuguese',
    'Geneva':'EasternEuropean', 'Krakow':'Polish', 'Oslo':'Scandinavian', 'Helsinki':'Scandinavian',
    'Bratislava':'EasternEuropean', 'Luxembourg':'French', 'Ljubljana':'EasternEuropean'})
data['is_local'] = data.apply(lambda x: 1 if x[local[x.City]] == 1 else 0, axis = 1)
data.sample(5)
df_train = data[data['sample'] == 1]
"""Распределение признака ranking"""
fig, axes = plt.subplots(1, 2, figsize=(20, 10));
df_train['Ranking'].hist(bins=100, ax=axes[0])
df_train.boxplot(column='Ranking', ax=axes[1])
"Посмотрим на распределение признака Ranking по городам (слева) и типам кухонь (справа)"
fig, axes = plt.subplots(1, 2, figsize=(20, 10));
for x in (df_train['City'].value_counts())[0:10].index:
    df_train['Ranking'][df_train['City'] == x].hist(bins=100, ax = axes[0])
for x in styles[0:10].index:
    df_train[df_train[x] == 1]['Ranking'].hist(bins = 100, ax = axes[1])
plt.show()
df_train['sqrt_ranking'] = data.apply(lambda x: x.Ranking**(1/3), axis = 1)
"Посмотрим на распределение признака Ranking по городам (слева) и типам кухонь (справа)"
fig, axes = plt.subplots(1, 2, figsize=(20, 10));
for x in (df_train['City'].value_counts())[0:10].index:
    df_train['sqrt_ranking'][df_train['City'] == x].hist(bins=100, ax = axes[0])
for x in styles[0:10].index:
    df_train[df_train[x] == 1]['sqrt_ranking'].hist(bins = 100, ax = axes[1])
plt.show()

#видим, что кубический корень атрибута Ranking по кухням распределен нормально, возьмем его в сет признаков
df_train['ln_ranking'] = data.apply(lambda x: math.log(x.Ranking), axis = 1)
"Посмотрим на распределение натурального логарифма признака Ranking по городам (слева) и типам кухонь (справа)"
fig, axes = plt.subplots(1, 2, figsize=(20, 10));
for x in (df_train['City'].value_counts())[0:10].index:
    df_train['ln_ranking'][df_train['City'] == x].hist(bins=100, ax = axes[0])
for x in styles[0:10].index:
    df_train[df_train[x] == 1]['ln_ranking'].hist(bins = 100, ax = axes[1])
plt.show()

c_mat = data.drop(['sample'], axis=1).corr()
plt.rcParams['figure.figsize'] = (30,25)
sns.heatmap(c_mat)
print('Ранг матрицы - {}, det(c_mat) = {}'.format(np.linalg.matrix_rank(c_mat), np.linalg.det(c_mat)))
c_mat.shape
# """Нормализуем признак Ranking относительно количества заведений в городе"""
# data['nRanking'] = data.Ranking / data.nCity 
# data.drop(['Ranking','nCity'], axis = 1, inplace = True)
# #было в старой ревизии, nCity заменено на rest_ratio
"""Объединим бары и пабы, японскую кухню и суши"""
data['Bar_Pub'] = data.Bar | data.Pub
data.drop(['Bar','Pub'], axis = 1, inplace = True)

data['Japan_Sushi'] = data.Japanese | data.Sushi
data.drop(['Japanese','Sushi'], axis = 1, inplace = True)
"""Объединим признаки об отсутствии ценового диапазона и кухни в один признак data_missing"""
data['data_missing'] = data.cuisine_isna | data.price_isna
data.drop(['cuisine_isna','price_isna'], axis = 1, inplace = True)
"""Объединим веганов с вегетарианцами"""
data['vegan_and_veg'] = data.VeganOptions | data.VegetarianFriendly
data.drop(['VeganOptions','VegetarianFriendly'], axis = 1, inplace = True)
# #ещё раз взглянув на матрицу корелляции, можно увидеть, что признак data_missing высоко обратно скореллирован с Price Range (-0.9)
# c_mat = data.drop(['sample'], axis=1).corr()
# print(c_mat['Price Range'].data_missing)
# #можно избавиться от data_missing
# data.drop(['data_missing'], axis = 1, inplace = True)

# так было в старой ревизии
c_mat = data.drop(['sample'], axis=1).corr() #проверим насколько поменялось значение определителя матрицы
print('Ранг матрицы - {}, det(c_mat) = {}'.format(np.linalg.matrix_rank(c_mat), np.linalg.det(c_mat)))
c_mat.shape
c_mat = data.drop(['sample'], axis=1).corr()
plt.rcParams['figure.figsize'] = (30,25)
sns.heatmap(c_mat)
data.drop('is_capital', axis = 1, inplace = True)
#импортируем все заново
df_train = pd.read_csv(DATA_DIR+'/main_task.csv')
df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')
df_train['sample'] = 1 # трейн
df_test['sample'] = 0 # тест
df_test['Rating'] = 0 # таргет

data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
data.info()
def preproc_data(df_input):
    '''includes several functions to pre-process the predictor data.'''
    
    df_output = df_input.copy()
    
    # ################### 1. Предобработка ############################################################## 
    # убираем не нужные для модели признаки
    df_output.drop(['Restaurant_id','ID_TA','URL_TA'], axis = 1, inplace=True)
    
    
    # ################### 2. NAN ############################################################## 
    """Заполним пустые значения количества ревью нулями, но создадим отдельный признак, говорящий о пустом значении ревью."""
    df_output['rev_isna'] = pd.isna(df_output['Number of Reviews']).astype('uint8')
    df_output['Number of Reviews'].fillna(0, inplace=True)
    
    
    # ################### 3. Encoding ##############################################################
    """Создадим dummy-признаки на основании городов"""
    df_output['dCity'] = df_output.City
    df_output = pd.get_dummies(df_output, columns=['dCity'], dummy_na=True)
    
    df_output['price_isna'] = pd.isna(df_output['Price Range']).astype('uint8') #создадим признак отсутствия ценового диапазона
    df_output['Price Range'] = df_output.apply(lambda x: x['Price Range'].replace('$$ - $$$', '2').replace('$$$$', '3').replace('$', '1')
                                 if type(x['Price Range']) == str else 2,axis = 1)
    df_output['Price Range'] = df_output.apply(lambda x: float(x['Price Range']), axis = 1)
    
    """Очистим строку со стилями кухни от лишних символов"""
    df_output['cuisine_isna'] = pd.isna(data['Cuisine Style']).astype('uint8')
    df_output['Cuisine Style'] = df_output.apply(lambda x: x['Cuisine Style'].replace('[','').replace(']','').replace("'",'').replace(' ','') 
                                   if type(x['Cuisine Style']) != float else x['Cuisine Style'], axis = 1)

    """Разберем признак который представлен строкой, на дамми-переменные по разделителю"""
    styles = df_output['Cuisine Style'].str.get_dummies(',').sum().sort_values(ascending = False)
    styles_drop = [x for x in styles.index if styles[x] < 100] # ограничимся только признаками которые имеют больше 100 ресторанов

    """Присоединим получившийся датафрейм новых признаков """
    df_output = df_output.join(df_output['Cuisine Style'].str.get_dummies(',').drop(styles_drop, axis = 1), how = 'left')
    
    """Дозаполним получившиеся dummy-признаки на основании ревью"""
    df_output = df_output.apply(lambda x: fill_styles(x),axis = 1)
    
    """Объединим признаки об отсутствии ценового диапазона и кухни в один признак data_missing"""
    df_output['data_missing'] = df_output.cuisine_isna | df_output.price_isna
    df_output.drop(['cuisine_isna','price_isna'], axis = 1, inplace = True)
    
    
    # ################### 4. Feature Engineering ####################################################
    
    """Создадим признак rest_ratio, на основании количества ресторанов в городе и популяции"""
    population = pd.Series ({
    'London':8.982, 'Paris':2.148,'Madrid':6.642,'Barcelona':5.575,
    'Berlin':3.769,'Milan':1.352,'Rome':2.873, 'Prague':1.309,
    'Lisbon':0.548, 'Vienna':1.897, 'Amsterdam':0.822, 'Brussels':0.174,
    'Hamburg':1.899, 'Munich':1.472, 'Lyon':0.513, 'Stockholm':0.976,
    'Budapest':1.752, 'Warsaw':1.708, 'Dublin':1.388, 'Copenhagen':0.602,
    'Athens':0.664, 'Edinburgh':0.482, 'Zurich':0.403, 'Oporto':0.214,
    'Geneva':0.499, 'Krakow':0.760, 'Oslo':0.681, 'Helsinki':0.631,
    'Bratislava':0.424, 'Luxembourg':0.614, 'Ljubljana':0.293})
    rest_ratio = restaurants / 1000 / population
    df_output['rest_ratio'] = df_output.apply(lambda x: rest_ratio[x.City], axis = 1)
    
    local = pd.Series ({
    'London':'British', 'Paris':'French','Madrid':'Spanish','Barcelona':'Spanish',
    'Berlin':'German','Milan':'Italian','Rome':'Italian', 'Prague':'Czech',
    'Lisbon':'Portuguese', 'Vienna':'Austrian', 'Amsterdam':'Dutch', 'Brussels':'Belgian',
    'Hamburg':'German', 'Munich':'German', 'Lyon':'French', 'Stockholm':'Scandinavian',
    'Budapest':'Hungarian', 'Warsaw':'Polish', 'Dublin':'British', 'Copenhagen':'Scandinavian',
    'Athens':'Greek', 'Edinburgh':'British', 'Zurich':'СentralEuropean', 'Oporto':'Portuguese',
    'Geneva':'EasternEuropean', 'Krakow':'Polish', 'Oslo':'Scandinavian', 'Helsinki':'Scandinavian',
    'Bratislava':'EasternEuropean', 'Luxembourg':'French', 'Ljubljana':'EasternEuropean'})
    df_output['is_local'] = df_output.apply(lambda x: 1 if x[local[x.City]] == 1 else 0,axis = 1)
    
    """Cоздадим признак is_capital - столица ли город"""
    capitals = ['Mariehamn', 'Tirana', 'Andorra la Vella', 'Vienna',
            'Minsk', 'Brussels', 'Sarajevo', 'Sofia',
            'Zagreb', 'Nicosia', 'Prague', 'Copenhagen',
            'Tallinn', 'Tórshavn', 'Helsinki', 'Paris',
            'Berlin', 'Gibraltar', 'Athens', 'St. Peter Port',
            'Budapest', 'Reykjavik', 'Dublin', 'Douglas',
            'Rome', 'Saint Helier', 'Pristina', 'Riga',
            'Vaduz', 'Vilnius', 'Luxembourg', 'Skopje',
            'Valletta', 'Chișinău', 'Monaco', 'Podgorica',
            'Amsterdam', 'Oslo', 'Warsaw', 'Lisbon',
            'Bucharest', 'Moscow', 'City of San Marino', 'Belgrade',
            'Bratislava', 'Ljubljana', 'Madrid', 'Longyearbyen',
            'Stockholm', 'Bern', 'Kiev', 'London', 'Vatican City']
    df_output['is_capital'] = df_output.City.apply(lambda x: 1 if x in capitals else 0)
    
#     """Нормируем признак Ranking относительно количества ресторанов в городе"""
#     df_output['pRanking'] = df_output.apply(lambda x: x.Ranking / population[x.City], axis = 1)
    """Нормируем признак Ranking относительно количества ресторанов и людей в городе"""
    df_output['nRanking'] = df_output.apply(lambda x: x.Ranking / restaurants[x.City],
                                            axis = 1)
    df_output['сuberoot_rank'] = df_output.apply(lambda x: x.Ranking**(1/3) / restaurants[x.City], axis = 1)
    df_output['ln_ranking'] = df_output.apply(lambda x: math.log(x.Ranking), axis = 1)
    df_output.drop('Ranking', axis = 1, inplace = True) #оригинал дропнем
        
    """Объединим бары и пабы, японскую кухню и суши"""
    df_output['Bar_Pub'] = df_output.Bar | df_output.Pub
    df_output.drop(['Bar','Pub'], axis = 1, inplace = True)

    df_output['Japan_Sushi'] = df_output.Japanese | df_output.Sushi
    df_output.drop(['Japanese','Sushi'], axis = 1, inplace = True)
    
    """Объединим веганов с вегетарианцами"""
    df_output['vegan_and_veg'] = df_output.VeganOptions | df_output.VegetarianFriendly
    df_output.drop(['VeganOptions','VegetarianFriendly'], axis = 1, inplace = True)
    
    """Создадим синтетический признак на основании даты самого свежего отзыва о ресторане"""
    df_output['rev_date'] = df_output.apply(lambda x: fresh_date(x['Reviews']), axis = 1)
    df_output['rev_date'] = df_output['rev_date'].apply(lambda x: (x - datetime.now()).days)
    df_output['rev_date'].fillna(df_output['rev_date'].mean(), inplace = True)
    df_output['rev_date'] = df_output['rev_date'] - df_output['rev_date'].mean()
    df_output['rev_date'] = (df_output['rev_date'] / np.linalg.norm(df_output['rev_date'])) * 10**15
    
    df_output['rev_ratio'] = df_output.apply(lambda x: x['Number of Reviews'] / population[x.City],
                                            axis = 1)
   
    
    
    # ################### 5. Clean #################################################### 
    # убираем признаки которые еще не успели обработать, 
    # модель на признаках с dtypes "object" обучаться не будет, просто выберим их и удалим
    object_columns = [s for s in df_output.columns if df_output[s].dtypes == 'object']
    df_output.drop(object_columns, axis = 1, inplace=True)
    
    # ################### 6. Essential feature set  #################################################### 
#     features = ['nRanking','Rating','Number of Reviews', 'rest_ratio', 'rev_ratio', 'rev_date',
#                  'сuberoot_rank', 'dCity_Rome','dCity_Madrid', 'Price Range', 'data_missing',
#                 'dCity_Amsterdam','sample']
#     df_output = df_output[features]
    
    return df_output
df_preproc = preproc_data(data)
df_preproc.sample(10)
"""Ещё раз посмотрим на все признаки, убедимся что нет пропусков"""
df_preproc.info()
# Теперь выделим тестовую часть
train_data = df_preproc.query('sample == 1').drop(['sample'], axis=1)
test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1)

y = train_data.Rating.values            # таргет
X = train_data.drop(['Rating'], axis=1)
# разобьем данные, выделим 20% на валидацию
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape # проверим
# Импортируем необходимые библиотеки:
from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
from sklearn import metrics # инструменты для оценки точности модели
# Создаём модель
regr = RandomForestRegressor(n_estimators=100)

# Обучаем модель на тестовом наборе данных
regr.fit(X_train, y_train)

# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
# Предсказанные значения записываем в переменную y_pred
y_pred = regr.predict(X_test)
y_pred = (y_pred * 2).round() / 2
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются
# Cчитаем Mean Absolute Error (MAE)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# Посмотрим на важность фич
plt.rcParams['figure.figsize'] = (10,10)
feat_importances = pd.Series(regr.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
test_data.sample(10)
test_data = test_data.drop(['Rating'], axis=1)
sample_submission
predict_submission = regr.predict(test_data)
predict_submission = (predict_submission * 2).round() / 2
predict_submission
sample_submission['Rating'] = predict_submission
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head(10)
