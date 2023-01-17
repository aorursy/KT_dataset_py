import pandas as pd

import numpy as np

import json

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели

from collections import Counter

from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

RANDOM_SEED = 42
!pip freeze > requirements.txt
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating'



main_task = pd.read_csv(os.path.join(DATA_DIR, 'main_task.csv'))

kaggle_task = pd.read_csv(os.path.join(DATA_DIR, 'kaggle_task.csv'))
main_task.info()
main_task.head(3)
kaggle_task.info()
kaggle_task.head(3)
# Изначально работала для извлечения признаков велась только на основе main_task.csv

# Потом внимательно посмотрел доступный baseline на kaggle (https://www.kaggle.com/itslek/baseline-sf-tripadvisor-rating-v2-7)

# и решил объеденить оба массива данных, чтобы их стало больше!))





main_task['is_main'] = True

kaggle_task['is_main'] = False



df = pd.concat([main_task, kaggle_task])

df.head()

# извлечение признаков из городов



# переводим категории (города) в числовые признаки

cities_coder = LabelEncoder()

cities_coder.fit(df['City'])

df['city_code'] = cities_coder.transform(df['City'])  



# колличество рестаранов на в городе

df['rest_count'] = df['City'].map(df.groupby(['City'])['City'].count().to_dict())



# население города

population = {

 'London': 8788,

 'Paris': 10958,

 'Madrid': 6559,

 'Barcelona': 5541,

 'Berlin': 3557,

 'Milan': 3136,

 'Rome': 4234,

 'Prague': 1319,

 'Lisbon': 2942,

 'Vienna': 1915,

 'Amsterdam': 1140,

 'Brussels': 2065,

 'Hamburg': 1791,

 'Munich': 1521,

 'Lyon': 1705,

 'Stockholm': 974,

 'Budapest': 1764,

 'Warsaw': 1776,

 'Dublin': 1215,

 'Copenhagen': 1334,

 'Athens': 3154,

 'Edinburgh': 531,

 'Zurich': 1383,

 'Oporto': 238,

 'Geneva': 202,

 'Krakow': 1832,

 'Oslo': 1020,

 'Helsinki': 1292,

 'Bratislava': 426,

 'Luxembourg': 614,

 'Ljubljana': 284,

}

df['population'] = df['City'].map(population)

df['people_per_res'] = df['population'] / df['rest_count']



# страна

country_mapping = {

 'London': 'UK',

 'Paris': 'France',

 'Madrid': 'Spain',

 'Barcelona': 'Spain',

 'Berlin': 'Germany',

 'Milan': 'Italy',

 'Rome': 'Italy',

 'Prague': 'Czech',

 'Lisbon': 'Portugalia',

 'Vienna': 'Austria',

 'Amsterdam': 'Nederlands',

 'Brussels': '144784 ',

 'Hamburg': 'Germany',

 'Munich': 'Germany',

 'Lyon': 'France',

 'Stockholm': 'Sweden',

 'Budapest': 'Hungary',

 'Warsaw': 'Poland',

 'Dublin': 'Ireland',

 'Copenhagen': 'Denmark',

 'Athens': 'Greece',

 'Edinburgh': 'Schotland',

 'Zurich': 'Switzerland',

 'Oporto': 'Portugalia',

 'Geneva': 'Switzerland',

 'Krakow': 'Poland',

 'Oslo': 'Norway',

 'Helsinki': 'Finland',

 'Bratislava': 'Slovakia',

 'Luxembourg': 'Luxembourg',

 'Ljubljana': 'Slovenija'

}

df['country'] = df['City'].map(country_mapping)

country_encoder = LabelEncoder()

country_encoder.fit(df['country'])

df['country_code'] = country_encoder.transform(df['country'])  

df = df.drop(['country'], axis=1)



# столица 

capitals = {

 'Amsterdam': True,

 'Athens': True,

 'Barcelona': False,

 'Berlin': True,

 'Bratislava': True,

 'Brussels': True,

 'Budapest': True,

 'Copenhagen': True,

 'Dublin': True,

 'Edinburgh': True,

 'Geneva': False,

 'Hamburg': False,

 'Helsinki': True,

 'Krakow': False,

 'Lisbon': True,

 'Ljubljana': True,

 'London': True,

 'Luxembourg': True,

 'Lyon': False,

 'Madrid': True,

 'Milan': False,

 'Munich': False,

 'Oporto': False,

 'Oslo': True,

 'Paris': True,

 'Prague': True,

 'Rome': True,

 'Stockholm': True,

 'Vienna': True,

 'Warsaw': True,

 'Zurich': False

}

# capitals = pd.read_csv('capitals.csv')

# capitals.head()

# capitals_set = capitals.capital.str.lower().unique()



# def is_capital(city):

#     return city.lower() in capitals_set



# df['capitals'] = df.City.apply(lambda x: is_capital(x))

# df.groupby(['City'])['capitals'].max().to_dict()

df['is_capital'] = df['City'].map(capitals)
df.head()
# Кол-во отзывов

# подсмотрел у своих коллег - бинарный признак, а были отзывы?..

df['has_review'] = ~df['Number of Reviews'].isna()



# избавляемся от пропусков

df['Number of Reviews'].fillna(1, inplace=True)



# кол-во человек на каждое ревью

df['people_on_review'] = df['Number of Reviews'] / df['population']
df.head()
# Ранги (Ranking)

df['relative_ranking'] = df['Ranking']/df['rest_count']
df.head()
# Кухни



# помечаем тех, у кого нет никакой информации о кухнях

df['has_cousine'] = ~df['Cuisine Style'].isna()

df['Cuisine Style'].fillna('[]', inplace=True)



def as_list(cs):

    return list(map(lambda x: x.strip(),  cs[1:-1].replace('\'', '').split(',')))



def count_of_cs(cs):

    count = len(list(map(lambda x: x.strip(), as_list(cs))))

    return count or 1



df['cousines_count'] = df['Cuisine Style'].apply(count_of_cs)





counter = Counter()

def inc_counter(x):

    counter[x] += 1

    

for cs in df['Cuisine Style']:

    list(map(lambda x: inc_counter(x), as_list(cs)))

    

counter.most_common()



all_coisines = counter.keys()



# уникальные кухни взял из каунтера выше, ключи со значением 1

unique_cs = ['Salvadoran', 'Xinjiang', 'Burmese', 'Latvian']

def has_unique_cs(cs):

    for x in unique_cs:

        if x in cs:

            return True

    return False



df['has_unique_coisine'] = df['Cuisine Style'].apply(has_unique_cs)



def has_cs(cs, t):

    return t in cs



for cs in all_coisines:

    df[cs] = df['Cuisine Style'].apply(lambda x: has_cs(x, cs))
df.head()
df['Price Range'].value_counts()
# поковыряем Price Range

df['has_price_range'] = ~df['Price Range'].isna()



# df.fillna('$', inplace=True)

df.fillna('$$ - $$$', inplace=True)  # почему-то лучший результат... хз.

# df.fillna('$$$$', inplace=True)



price_encoder = LabelEncoder()

price_encoder.fit(df['Price Range'])

df['price_code'] = price_encoder.transform(df['Price Range'])  



df['price_as_num'] = df['Price Range'].map({'$': 1, '$$ - $$$': 100, '$$$$': 1000})
df.head()
# попробуем извлечь признак из ID_TA



# idta_encoder = LabelEncoder()

# idta_encoder.fit(df['ID_TA'])

# df['id_ta_code'] = idta_encoder.transform(df['ID_TA'])



df['id_ta_code'] = df['ID_TA'].apply(lambda x: int(x[1:]))  # опять же хз почему, но про2стой парсинг поля дал лучший показатель
df.head()
# Ревью рестаранов

def extract_days_count(x):

        data = x.split('], [')[-1][1:-3].split('\', \'')

        if not all(data):

            return 0

        try:

            d1 = pd.to_datetime(data[0], format='%m/%d/%Y')

            d2 = pd.to_datetime(data[1], format='%m/%d/%Y')

            result = (d1 - d2).days

        except (IndexError, ValueError):

            result = 0

        return result



df['days_count'] = df['Reviews'].apply(extract_days_count)
df.head()
# удаляем колонки с нечисловыми значениями

training_df = df[df['is_main']]

training_df = training_df.drop(['Restaurant_id', 'City', 'Reviews', 'URL_TA', 'ID_TA', 'Price Range', 'Cuisine Style', 'is_main'], axis=1)
# Х - данные с информацией о ресторанах, у - целевая переменная (рейтинги ресторанов)

X = training_df.drop(['Rating'], axis = 1)

y = training_df['Rating']
# Подготовка тренировочных и тестовых данных

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED)
# Создаём модель

model = RandomForestRegressor(n_estimators=100, verbose=1, random_state=RANDOM_SEED)

model.fit(X_train, y_train)
# в RandomForestRegressor есть возможность вывести самые важные признаки для модели

plt.rcParams['figure.figsize'] = (10,10)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(15).plot(kind='barh')
# модель вангует на тестовых данных

def rating_round(x, base=0.5):

    return base * round(x/base)



def predict(ds):

    return np.array([rating_round(x) for x in model.predict(ds)])



y_pred = predict(X_test)
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
kaggle_df = df[~df['is_main']]

kaggle_df = kaggle_df.drop(['Rating', 'Restaurant_id', 'City', 'Reviews', 'URL_TA', 'ID_TA', 'Price Range', 'Cuisine Style', 'is_main'], axis=1)
kaggle_pred = predict(kaggle_df)
subm_df = pd.DataFrame()

subm_df['Restaurant_id'] = df[~df['is_main']]['Restaurant_id']

subm_df['Rating'] = kaggle_pred

subm_df.head()
subm_df.to_csv('submission.csv', index=False)