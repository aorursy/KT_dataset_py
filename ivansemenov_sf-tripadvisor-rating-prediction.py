# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели

import math as math

from collections import Counter

from datetime import datetime, date



# Загружаем инструмент для разделения датасета

from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Для воспроизводимости результатов зададим:

# - Общий параметр для генерации случайных чисел

RANDOM_SEED = 42

# - Общую текущую дату

current_date = pd.to_datetime('28/04/2020')
# Зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

!pip freeze > requirements.txt
# Зададим путь к папке с данными

DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'



# Сформируем пути к датасетам

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')



sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')



df_train.columns = ['restaurant_id','city','cuisine_style','ranking','rating','price_range','reviews_number','reviews',

             'url_ta','id_ta']

df_test.columns = ['restaurant_id','city','cuisine_style','ranking','price_range','reviews_number','reviews',

             'url_ta','id_ta']



# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['rating'] = 0 # в тесте у нас нет значения rating, мы его должны предсказать, поэтому пока просто заполняем нулями



# Oбъединяем

df = df_test.append(df_train, sort=False).reset_index(drop=True)

df.info()
df.head()
# Зафиксируем информацию о наличии пропусков в отзывах, ценовом сегменте и кухнях

df['number_of_reviews_is_nan'] = pd.isna(df['reviews_number']).astype('uint8')

df['price_range_is_nan'] = pd.isna(df['price_range']).astype('uint8')

df['cuisine_style_is_nan'] = pd.isna(df['cuisine_style']).astype('uint8')



# Заполним пропуски в числе отзывов на 0

df['reviews_number'] = df['reviews_number'].fillna(0)



# Создание признака "сетевой ресторан"

restaurant_chain = set()

for chain in df['restaurant_id']:

    restaurant_chain.update(chain)

def find_item(cell):

    if item in cell:

        return 1

    return 0

for item in restaurant_chain:

    df['restaurant_chain'] = df['restaurant_id'].apply(find_item)  

    

# Относительный ранг

rest_count = df.city.value_counts()

def get_rest_count(value):

    return rest_count[value]

# чем больше к единице, тем выше ранг ресторана

df['relative_ranking'] = 1-(df['ranking'] / df['city'].map(df.groupby(['city'])['ranking'].max()))



# Значения из столбца 'Restaurant_id' без 'id_'

df['restaurant_id'] = df['restaurant_id'].apply(lambda x: int(x[3::]))



# В url_ta есть 2 цифровых значения: g1068497 и d12160475;

# значения типа d12160475 полностью совпадает со значением в 'id_ta'

# Извлечем цифровые данные из значений типа g1068497

df['url_ta_gid'] = df['url_ta'].apply(lambda x: int((str(x).split('-'))[1][1::]))

df['id_ta'] = df['id_ta'].apply(lambda x: int(x[1::]))



# Заполним пропуски наиболее часто встречающимся значением в price_range

df['price_range'].fillna('$$ - $$$', inplace=True)



# Создадим справочник с диапазонами цен

price_range_dict = {'$': 10,'$$ - $$$': 100, '$$$$': 1000}



# Заменим на числовые значения

df['price_range'] = df['price_range'].map(price_range_dict)
# Создадим словарь с указанием количества ресторанов для каждого города

res_count = {'Paris': 17593,'Stockholm': 3131,'London': 22366,'Berlin': 8110, 

             'Munich': 3367,'Oporto': 2060, 'Milan': 7940,'Bratislava': 1331,

             'Vienna': 4387, 'Rome': 12086,'Barcelona': 10086,'Madrid': 11562,

             'Dublin': 2706,'Brussels': 3703,'Zurich': 1901,'Warsaw': 3210,

             'Budapest': 3445, 'Copenhagen': 2637,'Amsterdam': 4189,'Lyon': 2833,

             'Hamburg': 3501, 'Lisbon': 4985,'Prague': 5850,'Oslo': 1441, 

             'Helsinki': 1661,'Edinburgh': 2248,'Geneva': 1753,'Ljubljana': 647,

             'Athens': 2814,'Luxembourg': 759,'Krakow': 1832}



# Создадим словарь, в котором ключами буду названия городов, а значениями 1, если этот город столица, в противном случае 0

city_capital = {'London': 1,'Paris': 1,'Madrid': 1,'Barcelona': 0,'Berlin': 1,

              'Milan': 0,'Rome': 1,'Prague': 1,'Lisbon': 1,'Vienna': 1,

              'Amsterdam': 1,'Brussels': 1,'Hamburg': 0,'Munich': 0,'Lyon': 0,

              'Stockholm': 1,'Budapest': 1,'Warsaw': 1,'Dublin': 1,'Copenhagen': 1,

              'Athens': 1,'Edinburgh': 1,'Zurich': 1,'Oporto': 0,'Geneva': 1,

              'Krakow': 1,'Oslo': 1,'Helsinki': 1,'Bratislava': 1,'Luxembourg': 1,'Ljubljana': 1}



# Создадим словарь с информацией о населении города

city_population = {'London': 8173900,'Paris': 2240621,'Madrid': 3155360,'Barcelona': 1593075,

                   'Berlin': 3326002,'Milan': 1331586,'Rome': 2870493,'Prague': 1272690,

                   'Lisbon': 547733,'Vienna': 1765649,'Amsterdam': 825080,'Brussels': 144784,

                   'Hamburg': 1718187,'Munich': 1364920,'Lyon': 496343,'Stockholm': 1981263,

                   'Budapest': 1744665,'Warsaw': 1720398,'Dublin': 506211 ,'Copenhagen': 1246611,

                   'Athens': 3168846,'Edinburgh': 476100,'Zurich': 402275,'Oporto': 221800,

                   'Geneva': 196150,'Krakow': 756183,'Oslo': 673469,'Helsinki': 574579,

                   'Bratislava': 413192,'Luxembourg': 576249,'Ljubljana': 277554}



# Создадим словарь, в котором ключами буду названия городов, а значениями соответствующая страна

city_country = {'London': 'UK','Paris': 'France','Madrid': 'Spain','Barcelona': 'Spain',

                'Berlin': 'Germany','Milan': 'Italy','Rome': 'Italy','Prague': 'Czech',

                'Lisbon': 'Portugalia','Vienna': 'Austria','Amsterdam': 'Nederlands','Brussels': 'Belgium',

                'Hamburg': 'Germany','Munich': 'Germany','Lyon': 'France','Stockholm': 'Sweden',

                'Budapest': 'Hungary','Warsaw': 'Poland','Dublin': 'Ireland' ,'Copenhagen': 'Denmark',

                'Athens': 'Greece','Edinburgh': 'Schotland','Zurich': 'Switzerland','Oporto': 'Portugalia',

                'Geneva': 'Switzerland','Krakow': 'Poland','Oslo': 'Norway','Helsinki': 'Finland',

                'Bratislava': 'Slovakia','Luxembourg': 'Luxembourg','Ljubljana': 'Slovenija'}
# Создадим числовой признак, является ли город столицей

df['city_capital'] = df['city'].map(city_capital)



# Создадим признак, отражающий количество ресторанов в городе, в котором расположен данный ресторан

df['restaurants_count'] = df['city'].map(res_count)



# Создадим признак с населением города

df['population'] = df['city'].map(city_population)



# Создадим признак со страной города

df['country'] = df['city'].map(city_country)



from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

countries = LabelEncoder()

countries.fit(df['country'])

df['country_code'] = countries.transform(df['country'])



# Добавим числовой признак cколько человек в городе в среднем приходится на один ресторан

df['people_per_restaurant'] = df['population'] / df['restaurants_count']



# Добавим числовой признак cколько ресторанов на душу населения

df['restaurant_per_people'] = df['restaurants_count'] / df['population']



# Создадим словарь с данными о количестве ресторанов в каждом городе

restaurant_count = dict(df['city'].value_counts())

df['rest_count'] = df['city'].map(restaurant_count)



# Добавим числовой признак количество отзывов*100 на душу населения

df['reviews_per_people100'] = df.apply(lambda x:

                                           round(x['reviews_number']*10000/x['population'], 4), axis=1)
# Значение 'cuisine_number' равно длине списка из столбца 'cuisine_style' или 3(медиана), если этот список пустой

df['cuisine_number']=df['cuisine_style'].apply(lambda x: 

                                                        len(str(x).replace("['", "").

                                                            replace("']", "").

                                                            replace("', '", ",").

                                                            split(',')) 

                                                        if str(x).replace("['", "").

                                                        replace("']", "").

                                                        replace("', '", ",").split(',')[0]!='nan' else 33)





# Cловарь уникальных кухонь и их количества соответственно { кухня1: кол-во, ... }

CuisinesDict = {} 

# Функция заполнения словаря CuisinesDict

def CuisinesDictFill(cuisines):

    global CuisinesDict

    # из входящей строки вида "['German', 'Central European', 'Vegetarian Friendly']"

    cuisines = str(cuisines).replace("['", "").replace("']", "").replace("', '", ",").split(',')

    # получили список отдельных значений вида ['German', 'Central European', 'Vegetarian Friendly']

    if cuisines[0]!='nan': # если список кухонь не пустой

        for i in cuisines: # перебор полученного списка

            if CuisinesDict.get(i)==None: # нет такой кухни в словаре CuisinesDict

                CuisinesDict[i] = 1 # добавить кухню (ключ) в CuisinesDict со значением 1

            else: CuisinesDict[i] += 1 # увеличить значение для этой кухни (ключа) в словаре CuisinesDict



df['cuisine_style'].apply(CuisinesDictFill)





# Разделим кухни по признакам: тип завеедения, тип еды, здоровые опции, национальная привязка

Place = [ 'Bar', 'Brew Pub', 'Cafe', 'Cajun & Creole', 'Delicatessen', 'Diner', 'Gastropub', 

         'Pub', 'Steakhouse', 'Wine Bar' ]

Food = [ 'Barbecue', 'Fast Food', 'Grill', 'Halal', 'Kosher', 'Pizza',  'Seafood', 'Soups', 

        'Steakhouse', 'Street Food', 'Sushi' ]

Health = [ 'Healthy', 'Vegetarian Friendly', 'Vegan Options', 'Gluten Free Options' ]



Geo = []



for i in CuisinesDict.keys():

    if Place.count(i)==0 and Food.count(i)==0 and Health.count(i)==0: Geo.append(i)

        

        

# Посчитаем количество каждого вида кухонь для каждого ресторана

df['geo_number']=0

for i in Geo: df['geo_number']+=df['cuisine_style'].apply(lambda x: 1 if str(x).count(i)>0 else 0)



df['place_number']=0

for i in Place: df['place_number']+=df['cuisine_style'].apply(lambda x: 1 if str(x).count(i)>0 else 0)



df['food_number']=0

for i in Food: df['food_number']+=df['cuisine_style'].apply(lambda x: 1 if str(x).count(i)>0 else 0)



df['health_number']=0

for i in Health: df['health_number']+=df['cuisine_style'].apply(lambda x: 1 if str(x).count(i)>0 else 0)

    

    

# Указаны ли у ресторана опции: тип заведения, тип еды, здоровые опции, национальная привязка

df['geo'] = df['geo_number'].apply(lambda x: 1 if x>0 else 0)

df['place'] = df['place_number'].apply(lambda x: 1 if x>0 else 0)

df['food'] = df['food_number'].apply(lambda x: 1 if x>0 else 0)

df['health'] = df['health_number'].apply(lambda x: 1 if x>0 else 0)





cities = LabelEncoder()

cities.fit(df['city'])

df['city_code'] = cities.transform(df['city'])





# Сделаем из кухонь dummy-переменные

for i in CuisinesDict.keys(): # список всех уникальных кухонь

    df[i]=df['cuisine_style'].apply(lambda x: 1 if str(x).count(i)>0 else 0)

    

# Вычислим среднее количество кухонь в ресторанах по городам

df['cuisine_count_mean'] = df['city'].map(df.groupby('city')['cuisine_number'].mean())
# "[['relatively quite good value', 'Cheap takeout'], ['08/01/2017', '07/23/2017']]"

df['reviews'].fillna("[[], []]")



# Количество дней между последними двумя отзывами ресторанов

import re

pattern = re.compile('\d+\W\d+\W\d+')

def reviews_dates(reviews):

    dates = []

    if type(reviews) is str:

        dates = pattern.findall(reviews)

    return dates

def get_days_reviews(value):

    if type(value) is list:

        if len(value) == 2:

            return abs((pd.to_datetime(str(value[0]))-pd.to_datetime(str(value[1]))).days)

    return 0



df['dates'] = df['reviews'].apply(reviews_dates)

df['days_between_reviews'] = df['dates'].apply(get_days_reviews)





# Вычтем из сегодняшней даты дату последнего отзыва

def get_days_today(value):

    if type(value) is list:

        if len(value) == 2:

            return abs((current_date - pd.to_datetime(str(value[0]))).days)

    return 7300



df['days_today'] = df['dates'].apply(get_days_today)
# Добавим новые признаки комбинируя наиболее важные признаки

# Другие комбинации также были опробованы, но при их реализации либо метрика ухудшалась,

# либо возникали ошибки, на исправление которых не хватило времени

from itertools import combinations



def numeric_interaction_terms(df, columns):

    fe_df = pd.DataFrame()

    for c in combinations(columns,2):

        fe_df['{} * {}'.format(c[0], c[1]) ] = df1[c[0]] * df1[c[1]]

    return fe_df



to_interact_cols = ['ranking', 'days_today', 'days_between_reviews', 

                    'cuisine_count_mean', 'reviews_number', 'id_ta', 

                    'reviews_per_people100', 'restaurant_id', 'cuisine_number']

df1 = pd.DataFrame

object_columns = [s for s in df.columns if df[s].dtypes == 'object']

df1 = df.drop(object_columns, axis = 1)



df2 = pd.DataFrame

df_fe = numeric_interaction_terms(df,to_interact_cols)

df2 = pd.concat([df1, df_fe], axis=1)
# Теперь выделим тестовую часть

train_data = df2.query('sample == 1').drop(['sample'], axis=1)

test_data = df2.query('sample == 0').drop(['sample'], axis=1)

y = train_data.rating.values

X = train_data.drop(['rating'], axis=1)





# Воспользуемся функцией train_test_split для разбивки тестовых данных

# Выделим 20% данных на валидацию (параметр test_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)



# проверяем

test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape



# Импортируем необходимые библиотеки:

from sklearn import metrics # инструменты для оценки точности модели



# Создаём модель (НАСТРОЙКИ НЕ ТРОГАЕМ)

model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)



# Обучаем модель на тестовом наборе данных

model.fit(X_train, y_train)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

y_pred = model.predict(X_test)
# Округляем полученные значения рейтингов

def round_d(rec):

    if rec <0.25:

        return 0

    elif 0.25<rec<=0.75:

        return 0.5

    elif 0.75<rec<=1.25:

        return 1

    elif 1.25<rec<=1.75:

        return 1.5

    elif 1.75<rec<=2.25:

        return 2

    elif 2.25<rec<=2.75:

        return 2.5

    elif 2.75<rec<=3.25:

        return 3

    elif 3.25<rec<=3.75:

        return 3.5

    elif 3.75<rec<=4.25:

        return 4

    elif 4.25<rec<=4.75:

        return 4.5

    else:

        return 5

    

for i in range(y_pred.size):

    y_pred[i]=round_d(y_pred[i])
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# MAE: 0.160625
# в RandomForestRegressor есть возможность вывести самые важные признаки для модели

plt.rcParams['figure.figsize'] = (10,10)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(15).plot(kind='barh')
test_data = df2.query('sample == 0').drop(['sample'], axis=1)

test_data = test_data.drop(['rating'], axis=1)
predict_submission = model.predict(test_data)
#Заметим, что Rating - величина, округляемая до 0.5

#Поэтому и мы окраглим свой predict до 0.5

predict_submission = predict_submission // 0.5 * 0.5 + np.round((predict_submission % 0.5) / 0.5) * 0.5
len(predict_submission)
current_date.today('Europe/Moscow')
sample_submission['Rating'] = predict_submission

sample_submission.to_csv(f'submission_1.csv', index=False)

sample_submission.head()