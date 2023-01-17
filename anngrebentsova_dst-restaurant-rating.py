# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip freeze > requirements.txt
from os.path import join

import datetime as dt

import re

import pandas as pd

from sklearn.ensemble import RandomForestRegressor  

from sklearn import metrics 

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
folder_ds = '/kaggle/input/sf-dst-restaurant-rating/'



# Зададим названия файлов датасетов

main_file = 'main_task.csv'

kaggle_file = 'kaggle_task.csv'



# Сформируем пути к датасетам

mt_path = join(folder_ds, main_file)

kt_path = join(folder_ds, kaggle_file)



# Загрузим датасеты

main_df = pd.read_csv(mt_path)

kaggle_df = pd.read_csv(kt_path)

#посмотрим информацию о сете и его содержание

kaggle_df.info()

kaggle_df.head(5)
#посмотрим информацию о сете и его содержание



main_df.info()

main_df.head(5)
# Добавим признак, чтобы отделить главную выборку от тестовой

main_df['Main_sample'] = True

kaggle_df['Main_sample'] = False



# Недостающие данные по названиям ресторанов в основной выборке заполним названиями вида Name-id, где id - идентификатор ресторана

main_df['Name'] = main_df['Restaurant_id'].apply(lambda x: 'Name-'+x)



# Недостающие данные по рейтингу в тестовой выборке заполним нулями

kaggle_df['Rating'] = 0



# Объединим датасеты в один

df = pd.concat([main_df, kaggle_df])

df.info()
#переименуем колнки

df = df.rename(columns={'Number of Reviews': 'Number_of_Reviews',

                        'Price Range':'Price_Range',

                        'Cuisine Style':'Cuisine_Style'})
#запоним пропущенные значения 0

df.Number_of_Reviews.fillna(0, inplace = True)
#добавим столбец численный столица, где 1 - значит, что город является столицей 0 - нет

capitals = ['London', 'Paris', 'Madrid', 'Berlin', 'Rome', 'Prague', 

            'Lisbon', 'Vienna', 'Amsterdam', 'Brussels', 'Stockholm', 

            'Budapest', 'Warsaw', 'Dublin', 'Copenhagen', 'Athens', 

            'Oslo', 'Helsinki', 'Bratislava', 'Luxembourg', 'Ljubljana', 'Edinburgh']

cap = []

for i in df['City']:

    if i in capitals:

        cap.append(1)

    else:

        cap.append(0)

df['Capital'] = cap
#поменяем ценовой сегмент

df['Price_Range']=df['Price_Range'].apply(lambda pr: 1 if pr == '$' else (2 if pr == '$$ - $$$' else (3 if pr == '$$$$' else 0))) 
#посчитем кухни в каждом ресторане, создадим на основе подсчетов новый признак

df['Number_of_Style'] = df['Cuisine_Style'].str[2:-2].str.split("', '").str.len()

df.Number_of_Style.fillna(0, inplace = True)
#заполним пропуски

df['Reviews'] = df['Reviews'].fillna("[[], []]")



#Добавим новый признак -  количество дней между отзывами

pattern = re.compile('\d{2}/\d{2}/\d{4}')

reviews=[]



for i in df['Reviews']:

    reviews.append(re.findall(pattern, i))

df1=pd.DataFrame(reviews)



#выделим даты из Reviews

df['Reviews'] = df['Reviews'].str.replace(pattern, '')

df['Reviews'] = df['Reviews'].str[2:-2].str.split("', '")



#создадим колонки с датами 

df1.columns=['date1', 'date2']



df['date1'] = pd.to_datetime(df1['date1']) 

df['date2'] = pd.to_datetime(df1['date2']) 



#посчитаем количество дней между отзывами

df['Number_of_days']=df['date2']-df['date1']

df['Number_of_days']=df['Number_of_days']*(-1)



#поменяем тип данных

df['Number_of_days'] = df['Number_of_days'].dt.days.astype('float64')

df['Number_of_days']=df['Number_of_days'].apply(lambda nd: nd*(-1) if nd < 0 else nd)



#заполним 0 отсутсвующие значения

df.Number_of_days.fillna(0, inplace = True)

dfn=df
# Посмотрим в каких городах расположены рестораны, чтобы добавить информацию о населении

df['City'].unique()
#создадим таблицу с данными о населении каждого города

pop = {"Square2018":pd.Series([105.4, 188, 1572, 310.4, 891.8, 

                               41.42, 181.8, 367.6, 414.6, 1285, 101.9, 604.3, 32.61, 

                               115,	87.88, 517.2, 525.2, 88.25,	219.3, 47.87, 755.2, 100, 

                               496, 454, 213.8, 264, 15.93, 163.8, 2929, 2586, 327], 

                              index=['Paris', 'Stockholm', 'London', 'Munich', 'Berlin', 'Oporto',	

                                     'Milan', 'Bratislava',	'Vienna', 'Rome', 'Barcelona', 'Madrid',

                                     'Brussels', 'Dublin',	'Zurich', 'Warsaw', 'Budapest', 'Copenhagen', 

                                     'Amsterdam', 'Lyon', 'Hamburg',	'Lisbon', 'Prague',	'Oslo',	

                                     'Helsinki',	'Edinburgh', 'Geneva',	'Ljubljana', 'Athens', 

                                     'Luxembourg', 'Krakow']),

"Population2018": pd.Series([2187526, 962154, 8173941, 1525618, 3644826, 287591, 1378689, 

                             429564, 1888776, 2876614, 1636762, 3223334, 1208542, 5531650, 

                             428340, 1777972, 1752286, 1263700, 855000, 516092, 1841179, 550000, 

                             1262000, 673469, 643272, 513210, 201741, 278789, 3753783, 602005, 769498], 

                            index=['Paris', 'Stockholm', 'London', 'Munich', 'Berlin', 'Oporto',

                                   'Milan', 'Bratislava', 'Vienna', 'Rome', 'Barcelona', 'Madrid', 

                                   'Brussels', 'Dublin', 'Zurich', 'Warsaw', 'Budapest', 'Copenhagen', 

                                   'Amsterdam', 'Lyon', 'Hamburg', 'Lisbon', 'Prague', 'Oslo', 

                                   'Helsinki', 'Edinburgh', 'Geneva', 'Ljubljana', 'Athens', 

                                   'Luxembourg', 'Krakow'])}

pop = pd.DataFrame(pop)

pop['City'] = pop.index



#посчитаем плотность населения и добавим столбец ч данными в таблицу

pop['SquarePopulation'] = pop['Population2018']/pop['Square2018']
#объединим таблицу с данными о населении с сетом

df = pd.merge(dfn, pop, on = 'City')
df.info()
#посмотрим на рейтинги

#Разбиваем датафрейм на части, необходимые для обучения и тестирования модели  

# Х - данные с информацией о ресторанах, у - целевая переменная (рейтинги ресторанов)  



X = df[df['Main_sample']].drop(['Restaurant_id', 'Rating', 'Cuisine_Style', 'City', 

             'Reviews', 'URL_TA', 'date1', 'date2', 'Name',

             'ID_TA'], axis = 1)  



y = df[df['Main_sample']]['Rating']



# Загружаем специальный инструмент для разбивки:  

from sklearn.model_selection import train_test_split 



# Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.  

# Для тестирования мы будем использовать 25% от исходного датасета.  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)  



# Создаём модель  

regr = RandomForestRegressor(n_estimators=100)



# Обучаем модель на тестовом наборе данных  

regr.fit(X_train, y_train)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.  

# Предсказанные значения записываем в переменную y_pred  

y_pred = regr.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# Создадим датасет с предскаханными рейтингами

my_df = pd.DataFrame()

my_df['Restaurant_id'] = df[~df['Main_sample']]['Restaurant_id']

my_df['Rating'] = y_pred

#округлим рейтинги

my_df['Rating']=my_df['Rating'].apply(lambda r: round(r,1))
my_df
my_df.to_csv('MySolution.csv', index = False)  