# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import re

import datetime



import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



# Загружаем специальный удобный инструмент для разделения датасета:

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!

RANDOM_SEED = 42
# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

!pip freeze > requirements.txt
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
df_train.info()
df_train.head(5)
df_test.info()
df_test.head(5)
sample_submission.head(5)
sample_submission.info()
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



df = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
df.info()
df.iloc[1]
df['Reviews'][1]
for item in df.iloc[0:10]['URL_TA']:

    print(item)
# Предрасчитываем максимальный ренкинг по каждому городу

ranking_max = df.groupby(['City'])['Ranking'].max()
# Функция переводит абсолютный ренкинг ресторана в относителный

# Количество групп задается параметром

def ranking_by_steps(row, q):

    step = ranking_max[row['City']] / q

    return math.ceil(row['Ranking'] / step)
# Создаем новый столбец, в который вносим данные по ренкингу в относительных единицах

# 1 - 10% самых популярных, 10 - 10% самых непопулярных

df['Ranking Group'] = df.apply(lambda row: ranking_by_steps(row, 10), axis=1)
# Функция преобразует строку с описанием стилей кухни в список этих стилей

def cousine_list(text):

    if not text is np.nan:

        cousine = []

        regex = re.compile('\'.+?\'')

        res = regex.findall(text)

        for item in res:

            cousine.append(item[1:-1])

        return cousine

    else:

        return text
# Функция рассчитывает количество стилей кухни ресторана

def cousine_count(text):

    if not text is np.nan:

        return len(cousine_list(text))

    else:

        return text
# Добавляем новый столбец с количеством стилей кухни ресторана

df['Cuisine Style Quantity'] = df['Cuisine Style'].apply(cousine_count)
# Создадим таблицу присутствия разных стилей в ресторанах

cousine_count = {}



# Создадим словарь с парой "кухня - кол-во ресторанов, в которых данная кухня присутствует"

for item in df['Cuisine Style']:

    cousines = cousine_list(item)

    if not cousines is np.nan:

        for val in cousines:

            if val in cousine_count:

                cousine_count[val] += 1

            else:

                cousine_count[val] = 1

cousine_count_df = pd.DataFrame(cousine_count,index=['quantity']).T.sort_values(by=['quantity'], ascending=False) 
# Отобразим, количество ресторанов, в которых присутствует данных стиль, чтобы оценить границы будущих групп

ax = cousine_count_df[0:20].plot.bar()
ax = cousine_count_df[16:60].plot.bar()
ax = cousine_count_df[60:100].plot.bar()
ax = cousine_count_df[80:].plot.bar()
# В таблицу стилей кухи добавим информацию о распространенности кухни. Мерой выберем целом число логарифмированного количества ресторанов,

# в котоорых присутсвует кухня. Две нижнии группы объединим в одну.

def wide_spread(cousine_count):

    if np.log10(cousine_count) >= 2:

        return int(np.log10(cousine_count))

    else:

        return 1
# Преобразуем индекс в столбец

# cousine_count_df['cuisine'] = cousine_count_df.index

# Добавлям столбец с данными по распространенности кухни

cousine_count_df['wide_spread'] = cousine_count_df['quantity'].apply(wide_spread)
# Функция для каждого ресторана определяет принадлежность его стилей к тому или иному классу распространнености

# В зависимости от паратметра выдает максимальный или минимальный класс имеющихся у ресторана стилей

def cousine_spread(row, val='max'):

    if not row['Cuisine Style'] is np.nan:  

        cousines = cousine_list(row['Cuisine Style'])

        spread = []

        for item in cousines:

            spread.append(cousine_count_df.loc[item,'wide_spread'])

        spread.sort()

        if val == 'max':

            return spread[-1]

        elif val == 'min':

            return spread[0]

        else:

            return np.nan    

    else:

        return np.nan
# Добавляем столбец класса с самой распространеной кухни

df['Cuisine Wide Spread'] = df.apply(lambda row: cousine_spread(row, 'max'), axis=1)



# Добавляем столбец класса с самой экзотической кухней

df['Cuisine Exotic'] = df.apply(lambda row: cousine_spread(row, 'min'), axis=1)
# Фунция извекает список с датами из текста

def review_dates(text):

    dates = []

    regex = re.compile('\d\d/\d\d\/\d\d\d\d')

    res = regex.findall(text)

    for item in res:

        dates.append(take_date(item))

    if len(dates)!=0:

        return dates

    else:

        np.nan
# Функция преобразует текст в даты

def take_date(text):

    return datetime.datetime.strptime(text, "%m/%d/%Y")
# Функция рассчтиывает разницу между последним отзывом и текущей датой

def review_last_date(text):

    if type(text) is str:

        if review_dates(text)!=None:

            return (datetime.datetime.today() - max(review_dates(text))).days

        else:

            return np.nan

    else:

        return np.nan
# Добавляем столбец с разнице в дня между текущей датой и последним обзором

df['Review Last Date'] = df['Reviews'].apply(review_last_date)
# Функция рассчитывает разницу в днях между двумя обзорами

def review_dif(text):

    if type(text) is str:

        dates = review_dates(text)

        if dates!=None and len(dates)==2:

            return abs((dates[0]-dates[1]).days)

        else:

            return np.nan  

    else:

        return np.nan
# Добавляем столбец с разнице в днях между двумя обзорами

df['Review Date Dif'] = df['Reviews'].apply(review_dif)
# Фунция рассчитывает количество отзывов на сайте

def reviews_quantity(row):

    if not np.isnan(row['Review Last Date']) and not np.isnan(row['Review Date Dif']):

        return 2

    elif not np.isnan(row['Review Last Date']):

        return 1

    else:

        return 0
# Добавляем столбец с количеством отзывов

df['Reviews Quantity'] = df.apply(lambda row: reviews_quantity(row), axis=1)
# Функция приводит даныне по уровню цен к числовому виду

def price_num(text):

    if text=='$':

        return 1.0

    elif text=='$$ - $$$':

        return 4.0

    elif text=='$$$$':

        return 7.0

    else:

        return np.nan
# Добавляем столбец с числовым предоставлением уровня цен

df['Price Range Num'] = df['Price Range'].apply(price_num)
df.head()
df.isna().sum()
df.loc[df['Number of Reviews'] == 0]['Number of Reviews'].count()
# Создаем новый столбец, в который вносим данные по ренкингу в относительных единицах:

# 1 - самые популярные, 4 - самые непопулярные

df['Ranking Group Test'] = df.apply(lambda row: ranking_by_steps(row, 4), axis=1)
df.loc[df['Number of Reviews'].isna()]['Ranking Group Test'].hist(bins=3)
df['Number of Reviews'].fillna(0, inplace=True)
# Предрасчитываем данные

calculated = pd.DataFrame(df.groupby(['City'])['Number of Reviews'].sum())



# Добавим показатель доли отзывов, приходящихся на ресторан в данном городе

def number_of_reviews_share(row):

    return row['Number of Reviews'] / calculated.loc[row['City']]



df['Number of Reviews Share'] = df.apply(lambda row: number_of_reviews_share(row), axis=1)
# Проверим, как распределены рестораны по свойствам Cuisine Wide Spread/ Cuisine Exotic

df.groupby(['Cuisine Wide Spread','Cuisine Exotic'])['Restaurant_id'].count()
# Для ресторанов, у которых пропущен признак "Cuisine Wide Spread", установим его равным 4

df['Cuisine Wide Spread Nan'] = df['Cuisine Wide Spread'].isna().astype('int8')

df['Cuisine Wide Spread'].fillna(4, inplace=True) 
# Для ресторанов, у которых пропущен признак "Cuisine Exotic", установим его равным мединанному значению

# по группе 4 "Cuisine Wide Spread"

df['Cuisine Exotic Nan'] = df['Cuisine Exotic'].isna().astype('int8')

df['Cuisine Exotic'].fillna(df.loc[df['Cuisine Wide Spread']==4]['Cuisine Exotic'].median(), inplace=True) 
# Количество стилей кухни установим равным медианному количеству в выборке "Cuisine Wide Spread" = 4

# "Cuisine Exotic" = 3

df['Cuisine Style Quantity Nan'] = df['Cuisine Style Quantity'].isna().astype('int8')

df['Cuisine Style Quantity'].fillna(df.loc[(df['Cuisine Wide Spread']==4) & 

                                          (df['Cuisine Exotic']==3)]['Cuisine Style Quantity'].median(), inplace=True) 
# Устанавливаем признак, что в столбцах нет данных

df['Review Last Date Nan'] = df['Review Last Date'].isna().astype('int8')

df['Review Date Dif Nan'] = df['Review Date Dif'].isna().astype('int8')
# Предрасчитываем данные для замены

calculated = pd.DataFrame(df.groupby(['City','Ranking Group'])['Review Last Date'].median())

max_last = df['Review Last Date'].max()



# Функция замены Nan в столбце Review Last Date

# Eсли количество отзывов равно нулю, то по ресторану нет отзывов

# в этом случае проставляем максимальное значение по столбцу

# Если количество не равно нулю, то отзывы есть, у нас не хватает данных. В этом случае проставляем

# медианное значение по группе ренкинга и города

def last_date_change(row):

    if row['Number of Reviews']==0:

        return max_last

    else:

        return calculated.loc[(row['City'], row['Ranking Group'])]
# Избавляемся от Nan

df.loc[df['Review Last Date'].isna(),'Review Last Date'] = df.apply(lambda row: last_date_change(row), axis=1)
# Предрасчитываем данные для замены

calculated = pd.DataFrame(df.groupby(['City','Ranking Group'])['Review Date Dif'].median())

max_dif = df['Review Date Dif'].max()



# Функция замены Nan в столбце Review Date Dif

# Eсли количество отзывов меньше 2, то по ресторану второго отзыва

# в этом случае проствляем максимальное значение по столбцу

# Если количество 2 и больше, то второй отзыв есть, у нас не хватает данных. В этом случае проставляем

# медианное значение по группе ренкинга и города

def date_dif_change(row):

    if not np.isnan(row['Review Date Dif']):

        return row['Review Date Dif']

    elif row['Number of Reviews']<2:

        return max_dif

    else:

        return calculated.loc[(row['City'], row['Ranking Group'])]
# Избавляемся от Nan

df['Review Date Dif'] = df.apply(lambda row: date_dif_change(row), axis=1)
# Заменяет Nan

df['Price Range Num Nan'] = df['Price Range'].isna().astype('int8')

# Отсутствующее указание ценового диапазона обычно свойственно более дешевым ресторанам

df['Price Range Num'].fillna(1, inplace=True)

df['Price Range Num'] = df['Price Range Num'].astype('int8')
df.info()
# В первых 16-ти городов находится более 80% все ресторанов 

cities_with_freqs = list(df['City'].value_counts(normalize=True))

sum(cities_with_freqs[0:16])
main = df['City'].value_counts().index[0:16]
# Функция оставляет первые 16-ть городов, остальные переименовывает в Others

def main_cities(text):

    if text in main:

        text = text

    else:

        text = 'Others'

    return text
# Добавляем столбцы-города

df['City New'] = df['City'].apply(main_cities)

city_dum = pd.get_dummies(df['City New'], drop_first=True)

df = df.join(city_dum)
df.head()
# Удаляем лишние столбцы

df.drop(['City','Cuisine Style','Price Range','Reviews','URL_TA','ID_TA',

        'City New', 'Ranking Group Test','Restaurant_id'],axis=1, inplace=True)
df.info()
# Создаем матрицу корреляции

plt.rcParams['figure.figsize'] = (15,10)

sns.heatmap(df.drop(['sample'], axis=1).corr(),)
df_preproc = df

df_preproc.sample(10)
df_preproc.info()
# Теперь выделим тестовую часть

train_data = df_preproc.query('sample == 1').drop(['sample'], axis=1)

test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1)



y = train_data.Rating.values            # наш таргет

X = train_data.drop(['Rating'], axis=1)
# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных

# выделим 20% данных на валидацию (параметр test_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
# проверяем

test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape
# Импортируем необходимые библиотеки:

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели
# Создаём модель (НАСТРОЙКИ НЕ ТРОГАЕМ)

model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)
# Обучаем модель на тестовом наборе данных

model.fit(X_train, y_train)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

y_pred = model.predict(X_test)
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# в RandomForestRegressor есть возможность вывести самые важные признаки для модели

plt.rcParams['figure.figsize'] = (10,10)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(15).plot(kind='barh')
test_data.sample(10)
test_data = test_data.drop(['Rating'], axis=1)
sample_submission
predict_submission = model.predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)