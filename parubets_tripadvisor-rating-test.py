# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



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



# Any results you write to the current directory are saved as output.
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



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
data.info()
data.sample(5)
data.Reviews[1]
# Для примера я возьму столбец Number of Reviews

data['Cuisine Style_isNAN'] = pd.isna(data['Cuisine Style']).astype('uint8')

data['Price Range_isNAN'] = pd.isna(data['Price Range']).astype('uint8')

data['Number of ReviewsNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')

data['Cuisine StyleNAN'] = pd.isna(data['Cuisine Style']).astype('uint8')
data['Number of ReviewsNAN']
data['Number of Reviews'].value_counts()
data['Number of Reviews'].mode()
# Далее заполняем пропуски значением '2'

data['Number of Reviews'] = data['Number of Reviews'].fillna(

    data['Number of Reviews'].mode().iloc[0])
data.sample(20)
data.nunique(dropna=False)
data.sample(5)
data['Price Range'].value_counts()
data['Price Range'] = data['Price Range'].apply(

    lambda x: 1 if x == '$' else 2 if x == '$$ - $$$' else 3 if x == '$$$$' else 2)
# добавляем информацию по населению городу в млн чел

population = {

    'London': 8.98,

    'Paris': 2.48,

    'Madrid': 6.64,

    'Barcelona': 5.58,

    'Berlin': 3.77,

    'Milan': 1.35,

    'Rome': 2.87,

    'Prague': 1.3,

    'Lisbon': 0.5,

    'Vienna': 1.9,

    'Amsterdam': 0.8,

    'Brussels': 0.174,

    'Hamburg': 1.9,

    'Munich': 1.472,

    'Lyon': 0.5,

    'Stockholm': 1,

    'Budapest': 1.752,

    'Warsaw': 1.708,

    'Dublin': 1.388,

    'Copenhagen': 0.6,

    'Athens': 0.66,

    'Edinburgh': 0.5,

    'Zurich': 0.4,

    'Oporto': 0.2,

    'Geneva': 0.5,

    'Krakow': 0.8,

    'Oslo': 0.681,

    'Helsinki': 0.631,

    'Bratislava': 0.424,

    'Luxembourg': 0.613,

    'Ljubljana': 0.279

}
# добавляем информацию по зарплатам в EUR

salary = {

    'London': 2460,

    'Paris': 3617,

    'Madrid': 3000,

    'Barcelona': 2700,

    'Berlin': 3944,

    'Milan': 2500,

    'Rome': 1846,

    'Prague': 1400,

    'Lisbon': 860,

    'Vienna': 3406,

    'Amsterdam': 2855,

    'Brussels': 3000,

    'Hamburg': 3296,

    'Munich': 3566,

    'Lyon': 3455,

    'Stockholm': 2700,

    'Budapest': 750,

    'Warsaw': 900,

    'Dublin': 2500,

    'Copenhagen': 2700,

    'Athens': 1938,

    'Edinburgh': 3000,

    'Zurich': 4000,

    'Oporto': 1901,

    'Geneva': 6388,

    'Krakow': 1652,

    'Oslo': 2916,

    'Helsinki': 2500,

    'Bratislava': 1932,

    'Luxembourg': 4000,

    'Ljubljana': 1100

}

# добавляем информацию по кол-ву туристов в млн чел

tourists = {

    'London': 21.7,

    'Paris': 50,

    'Madrid': 7.6,

    'Barcelona': 32,

    'Berlin': 13.4,

    'Milan': 11,

    'Rome': 42,

    'Prague': 6.67,

    'Lisbon': 7,

    'Vienna': 7,

    'Amsterdam': 10,

    'Brussels': 8.5,

    'Hamburg': 5,

    'Munich': 2.5,

    'Lyon': 5,

    'Stockholm': 1.8,

    'Budapest': 2.8,

    'Warsaw': 25,

    'Dublin': 10.6,

    'Copenhagen': 8.8,

    'Athens': 5.7,

    'Edinburgh': 3,

    'Zurich': 10,

    'Oporto': 13,

    'Geneva': 5,

    'Krakow': 8.1,

    'Oslo': 5.8,

    'Helsinki': 8.5,

    'Bratislava': 1.4,

    'Luxembourg': 5.6,

    'Ljubljana': 5.9

}
data['tourists'] = data['City'].apply(lambda x: tourists[x])

data['population'] = data['City'].apply(lambda x: population[x])

data['salary'] = data['City'].apply(lambda x: salary[x])
# добавим количество кухонь в каждом ресторане, при отсутствии информации напишем "No information",

# т.е. будем считать, что представлен как минимум один вид



data['Cuisine Style'] = data['Cuisine Style'].fillna('No information]').apply(lambda x: x[x.find(

    '[')+1:x.find(']')].replace("'", '').replace(', ', ',').strip(' ').split(','))
data['Cuisine_count'] = data['Cuisine Style'].apply(lambda x: len(x))
from collections import Counter



сuisine_count = []

for x in data['Cuisine Style']:

    for i in x:

        сuisine_count.append(i)

Counter(сuisine_count).most_common()[0:7]
data['Vegetarian Friendly'] = data['Cuisine Style'].apply(

    lambda x: 1 if 'Vegetarian Friendly' in x else 0)

data['European'] = data['Cuisine Style'].apply(

    lambda x: 1 if 'European' in x else 0)

data['Mediterranean'] = data['Cuisine Style'].apply(

    lambda x: 1 if 'Mediterranean' in x else 0)

data['Italian'] = data['Cuisine Style'].apply(

    lambda x: 1 if 'Italian' in x else 0)

data['Vegan Options'] = data['Cuisine Style'].apply(

    lambda x: 1 if 'Vegan Options' in x else 0)

data['Gluten Free Options'] = data['Cuisine Style'].apply(

    lambda x: 1 if 'Gluten Free Options' in x else 0)
data['id_ta'] = data['ID_TA'].apply(lambda x: int(x[x.find('d')+1:]))
data['Reviews'].isna().sum()
# заполним пропуски модой

data['Reviews'] = data['Reviews'].fillna(

    data['Reviews'].mode().iloc[0])
#ф-ция для обработки времени 

def date_(x):

    if x == '[[], []]':

        return []

    else:

        x = x.replace(']]', '')

        x = x.replace("'", '')

        x = x.split('], [')[1]

        x = x.split(', ')

        return x
# работаем с датой 

data['Reviews_date'] = data['Reviews'].apply(date_)

data['Reviews_date_first'] = data['Reviews_date'].apply(

    lambda x: x[1] if len(x) == 2 else None)

data['Reviews_date_last'] = data['Reviews_date'].apply(

    lambda x: x[0] if len(x) > 0 else None)



# Преобразуем в формат дат

data['Reviews_date_first'] = pd.to_datetime(data['Reviews_date_first'])

data['Reviews_date_last'] = pd.to_datetime(data['Reviews_date_last'])
# заполним пропуски

data['Reviews_date_first'] = data['Reviews_date_first'].fillna(

    data['Reviews_date_first'].min())

data['Reviews_date_last'] = data['Reviews_date_last'].fillna(

    data['Reviews_date_last'].max())
# считаем дельту в днях

data['Delta_days'] = data['Reviews_date_last'] - data['Reviews_date_first']

data['Delta_days'] = data['Delta_days'].apply(lambda x: x.days)
# смотрим день недели последнего отзыва

data['Day_of_week_last_review'] = data['Reviews_date_last'].dt.dayofweek
data.info()
#### Стандартизация не привела к улучшению модели

#### upd. позже была найдена информация, что нормализация/стандартизация не влияют на качество данной модели :(
# Стандартизация

#from sklearn.preprocessing import StandardScaler

#scaler = StandardScaler()

#df[['population', 'salary', 'tourists', 'Ranking', 'Number of Reviews', 'Cuisine_count',

#    'id_ta', 'Delta_days']] = scaler.fit_transform(df[['population', 'salary', 'tourists', 'Ranking', 'Number of Reviews', 'Cuisine_count',

#                                                       'id_ta', 'Delta_days']])
plt.rcParams['figure.figsize'] = (10,7)

df_train['Ranking'].hist(bins=100)
df_train['City'].value_counts(ascending=True).plot(kind='barh')
df_train['Ranking'][df_train['City'] =='London'].hist(bins=100)
# посмотрим на топ 10 городов

for x in (df_train['City'].value_counts())[0:10].index:

    df_train['Ranking'][df_train['City'] == x].hist(bins=100)

plt.show()
df_train['Rating'].value_counts(ascending=True).plot(kind='barh')
df_train['Ranking'][df_train['Rating'] == 5].hist(bins=100)
df_train['Ranking'][df_train['Rating'] < 4].hist(bins=100)
plt.rcParams['figure.figsize'] = (15,10)

sns.heatmap(data.drop(['sample'], axis=1).corr(),)
# количество ресторанов в зависимости от города

rest_count = data['City'].value_counts().to_dict()

data['rest_count'] = data['City'].apply(lambda x: rest_count[x])
# количество ресторанов на одного туриста

data['rest_count_on_one_tourist'] = data['rest_count'] / data['tourists']
# количество отзывов на один ресторан

data['review_on_one_rest'] = data['Number of Reviews'] / data['rest_count']
# количество отзывов в зависимости от кол-ва туристов

data['review_on_one_rest_tour'] = data['Number of Reviews'] / data['tourists']
# ранг делим на кол-во ресторанов (чем выше - тем хуже)

data['Ranking_rest_count'] = data['Ranking'] / data['rest_count']
data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
data = data.drop(['Restaurant_id','Cuisine Style', 'Reviews', 'URL_TA', 'ID_TA',

              'Reviews_date', 'Reviews_date_first', 'Reviews_date_last'], axis=1)
# Теперь выделим тестовую часть

train_data = data.query('sample == 1').drop(['sample'], axis=1)

test_data = data.query('sample == 0').drop(['sample'], axis=1)



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