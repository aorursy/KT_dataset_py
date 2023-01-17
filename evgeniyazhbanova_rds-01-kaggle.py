# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from os.path import join

import re

import ast



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



# Загружаем специальный удобный инструмент для разделения датасета:

from sklearn.model_selection import train_test_split



from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Для воспроизводимости экспериментов фиксируем RANDOM_SEED - общий параметр для генерации случайных чисел:

RANDOM_SEED = 42
# Зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

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



main_df = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
main_df.info()
main_df.sample(5)
# Проверим наличие NaN в признаке Cuisine Style и добавим новый признак CS NaN

main_df['CS NaN'] = pd.isna(main_df['Cuisine Style']).astype('uint8')

main_df.head()
# Проверим наличие NaN в признаке Price Range и добавим новый признак PR NaN

main_df['PR NaN'] = pd.isna(main_df['Price Range']).astype('uint8')

main_df.head()
# Проверим наличие NaN в признаке Number of Reviews и добавим новый признак NR NaN

main_df['NR NaN'] = pd.isna(main_df['Number of Reviews']).astype('uint8')

main_df.head()
# Проверим наличие NaN в признаке Number of Reviews и добавим новый признак NR NaN

main_df['R NaN'] = pd.isna(main_df['Reviews']).astype('uint8')

main_df.head()
# Создадим словарь, определяющий принадлежность каждого из городов к "столичному признаку"

# Сначала выведем список всех городов

main_df.City.value_counts()
# Далее создадим словарь и проставим в нем значение True, если город является столицей, и False, если нет

is_capital = {

    'London': 1,

    'Paris': 1,

    'Madrid': 1,

    'Barcelona': 0,

    'Berlin': 1,

    'Milan': 0,

    'Rome': 1,

    'Prague': 1,

    'Lisbon': 1,

    'Vienna': 1,

    'Amsterdam': 1,

    'Brussels': 1,

    'Hamburg': 0,

    'Munich': 0,

    'Lyon': 0,

    'Stockholm': 1,

    'Budapest': 1,

    'Warsaw': 1,

    'Dublin': 1,

    'Copenhagen': 1,

    'Athens': 1,

    'Edinburgh': 0,

    'Zurich': 0,

    'Oporto': 0,

    'Geneva': 0,

    'Krakow': 0,

    'Oslo': 1,

    'Helsinki': 1,

    'Bratislava': 1,

    'Luxembourg': 1,

    'Ljubljana': 1

}
# И добавим в Датафрейм новый признак Capital, который будет показывать, находится ресторан в столице или нет:

main_df['Capital'] = main_df['City'].map(is_capital)
main_df.head()
main_df.info()
# Далее создадим словарь с количеством жителей в каждом городе

is_population = {

    'London': 8908081,

    'Paris': 2190327,

    'Madrid': 3165541,

    'Barcelona': 1636762,

    'Berlin': 3574830,

    'Milan': 1378689,

    'Rome': 2875805,

    'Prague': 1301132,

    'Lisbon': 505526,

    'Vienna': 1863881,

    'Amsterdam': 857713,

    'Brussels': 179277,

    'Hamburg': 1810438,

    'Munich': 1450381,

    'Lyon': 506615,

    'Stockholm': 961609,

    'Budapest': 1757618,

    'Warsaw': 1748916,

    'Dublin': 1347359,

    'Copenhagen': 615993,

    'Athens': 664046,

    'Edinburgh': 488100,

    'Zurich': 428737,

    'Oporto': 237591,

    'Geneva': 200548,

    'Krakow': 769498,

    'Oslo': 673469,

    'Helsinki': 643272,

    'Bratislava': 425923,

    'Luxembourg': 602005,

    'Ljubljana': 284355

}
# И добавим в Датафрейм новый признак Population, определяющий количество жителей в городе, в котором расположен ресторан

main_df['Population'] = main_df['City'].map(is_population)
main_df.head()
# Создадим словарь количества туристов по городам, млн.человек в год

is_tourists = {

    'London': 20,

    'Paris': 16.1,

    'Madrid': 5.5,

    'Barcelona': 8.9,

    'Berlin': 5.1,

    'Milan': 8.4,

    'Rome': 7.3,

    'Prague': 6.4,

    'Lisbon': 7.0,

    'Vienna': 6.63,

    'Amsterdam': 8.7,

    'Brussels': 6.7,

    'Hamburg': 1.42,

    'Munich': 5.4,

    'Lyon': 3.0,

    'Stockholm': 7.5,

    'Budapest': 4.8,

    'Warsaw': 8.5,

    'Dublin': 5.59,

    'Copenhagen': 8.8,

    'Athens': 33.0,

    'Edinburgh': 13.0,

    'Zurich': 5.6,

    'Oporto': 12.8,

    'Geneva': 4.2,

    'Krakow': 13.5,

    'Oslo': 4.0,

    'Helsinki': 7.4,

    'Bratislava': 3.5,

    'Luxembourg': 3.0,

    'Ljubljana': 0.9

}
# И добавим в Датафрейм новый признак Tourists, определяющий количество туристов в городе, в котором расположен ресторан

main_df['Tourists'] = main_df['City'].map(is_tourists)
main_df.head()
# Создадим словарь соответствия города стране

is_country = {

    'London': 'UK',

    'Paris': 'France',

    'Madrid': 'Spain',

    'Barcelona': 'Spain',

    'Berlin': 'Germany',

    'Milan': 'Italy',

    'Rome': 'Italy',

    'Prague': 'Czech Republic',

    'Lisbon': 'Portugal',

    'Vienna': 'Austria',

    'Amsterdam': 'Netherlands',

    'Brussels': 'Belgium',

    'Hamburg': 'Germany',

    'Munich': 'Germany',

    'Lyon': 'France',

    'Stockholm': 'Sweden',

    'Budapest': 'Hungary',

    'Warsaw': 'Poland',

    'Dublin': 'Ireland',

    'Copenhagen': 'Denmark',

    'Athens': 'Greece',

    'Edinburgh': 'Scotland',

    'Zurich': 'Switzerland',

    'Oporto': 'Portugal',

    'Geneva': 'Switzerland',

    'Krakow': 'Poland',

    'Oslo': 'Norway',

    'Helsinki': 'Finland',

    'Bratislava': 'Slovakia',

    'Luxembourg': 'Luxembourg',

    'Ljubljana': 'Slovenia'

}
# И добавим в Датафрейм новый признак Tourists, определяющий количество туристов в городе, в котором расположен ресторан

main_df['Country'] = main_df['City'].map(is_country)
main_df.head()
# Используем готовую функцию pandas - get_dummies - для One-Hot Encoding для кодирования стран

# Поскольку словарь соответствия города стране создавался вручную и не имеет пустых значений, выставим параметр dummy_na=False 

main_df = pd.get_dummies(main_df, columns=[ 'Country',], dummy_na = False)
main_df.head()
main_df.info()
# Для удобства заменим NaN в колонке Cuisine Style на 'Universal'

main_df['Cuisine Style'] = main_df['Cuisine Style'].fillna('Universal')
main_df.sample(5)
# Преобразуем данные признака Cuisine Style из str в list

# Функция, преобразующая строковые данные из строки Cuisine Style в данные типа список

def str_to_list(x):

    y = []

    a = ['Universal']

    if x != 'Universal':

        x = str(x[2:-2]).split('\', \'')

        for i in x:

            y.append(i)

        return y

    else:

        return a



# Заменим значения в колонке Cuisine Style на полученные списки

main_df['Cuisine Style'] = main_df['Cuisine Style'].apply(str_to_list)
# Создадим словарь кухонь и посчитаем общее количество уникальных видов кухни

cuisines_dict = dict()

cuisines = main_df['Cuisine Style']

for cuisines_list in cuisines:

    for cuisine in cuisines_list:

        try:

            cuisines_dict[cuisine] += 1

        except:

            cuisines_dict[cuisine] = 1



# Выведем список кухонь и общее количество видов кухни

print('\n'.join(cuisines_dict))

print(f'Количество видов кухни: {len(cuisines_dict)}')
# Подсчитаем количество видов кухни для каждого ресторана

def cuisines_count(data):

    if type(data) == list:

        return len(data)



# Добавим признак Cousines Count, определяющий количество кухонь в каждом ресторане

main_df['Cousines Count'] = main_df['Cuisine Style'].apply(cuisines_count)
main_df.head()
main_df.info()
# Создадим новый пустой dataframe для хранения всех типов кухонь в качестве отдельных признаков

cuisines_types_df = pd.DataFrame()



# Функция, записывающая все типы кухонь в множество

cuisines_set = set()

def count_cuisines(x):

    if type(x) == list:

        for cuisine in x:

            cuisines_set.add(cuisine)

    return x



main_df['Cuisine Style'] = main_df['Cuisine Style'].apply(count_cuisines)
# Функция, заполняющая пустой dataframe новыми признаками

def columns_cuisines(data):

    if cuisine in data:

        return 1

    return 0



for cuisine in cuisines_set:

    cuisines_types_df[cuisine] = main_df['Cuisine Style'].apply(columns_cuisines)
cuisines_types_df.head()
# Заменим NaN в признаке Price Range наиболее часто встречающимся значением:

main_df['Price Range'].value_counts()
main_df['Price Range'].value_counts().index[0]
# Заменим NaN в признаке Price Range на '$$ - $$$'

main_df['Price Range'].fillna(main_df['Price Range'].value_counts().index[0], inplace = True)

main_df.head()
# Заменим значения Price Range на числовые в зависимости от категории цены:

def price_range(row):

    if row['Price Range'] == '$':

        return int(1)

    if row['Price Range'] == '$$ - $$$':

        return int(2)

    if row['Price Range'] == '$$$$':

        return int(3)



main_df['Price Range'] = main_df.apply(price_range, axis = 1)
main_df.info()
# Заменим NaN в признаке Number of Reviews на среднее арифметическое

main_df['Number of Reviews'].fillna(main_df['Number of Reviews'].median(), inplace = True)
main_df.info()
main_df.head()
# Приведем Restaurant_id к int

def rid_change(t):

    if t == '#error':

        return ''

    else:

        return int(t[3:])

main_df['Restaurant_id'] = main_df.Restaurant_id.apply(rid_change)
# Приведем ID_TA к int

main_df['ID_TA'] = main_df['ID_TA'].apply(lambda id_ta: int(id_ta[1:]))
main_df.info()
# Склеим основную таблицу и таблицу по видам кухонь, чтобы проверить что у нас получилось.

# Этот прием пригодится в дальнейшем для обработки всех признаков

train_df = pd.concat([main_df, cuisines_types_df], axis=1)

train_df.head(5)
train_df.info()
# Убираем все не числовые признаки из датафрейма

train_df = train_df.drop(columns=['City', 'Cuisine Style','Reviews','URL_TA'])
train_df.info()
#def sample_col(data):

#    if data == 0:

#        return 0

#    else:

#        return 1

    

#train_df['Sample'] = train_df['Rating'].apply(sample_col)
train_df
# Теперь выделим тестовую часть

train_data = train_df.query('sample == 1').drop(['sample'], axis=1)

test_data = train_df.query('sample == 0').drop(['sample'], axis=1)



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
y_pred
# Функция, округляющая результаты предсказанных рейтингов

def final_rating(rating_pred):

    if rating_pred <= 0.25:

        return 0.0

    if rating_pred <= 0.75:

        return 0.5

    if rating_pred <= 1.25:

        return 1.0

    if rating_pred <= 1.75:

        return 1.5

    if rating_pred <= 2.25:

        return 2.0

    if rating_pred <= 2.75:

        return 2.5

    if rating_pred <= 3.25:

        return 3.0

    if rating_pred <= 3.75:

        return 3.5

    if rating_pred <= 4.25:

        return 4.0

    if rating_pred <= 4.75:

        return 4.5

    return 5.0
# Применим функцию к результату

for i in range(len(y_pred)):

    y_pred[i] = final_rating(y_pred[i])
y_pred
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# в RandomForestRegressor есть возможность вывести самые важные признаки для модели

plt.rcParams['figure.figsize'] = (10,10)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(15).plot(kind='barh')
test_data
test_data = test_data.drop(['Rating'], axis=1)
sample_submission
predict_submission = model.predict(test_data)
predict_submission
for i in range(len(predict_submission)):

    predict_submission[i] = final_rating(predict_submission[i])
final_submission = pd.DataFrame()
final_submission['Restaurant_id'] = df_test['Restaurant_id']

final_submission['Rating'] = predict_submission
final_submission
final_submission.to_csv('submission.csv', index=False)

final_submission
final_submission.info()