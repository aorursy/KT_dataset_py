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



# Загружаем библиотеки для обработки дат

import re

from datetime import datetime, timedelta

import time
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
'''# Для примера я возьму столбец Number of Reviews

data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')'''
'''data['Number_of_Reviews_isNAN']'''
'''# Далее заполняем пропуски 0, вы можете попробовать заполнением средним или средним по городу и тд...

data['Number of Reviews'].fillna(0, inplace=True)'''
# Функция заполняет пропуски в столбце 'Number of Reviews' средним значением по городу

def fillna_number_reviews(row):

    if np.isnan(row['Number of Reviews']):

        row['Number of Reviews'] = data[data['City'] == row['City']]['Number of Reviews'].mean()

        return row['Number of Reviews']

    return row['Number of Reviews']
# Применяем функция для заполнения пропусков

data['Number of Reviews'] = data.apply(lambda row: fillna_number_reviews(row), axis=1)
# Заменяем пропуски в 'Cuisine Style' на значение 'Not data'

data['Cuisine Style'].fillna("['Not data']", axis=0, inplace=True)
# Заменяем пропуски в 'Price Range' на значение 0

data['Price Range'].fillna(0, axis=0, inplace=True)
# Заменяем пропуски в 'Reviews' на значение 'Not data'

data['Reviews'].fillna("['Not data']", axis=0, inplace=True)
data.nunique(dropna=False)
'''# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)'''
'''data.sample(5)'''
data['Price Range'].value_counts()
# Преобразовыем категориальные данные в числовой формат

data['Price Range'] = data['Price Range'].map({'$$ - $$$': 2, '$': 1, '$$$$':3}) 
# тут ваш код на обработку других признаков

# .....
# Пребразование строки в список стилей кухонь, представленных в ресторанах

data['Cuisine Style'] = data['Cuisine Style'].apply(lambda x: x[2:-2])

data['Cuisine Style'] = data['Cuisine Style'].str.split("', '")
# Создаем новый признак с количество стилей кухонь, представленных в ресторанах

data['Many cuisins'] = data['Cuisine Style'].apply(lambda x: len(x))
# Создаем множество стилей кухонь

cuisins = set()

for cuisins_style in data['Cuisine Style']:

    for cuisin in cuisins_style:

        cuisins.add(cuisin)



len(cuisins) #кол-во стилей кухонь, +1 - нет данных
# Создаем функцию для преобразования в dummy-переменные

def find_item(cell):

    if item in cell:

        return 1

    return 0
# Применяем функцию для преобразования 'Cuisine Style' в dummy-переменные

for item in cuisins:

    data[item] = data['Cuisine Style'].apply(find_item)
# Выделяем значения дат отзывов в остельный столбец

data['Date reviews'] = data['Reviews'].apply(lambda x: re.findall(r'\d{2}/\d{2}/\d{4}', x)) 
# Преобразовываем строки с информацией о датах отзывов в список со значениями в формате datetime

for i in range(len(data['Date reviews'])):

    data['Date reviews'][i] = [datetime.strptime(x, '%m/%d/%Y') for x in data['Date reviews'][i]] 
# Для удобства вычисления преобразуем даты в формат "кол-во дней с 1970 года"

for i in range(len(data['Date reviews'])):

    data['Date reviews'][i] = list(map(datetime.timestamp, data['Date reviews'][i]))
# Добавляем признак даты последнего отзыва, в формате "кол-во дней с 1970 года"

data['Date last review'] = data.apply(lambda x: max(x['Date reviews']) if len(x['Date reviews']) > 0 else 0, axis=1)
data['City'].value_counts()
# Создаем новый признак с id города, на основании телефонного кода

data['City ID'] = data['City'].map({'London': 4420, 'Paris': 331, 'Madrid':341, 'Barcelona':343,

                                   'Berlin':4930, 'Milan':392, 'Rome':396, 'Prague':4202,

                                   'Lisbon':35121, 'Vienna':431, 'Amsterdam':3120, 'Brussels':322,

                                   'Hamburg':4940, 'Munich':4989, 'Lyon':33437, 'Stockholm':468,

                                   'Budapest':361, 'Warsaw':4822, 'Dublin':3531, 'Copenhagen':451,

                                   'Athens':30210, 'Edinburgh':44131, 'Zurich':411, 'Oporto':3512,

                                   'Geneva':4122, 'Krakow':4812, 'Oslo':4722, 'Helsinki':3589,

                                   'Bratislava':4212, 'Luxembourg':352, 'Ljubljana':3861}) 
# Создаем новый признак с населением города, млн.чел.

data['City population'] = data['City'].map({'London': 8.982, 'Paris': 2.140, 'Madrid':6.642, 'Barcelona':5.575,

                                           'Berlin':3.769, 'Milan':1.332, 'Rome':2.870, 'Prague':1.309,

                                           'Lisbon':0.507, 'Vienna':1.897, 'Amsterdam':0.825, 'Brussels':1.209,

                                           'Hamburg':1.841, 'Munich':1.472, 'Lyon':0.496, 'Stockholm':0.976,

                                           'Budapest':1.752, 'Warsaw':1.791, 'Dublin':1.388, 'Copenhagen':1.247,

                                           'Athens':3.169, 'Edinburgh':0.525, 'Zurich':0.402, 'Oporto':0.222,

                                           'Geneva':0.499, 'Krakow':0.779, 'Oslo':0.681, 'Helsinki':0.655,

                                           'Bratislava':0.438, 'Luxembourg':0.122, 'Ljubljana':0.284}) 
# Создаем новый с площадью города, км.кв.

data['City square'] = data['City'].map({'London': 1572, 'Paris': 105.4, 'Madrid':604.3, 'Barcelona':101.9,

                                       'Berlin':891.8, 'Milan':181.8, 'Rome':1285, 'Prague':496,

                                       'Lisbon':100, 'Vienna':414.6, 'Amsterdam':219.3, 'Brussels':32.61,

                                       'Hamburg':755.2, 'Munich':310.4, 'Lyon':47.87, 'Stockholm':188,

                                       'Budapest':525.2, 'Warsaw':517.2, 'Dublin':115, 'Copenhagen':88.25,

                                       'Athens':2929, 'Edinburgh':264, 'Zurich':87.88, 'Oporto':41.42,

                                       'Geneva':15.93, 'Krakow':327, 'Oslo':454, 'Helsinki':213.8,

                                       'Bratislava':367.6, 'Luxembourg':51.46, 'Ljubljana':163.8}) 
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
# на всякий случай, заново подгружаем данные

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем

data.info()
def preproc_data(df_input):

    '''includes several functions to pre-process the predictor data.'''

    

    df_output = df_input.copy()

    

    # ################### 1. Предобработка ############################################################## 

    # убираем не нужные для модели признаки

    df_output.drop(['Restaurant_id','ID_TA',], axis = 1, inplace=True)

    

    

    # ################### 2. NAN ############################################################## 

    # Функция заполняет пропуски в столбце 'Number of Reviews' средним значением по городу

    def fillna_number_reviews(row):

        if np.isnan(row['Number of Reviews']):

            row['Number of Reviews'] = round(df_output[df_output['City'] == row['City']]['Number of Reviews'].mean(),0)

            return row['Number of Reviews']

        return row['Number of Reviews']

    

    # Применяем функция для заполнения пропусков

    df_output['Number of Reviews'] = df_output.apply(lambda row: fillna_number_reviews(row), axis=1)

    

    # Заменяем пропуски в 'Cuisine Style' на значение 'Not data'

    df_output['Cuisine Style'].fillna("['Not data']", axis=0, inplace=True)

    

    # Заменяем пропуски в 'Price Range' на значение 0

    df_output['Price Range'].fillna('0', axis=0, inplace=True)

    

    # Заменяем пропуски в 'Reviews' на значение 'Not data'

    df_output['Reviews'].fillna("['Not data']", axis=0, inplace=True)

    

    

    # ################### 3. Encoding and 4. Feature Engineering ############################################################## 

    # Преобразовыем категориальные данные в числовой формат

    df_output['Price Range'] = df_output['Price Range'].map({'$$ - $$$': 2, '$': 1, '$$$$':3, '0':0}) 

    

    # Пребразование строки в список стилей кухонь, представленных в ресторанах

    df_output['Cuisine Style'] = df_output['Cuisine Style'].apply(lambda x: x[2:-2])

    df_output['Cuisine Style'] = df_output['Cuisine Style'].str.split("', '")

    

    # Создаем новый признак с количество стилей кухонь, представленных в ресторанах

    df_output['Many cuisins'] = df_output['Cuisine Style'].apply(lambda x: len(x))

    

    # Создаем множество стилей кухонь

    cuisins = set()

    for cuisins_style in df_output['Cuisine Style']:

        for cuisin in cuisins_style:

            cuisins.add(cuisin)

    

    # Создаем функцию для преобразования в dummy-переменные

    def find_item(cell):

        if item in cell:

            return 1

        return 0

    

    # Применяем функцию для преобразования 'Cuisine Style' в dummy-переменные

    for item in cuisins:

        df_output[item] = df_output['Cuisine Style'].apply(find_item)

        

    # Удаляем столбцы с наименьшей информативностью

    columns_cuisins_drop = [s for s in cuisins if df_output[s].sum() < 50]

    df_output.drop(columns_cuisins_drop, axis = 1, inplace=True)

            

    # Выделяем значения дат отзывов в остельный столбец

    df_output['Date reviews'] = df_output['Reviews'].apply(lambda x: re.findall(r'\d{2}/\d{2}/\d{4}', x)) 

    

    # Преобразовываем строки с информацией о датах отзывов в список со значениями в формате datetime

    for i in range(len(df_output['Date reviews'])):

        df_output['Date reviews'][i] = [datetime.strptime(x, '%m/%d/%Y') for x in df_output['Date reviews'][i]]

    

    # Для удобства вычисления преобразуем даты в формат "кол-во дней с 1970 года"

    for i in range(len(df_output['Date reviews'])):

        df_output['Date reviews'][i] = list(map(datetime.timestamp, df_output['Date reviews'][i]))

    

    # Добавляем признак даты последнего отзыва, в формате "кол-во дней с 1970 года"

    df_output['Date last review'] = df_output.apply(lambda x: max(x['Date reviews']) if len(x['Date reviews']) > 0 else 0, axis=1)

    

    # Создаем новый признак с id города, на основании телефонного кода

    df_output['City ID'] = df_output['City'].map({'London': 4420, 'Paris': 331, 'Madrid':341, 'Barcelona':343,

                                                 'Berlin':4930, 'Milan':392, 'Rome':396, 'Prague':4202,

                                                 'Lisbon':35121, 'Vienna':431, 'Amsterdam':3120, 'Brussels':322,

                                                 'Hamburg':4940, 'Munich':4989, 'Lyon':33437, 'Stockholm':468,

                                                 'Budapest':361, 'Warsaw':4822, 'Dublin':3531, 'Copenhagen':451,

                                                 'Athens':30210, 'Edinburgh':44131, 'Zurich':411, 'Oporto':3512,

                                                 'Geneva':4122, 'Krakow':4812, 'Oslo':4722, 'Helsinki':3589,

                                                 'Bratislava':4212, 'Luxembourg':352, 'Ljubljana':3861})

    

    # Создаем новый признак с населением города, млн.чел.

    df_output['City population'] = df_output['City'].map({'London': 8.982, 'Paris': 2.140, 'Madrid':6.642, 'Barcelona':5.575,

                                                         'Berlin':3.769, 'Milan':1.332, 'Rome':2.870, 'Prague':1.309,

                                                         'Lisbon':0.507, 'Vienna':1.897, 'Amsterdam':0.825, 'Brussels':1.209,

                                                         'Hamburg':1.841, 'Munich':1.472, 'Lyon':0.496, 'Stockholm':0.976,

                                                         'Budapest':1.752, 'Warsaw':1.791, 'Dublin':1.388, 'Copenhagen':1.247,

                                                         'Athens':3.169, 'Edinburgh':0.525, 'Zurich':0.402, 'Oporto':0.222,

                                                         'Geneva':0.499, 'Krakow':0.779, 'Oslo':0.681, 'Helsinki':0.655,

                                                         'Bratislava':0.438, 'Luxembourg':0.122, 'Ljubljana':0.284}) 

    

    # Создаем новый с площадью города, км.кв.

    df_output['City square'] = df_output['City'].map({'London': 1572, 'Paris': 105.4, 'Madrid':604.3, 'Barcelona':101.9,

                                                     'Berlin':891.8, 'Milan':181.8, 'Rome':1285, 'Prague':496,

                                                     'Lisbon':100, 'Vienna':414.6, 'Amsterdam':219.3, 'Brussels':32.61,

                                                     'Hamburg':755.2, 'Munich':310.4, 'Lyon':47.87, 'Stockholm':188,

                                                     'Budapest':525.2, 'Warsaw':517.2, 'Dublin':115, 'Copenhagen':88.25,

                                                     'Athens':2929, 'Edinburgh':264, 'Zurich':87.88, 'Oporto':41.42,

                                                     'Geneva':15.93, 'Krakow':327, 'Oslo':454, 'Helsinki':213.8,

                                                     'Bratislava':367.6, 'Luxembourg':51.46, 'Ljubljana':163.8}) 

    

    

    # ################### 5. Clean #################################################### 

    # убираем признаки которые еще не успели обработать, 

    # модель на признаках с dtypes "object" обучаться не будет, просто выберим их и удалим

    object_columns = [s for s in df_output.columns if df_output[s].dtypes == 'object']

    df_output.drop(object_columns, axis = 1, inplace=True)

    

    return df_output
df_preproc = preproc_data(data)

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

y_pred = np.round(model.predict(X_test), 1)
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
predict_submission = np.round(model.predict(test_data), 1)
predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)