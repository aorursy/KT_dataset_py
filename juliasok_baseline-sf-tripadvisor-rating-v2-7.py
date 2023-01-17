# Импортируем необходимые библиотеки:

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели



import re

import datetime



import matplotlib.pyplot as plt

%matplotlib inline



# Загружаем специальный удобный инструмент для разделения датасета:

from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Фиксируем RANDOM_SEED, чтобы эксперименты были воспроизводимы!

RANDOM_SEED = 42



# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

!pip freeze > requirements.txt
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')



# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



df = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
# Оцифровка ценового сегмента ресторана

def number_price(price):

    number = 0

    if price == '$':

        number = 1

    elif price == '$$ - $$$':

        number = 2

    elif price == '$$$$':

        number = 3

    return number



df['NumberPrice'] = df['Price Range'].apply(number_price)
# Пропуск в столбце Cuisine Style заполнили кухней Standart

df['Cuisine Style'].fillna("['Standart']", inplace=True)



# Функция подсчитывает количество представленных в ресторане кухонь

def count_cuosine_stile(s):

    count = s.count(',') + 1

    return count



# Добавляем числовой признак 

# CountCuisine - количество представленных в ресторане кухонь

df['CountCuisine'] = df['Cuisine Style'].apply(count_cuosine_stile)
# Сколько типов кухонь представлено в наборе данных

cuisines = df['Cuisine Style']

cuisines = cuisines.str[2:-2].str.split("', '")

cuisines

cuis_all = []

for cuisine  in cuisines:

    for item in cuisine:

        cuis_all.append(item)

    

unique_cuisines = pd.Series(cuis_all).value_counts()
BestClassCuisine = unique_cuisines[:5:]     # Выделяем кухни первого класса



# Функция определяет наличие в меню ресторана меню кухони первого класса

def is_best_cuisine(row):

    is_best = 0

    for item in row:

        if item in BestClassCuisine:

            is_best = 1

    return is_best



# Добавляем числовой признак 

# isBestCuisine - наличие в меню ресторана меню кухони первого класса

df['isBestCuisine'] = cuisines.apply(is_best_cuisine)
MidleClassCuisine = unique_cuisines[5:10:]  # Выделяем кухни второго класса



# Функция определяет наличие в меню ресторана меню кухони второго класса

def is_midle_cuisine(row):

    is_midle = 0

    for item in row:

        if item in MidleClassCuisine:

            is_midle = 1

    return is_midle



# Добавляем числовой признак 

# isMidleCuisine - наличие в меню ресторана меню кухони второго класса

df['isMidleCuisine'] = cuisines.apply(is_midle_cuisine)
LowerClassCuisine = unique_cuisines[10::]   # Выделяем кухни третьего класса



# Функция определяет наличие в меню ресторана меню кухони третьего класса

def is_lower_cuisine(row):

    is_lower = 0

    for item in row:

        if item in LowerClassCuisine:

            is_lower = 1

    return is_lower



# Добавляем числовой признак 

# isLowerCuisine - наличие в меню ресторана меню кухони третьего класса

df['isLowerCuisine'] = cuisines.apply(is_lower_cuisine)
# Оцифровка кода города

list_city=list(df.City.unique())

dict_city={}

for i in range(len(list_city)):

    dict_city[list_city[i]] = i



df['codeCity']=df['City'].apply(lambda x: dict_city[x])
# Словарь с информацией о количестве жителей в каждом городе

# и информацией о том, столица это или нет

dict_population = {

    'Amsterdam': [821_752, 1],

    'Athens': [664_046, 1],

    'Barcelona': [5_515_000, 0],

    'Berlin': [3_748_000, 1],

    'Bratislava': [424_428, 1],

    'Brussels': [174_383, 1],

    'Budapest': [1_752_000, 1],

    'Copenhagen': [602_481, 1],

    'Dublin': [1_361_000, 1],

    'Edinburgh': [482_005, 1],

    'Geneva': [495_249, 0],

    'Hamburg': [1_822_000, 0],

    'Helsinki': [631_695, 1],

    'Krakow': [769_489, 0],

    'Lisbon': [504_718, 1],

    'Ljubljana': [279_631, 1],

    'London': [8_900_000, 1],

    'Luxembourg': [602_005, 1],

    'Lyon': [513_275, 0],

    'Madrid': [6_550_000, 1],

    'Milan': [1_352, 0],

    'Munich': [1_456_000, 0],

    'Oporto': [214_349, 0],

    'Oslo': [673_469, 1],

    'Paris': [2_148_000, 1],

    'Prague': [1_319_000, 1],

    'Rome': [2_873_000, 1],

    'Stockholm': [974_073, 1],

    'Vienna': [1_889_000, 1],

    'Warsaw': [1_708_000, 1],

    'Zurich': [402_762, 0]

}



# Добавляем числовой признак 

# Capital - является ли город столицей

df['Capital'] = df['City'].apply(lambda x: dict_population[x][1])



# Добавляем числовой признак 

# Population - население города

df['Population'] = df['City'].apply(lambda x: dict_population[x][0])



# Словарь с информацией о количестве ресторанов в каждом городе

dict_count_restaurants_City = {}

sum = 0

for item in df['City'].unique():

    dict_count_restaurants_City [item] = df['City'][df['City'] == item].count()

    sum += dict_count_restaurants_City [item] 



# Добавляем числовой признак 

# CountRestaurantsInCity - количество ресторанов в городе

df['CountRestaurantsInCity'] = df['City'].apply(lambda x: dict_count_restaurants_City[x])



# Добавляем числовой признак 

# DensityOnRestaurant - количество жителей на одит ресторан

df['DensityOnRestaurant'] = df['Population'] / df['CountRestaurantsInCity']
pattern = "\d\d/\d\d/\d\d\d\d"

time = []

for time_str in df['Reviews']:

    time.append(re.findall(pattern, str(time_str)))  
rev=pd.DataFrame(time)

rev.columns=['date1', 'date2']

rev['date1'] = pd.to_datetime(rev['date1']) 

rev['date2'] = pd.to_datetime(rev['date2'])

                                  

rev['delta'] =abs(rev['date2'] -  rev['date1']) 

rev['delta'] = rev['delta'].apply(lambda x: x.days)



rev['date1'] = datetime.datetime.now() - rev['date1']

rev['date1'] = rev['date1'].apply(lambda x: x.days)

rev['date2'] = datetime.datetime.now() - rev['date2']

rev['date2'] = rev['date2'].apply(lambda x: x.days)



rev['delta'].fillna(rev['delta'].mean(), inplace=True)

rev['date1'].fillna(rev['date1'].mean(), inplace=True)

rev['date2'].fillna(rev['date2'].mean(), inplace=True)



df['delta'] = rev['delta'].apply(lambda x: int(x))

df['date1'] = rev['date1'].apply(lambda x: int(x))

df['date2'] = rev['date2'].apply(lambda x: int(x))
reviews = df['Reviews'].str[2:-2]

reviews = reviews.apply(lambda x: str(x).replace("',", '/~'))

reviews = reviews.apply(lambda x: str(x).replace('],', '/~'))

reviews = reviews.apply(lambda x: str(x).replace('[', ''))

reviews = reviews.apply(lambda x: str(x).replace("'", ''))

reviews = reviews.apply(lambda x: x if str(x)!='/~ ' else str(x).replace('/~ ', ''))

reviews = reviews.str.split("/~")

df['Reviews']=reviews
# Функция определяет количество отзывов о ресторане

def count_reviews(row):

    if pd.isnull(row['Number of Reviews']) or row['Number of Reviews'] == 0:

        if len(row['Reviews']) == 2:

            count = 1

        elif len(row['Reviews']) > 2:

            count = 2

        else:

            count = 0

    else:

        count = int(row['Number of Reviews']) 

    return count



# Заполним пропуски в столбце Number of Reviews

df['Number of Reviews'] = df.apply(count_reviews, axis = 1) 
# убираем не нужные для модели признаки

df.drop(['Restaurant_id', 'City', 'Cuisine Style','Price Range', 'Reviews', 'URL_TA', 'ID_TA'], axis='columns', inplace=True)
# Теперь выделим тестовую часть

train_data = df.query('sample == 1').drop(['sample'], axis=1)

test_data = df.query('sample == 0').drop(['sample'], axis=1)



y = train_data.Rating.values            # наш таргет

X = train_data.drop(['Rating'], axis=1)
# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных

# выделим 20% данных на валидацию (параметр test_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
# проверяем

test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape
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
test_data = test_data.drop(['Rating'], axis=1)
predict_submission = model.predict(test_data)
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)