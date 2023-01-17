# Импортируем все необходимое

import ast

import matplotlib.pyplot as plt

from os.path import join

import pandas as pd

from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Сделаем так, чтобы matplotlib рисовал все сразу без напоминаний

%matplotlib inline
# Для воспроизводимости результатов зададим:

# - общий параметр для генерации случайных чисел

RANDOM_SEED = 42

# - общую текущую дату

CURRENT_DATE = pd.to_datetime('22/02/2020')
# Для тех, кто будет запускать мой ноутбук на своей машине создаем requirements.txt

!pip freeze > requirements.txt
# Зададим путь к папке с данными

ds_folder = '/kaggle/input/sf-dst-restaurant-rating/'



# Зададим названия файлов датасетов

mt_ds = 'main_task.csv'

kt_ds = 'kaggle_task.csv'



# Сформируем пути к датасетам

mt_path = join(ds_folder, mt_ds)

kt_path = join(ds_folder, kt_ds)



# Загрузим имеющиеся датасеты

mt_df = pd.read_csv(mt_path)

kt_df = pd.read_csv(kt_path)
# Проанализируем данные из main_task.csv

mt_df.head(1)
mt_df.info()
mt_df.nunique()
# Сформируем множество дублирующихся идентификаторов ресторанов на сайте TripAdvisor

duplicated_ids = set(mt_df.groupby('ID_TA')['ID_TA'].count().sort_values(ascending=False)[:20].index)



# Смотрим на непосредственно данные, относящиеся к этим дубликатам

mt_df[mt_df['ID_TA'].apply(lambda x: x in duplicated_ids)].sort_values(by='ID_TA')
# mt_df.drop_duplicates('ID_TA', inplace=True) # По-умолчанию оставляем первые из дублирующихся строк
# Проанализируем данные из kaggle_task.csv

kt_df.head(1)
kt_df.info()
# Как видим, в датасете kaggle_task.csv отсутствует признак Rating,

# но добавлен признак с названием ресторана 'Name'
# Добавим признак 'Main' для отделения основной выборки от валидационной

mt_df['Main'] = True

kt_df['Main'] = False



# Недостающие данные по названиям ресторанов в основной выборке заполним названиями вида Name-id, где id - идентификатор ресторана

mt_df['Name'] = mt_df['Restaurant_id'].apply(lambda x: 'Name-'+x)



# Недостающие данные по рейтингу в тестовой выборке заполним нулями

kt_df['Rating'] = 0



# Объединим датасеты в один для полного анализа по всем данным

df = pd.concat([mt_df, kt_df])
df.info()
# Создадим справочник с указанием количества ресторанов для каждого города, присутствующего в датасете

res_count = {

    'Paris': 17593,

    'Stockholm': 3131,

    'London': 22366,

    'Berlin': 8110, 

    'Munich': 3367,

    'Oporto': 2060, 

    'Milan': 7940,

    'Bratislava': 1331,

    'Vienna': 4387, 

    'Rome': 12086,

    'Barcelona': 10086,

    'Madrid': 11562,

    'Dublin': 2706,

    'Brussels': 3703,

    'Zurich': 1901,

    'Warsaw': 3210,

    'Budapest': 3445, 

    'Copenhagen': 2637,

    'Amsterdam': 4189,

    'Lyon': 2833,

    'Hamburg': 3501, 

    'Lisbon': 4985,

    'Prague': 5850,

    'Oslo': 1441, 

    'Helsinki': 1661,

    'Edinburgh': 2248,

    'Geneva': 1753,

    'Ljubljana': 647,

    'Athens': 2814,

    'Luxembourg': 759,

    'Krakow': 1832       

}
# Создадим новый признак 'Restaurants Count', отражающий общее количество ресторанов в городе, в котором расположен данный ресторан

df['Restaurants Count'] = df['City'].map(res_count)
df.info()
df.head()
# А теперь создадим относительный признак 'Relative Ranking' = 'Ranking' / 'Restaurants Count'

df['Relative Ranking'] = df['Ranking'] / df['Restaurants Count']
df.info()
df.head()
# Посмотрим на данные в признаке 'Price Range'

df['Price Range'].unique()
# Следуя рекомендациям более опытных коллег, до заполнения отсутствующих данных

# создадим бинарный признак отсутствия данных 'Price Range Was NAN'

df['Price Range Was NAN'] = df['Price Range'].isna()
df.info()
df.head()
df['Number of Reviews'].value_counts()
# Как видно, кроме незаполненных данных есть три типа диапазонов цен '$', '$$-$$$' и '$$$$'.

# Посмотрим на гистограмму количества отзывов для тех ресторанов, по которым отсутствует

# информация о диапазоне цен.

plt.xlabel('Number of Reviews')

plt.ylabel('Count')

plt.title('Number of Reviews distribution for restaurants that have no Price Range data.')

df[df['Price Range'].isna()]['Number of Reviews'].hist(bins=100, range=(0, 99))
# Посмотрим на гистаграмму распределения количества отзывов 

# о ресторанах для разных диапазонов цен.

plt.xlabel('Number of Reviews')

plt.ylabel('Count')

plt.title('$')

df[df['Price Range'] == '$']['Number of Reviews'].hist(bins=100, range=(0, 300))
plt.xlabel('Number of Reviews')

plt.ylabel('Count')

plt.title('\$\$ - \$\$\$')

df[df['Price Range'] == '$$ - $$$']['Number of Reviews'].hist(bins=100, range=(0, 300))
plt.xlabel('Number of Reviews')

plt.ylabel('Count')

plt.title('\$\$\$\$')

df[df['Price Range'] == '$$$$']['Number of Reviews'].hist(bins=100, range=(0, 300))
# Для начала создадим бинарный признак о том, было ли пропущено значение в поле 'Number of Reviews

df['Number of Reviews Was NAN'] = df['Number of Reviews'].isna()

# Заполним недостающие данные по количеству отзывов единицами

df['Number of Reviews'].fillna(1, inplace=True)
# Зададим функцию, возвращающую диапазон цен в зависимости от количества отзывов

# def get_price_range(reviews_count):

    

#     if reviews_count <= 5:

#         return '$'



#     return '$$ - $$$'
# Перед заполнением пропусков создадим бинарный признак того, что раньше здесь был NAN

df['Price Range Was NAN'] = df['Price Range'].isna()

# Заполним пропуски данными функции get_price_range

# df['Price Range'].fillna(df['Number of Reviews'].apply(get_price_range), inplace=True)

# Ну уж нет. Заполним тупо значением '$$ - $$$' через fillna()

df['Price Range'].fillna('$$ - $$$', inplace=True)
# Теперь самое время перевести признак Price Range в числовой формат.

# Так как число вариаций диапазонов цен не велико сэкономим память и вычисления

# переведя строки в числа при помощи словаря и метода map вместо использования

# sklearn.preprocessing.LabelEcnoder



price_range_dict = {

    '$': 1,

    '$$ - $$$': 100,

    '$$$$': 1000

}



df['Price Range'] = df['Price Range'].map(price_range_dict)
# Еще раз посмотрим на информацию о данных в df

df.info()
# Как видим, информация о городах присутствует для всех ресторанов

# Посмотрим в скольких разных городах расположены рестораны, данные по которым присутствуют в датасете

df['City'].nunique()
# Посмотрим, как рестораны распределены по городам

df.groupby('City')['City'].count().sort_values(ascending=False)
# Переведем информацию о городах в числовой формат

# с применением sklearn.preprocessign.LabelEncoder

cities_le = LabelEncoder()

cities_le.fit(df['City'])

df['City Code'] = cities_le.transform(df['City'])
df.head()
# Создадим словарь, в котором ключами буду названия городов, а значениями True, если этот город столица, в противном случае False

is_capital = {

    'London': True,

    'Paris': True,

    'Madrid': True,

    'Barcelona': False,

    'Berlin': True,

    'Milan': False,

    'Rome': True,

    'Prague': True,

    'Lisbon': True,

    'Vienna': True,

    'Amsterdam': True,

    'Brussels': True,

    'Hamburg': False,

    'Munich': False,

    'Lyon': False,

    'Stockholm': True,

    'Budapest': True,

    'Warsaw': True,

    'Dublin': True,

    'Copenhagen': True,

    'Athens': True,

    'Edinburgh': True,

    'Zurich': True,

    'Oporto': False,

    'Geneva': True,

    'Krakow': True,

    'Oslo': True,

    'Helsinki': True,

    'Bratislava': True,

    'Luxembourg': True,

    'Ljubljana': True

}
# Создадим числовой признак, является ли город столицей

df['Is Capital'] = df['City'].map(is_capital)
city_population = {

    'London': 8173900,

    'Paris': 2240621,

    'Madrid': 3155360,

    'Barcelona': 1593075,

    'Berlin': 3326002,

    'Milan': 1331586,

    'Rome': 2870493,

    'Prague': 1272690,

    'Lisbon': 547733,

    'Vienna': 1765649,

    'Amsterdam': 825080,

    'Brussels': 144784,

    'Hamburg': 1718187,

    'Munich': 1364920,

    'Lyon': 496343,

    'Stockholm': 1981263,

    'Budapest': 1744665,

    'Warsaw': 1720398,

    'Dublin': 506211 ,

    'Copenhagen': 1246611,

    'Athens': 3168846,

    'Edinburgh': 476100,

    'Zurich': 402275,

    'Oporto': 221800,

    'Geneva': 196150,

    'Krakow': 756183,

    'Oslo': 673469,

    'Helsinki': 574579,

    'Bratislava': 413192,

    'Luxembourg': 576249,

    'Ljubljana': 277554

}



city_country = {

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

    'Dublin': 'Ireland' ,

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
# Применим эти справочники к данным в столбце 'City' датафрейма df используя метод map

df['Population'] = df['City'].map(city_population)

df['Country'] = df['City'].map(city_country)
# Посмотрим, что получилось

df.info()
df.head()
# Проверим данные еще раз

df.info()
# Переведем информацию о странах в числовой формат

# с применением sklearn.preprocessign.LabelEncoder

countries_le = LabelEncoder()

countries_le.fit(df['Country'])

df['Country Code'] = countries_le.transform(df['Country'])
# Добавим числовой признак 'People Per Restaurant' = 'Population' / 'Restaurants Count'

df['People Per Restaurant'] = df['Population'] / df['Restaurants Count']
df.info()
df.head()
# Проверим данные

df.info()
# Получим Series столбца 'Cuisine Style' без пропусков данных

cuisines = df['Cuisine Style'].dropna()

cuisines
type(cuisines.iloc[0])
# Создадим признак 'Cusine Style Was NAN', показывающий, что в столбце 'Cuisine Style' данные отсутствовали

df['Cuisine Style Was NAN'] = df['Cuisine Style'].isna()

# Заполним недостающие данные значением ['Usual']

df['Cuisine Style'].fillna("['Usual']", inplace=True)
df.info()
# Применим библиотеку ast для перевода строковых представлений списков в соответствии с ситнтаксисом языка Python в списки Python

def get_list(list_string):

    result_list = ast.literal_eval(list_string)

    return result_list



# Преобразуем данные в столбце 'Cuisine Style' к списку

cuisines = cuisines.apply(get_list)



# Создадим словарь кухонь

cuisines_dict = dict()



for cuisines_list in cuisines:

    for cuisine in cuisines_list:

        try:

            cuisines_dict[cuisine] += 1

        except:

            cuisines_dict[cuisine] = 1



# Выведем количество различных кухонь



print(f'Множество различных кухонь: {len(cuisines_dict)}')
# Создадим еще один числовой признак 'cuisines_count'

def get_cuisines_count(cuisines):

    if type(cuisines) == str:

        return len(get_list(cuisines))

    return 1



df['Cuisines Count'] = df['Cuisine Style'].apply(get_cuisines_count)

df.info()
df.head()
uno_cuisine_count = 0

unique_cuisines = set()

for cuisine, count in cuisines_dict.items():

    if count == 1:

        unique_cuisines.add(cuisine)

        uno_cuisine_count += 1

print(f'Количество типов кухонь, предлагаемых только в одном ресторане: {uno_cuisine_count}')

print(f'Уникальные кухни:')

unique_cuisines
# На основе этих данных создадим столбец с числовыми данными is_unique_cuisine, указывающими на уникальную кухню данного ресторана

def is_unique_cuisine(cuisine):

    cuisines_list = get_list(cuisine)

    cuisines_set = set(cuisines_list)

    return not cuisines_set.isdisjoint(unique_cuisines)



df['Unique Cuisine'] = df['Cuisine Style'].apply(is_unique_cuisine)
df.info()
df.head()
# Используем sklearn.preprocessing.MultiLabelBinarizer кодирование для признака Cuisine Style

mlb = MultiLabelBinarizer()

encoded = pd.DataFrame(mlb.fit_transform(df['Cuisine Style'].apply(get_list)),

                       columns=mlb.classes_, index=df.index)
#  Посмотрим, что получилось

encoded.info()
encoded.head()
# Сначала заменим nan в строковых представлениях списков отзывов и их дат на строку с датой по-умолчанию

# а также заменим пустой список вида [] на список с незаполненными элементами вида ['01/01/2000', '01/01/2000']

def nan_to_default_date(list_string):

    try:

        list_string = list_string.replace('[nan', "['01/01/2000'")

        list_string = list_string.replace('nan]', "'01/01/2000']")

        list_string = list_string.replace('[]', "['01/01/2000', '01/01/2000']")

    except:

        list_string = "[['None Review', 'None Review'], ['01/01/2000', '01/01/2000']]"

    return list_string



df['Reviews'] = df['Reviews'].apply(nan_to_default_date)
# Преобразуем данные в столбце 'Reviews' к типу list

last_review = []

last_review_date = []

prelast_review = []

prelast_review_date = []

for reviews in df['Reviews']:

    reviews_list = get_list(reviews)

    if len(reviews_list) == 2:

        if (len(reviews_list[0]) == 2 and len(reviews_list[1]) == 2):

            last_review.append(reviews_list[0][0])

            last_review_date.append(reviews_list[1][0])

            prelast_review.append(reviews_list[0][1])

            prelast_review_date.append(reviews_list[1][1])

        elif (len(reviews_list[0]) == 1 and len(reviews_list[1]) == 1):

            last_review.append(reviews_list[0][0])

            last_review_date.append(reviews_list[1][0])

            prelast_review.append('None review')

            prelast_review_date.append('01/01/2000')

        else:

            print(reviews_list)
# Добавим новые признаки:

# - последний отзыв

df['Last Review'] = last_review

# - дата последнего отзыва

df['Last Review Date'] = last_review_date

# - предпоследний отзыв

df['Prelast Review'] = prelast_review

# - дата предпоследнего отзыва

df['Prelast Review Date'] = prelast_review_date
# Переведем даты в формат datetime

df['Last Review Date'] = pd.to_datetime(df['Last Review Date'])

df['Prelast Review Date'] = pd.to_datetime(df['Prelast Review Date'])
# Посмотрим на данные

df.info()
df.head()
df['Days Between Reviews'] = (df['Last Review Date'] - df['Prelast Review Date'])

def get_days(timedelta):

    return timedelta.days

df['Days Between Reviews'] = df['Days Between Reviews'].apply(get_days)
df.info()
df.head()
df['Days Since Last Review'] = df['Last Review Date'].apply(lambda date: CURRENT_DATE - date)

df['Days Since Last Review'] = df['Days Since Last Review'].apply(get_days)
df.info()
df.head()
# Создадим числовой признак 'ID_TA Numeric' на основе 'ID_TA'

df['ID_TA Numeric'] = df['ID_TA'].apply(lambda id_ta: int(id_ta[1:]))
# Создадим числовой признак 'Relative Price Range' на основе признаков 'Price Range' и 'Relative Ranking'

# df['Relative Price Range'] = df['Price Range'] / df['Relative Ranking']

df['Relative Price Range'] = df['Price Range'] / df['Relative Ranking']
# Создадим числовой признак 'People Per Review' на основе признаков 'Population' и 'Number of Reviews'

df['People Per Review'] = df['Population'] / df['Number of Reviews']
df.info()
# Подготовим датафрейм, содержащий все числовые признаки и основанный на данных из основного датасета

# Для удобства сформируем множество численных признаков, на которых можно тренировать модель:

train_features_set = {

    'Ranking',

    'Price Range',

    'Number of Reviews',

    'Restaurants Count',

    'Relative Ranking',

    'City Code',

    'Is Capital',

    'Population',

    'Country Code',

    'People Per Restaurant',

    'Cuisines Count',

    'Unique Cuisine',

    'Days Between Reviews',

    'Days Since Last Review',

    'ID_TA Numeric',

    'Relative Price Range',

#     'People Per Review',

    'Price Range Was NAN',

    'Number of Reviews Was NAN',

    'Cuisine Style Was NAN',

}



# Дополним этот список бинарными признаками присутствия той или иной кухни в меню ресторана

train_features_set = train_features_set.union(set(encoded.columns))



# Создадим объединенный датафрейм для тренировки модели, исключив из него данные из датасета kaggle_task.csv

train_df = pd.concat([df, encoded], axis=1)

train_df = train_df[train_df['Main']]

train_df = train_df[train_features_set]
# Готовим данные

X = train_df

y = df[df['Main']]['Rating']



# Воспользуемся специальной функцией train_test_split для разбивки тестовых данных

# выделим 20% данных на тестирование (параметр test_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
# Создаем модель

rfm = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)
# И наконец тренируем нашу модель

rfm.fit(X_train, y_train)
# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

y_pred = rfm.predict(X_test)
# Поcмотрим на предсказанные данные:

y_pred
# И сразу видим возможность улучшения точности модели.

# Сравним с исходными данными:

df['Rating'].unique()
# Видим разницу в том, что реальные рейтинги всегда кратны 0.5

# Напишем функцию соответствующей корректировки предсказанных рейтингов

def fine_rating_pred(rating_pred):

    if rating_pred <= 0.5:

        return 0.0

    if rating_pred <= 1.5:

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
# Применим такое округление

for i in range(len(y_pred)):

    y_pred[i] = fine_rating_pred(y_pred[i])
y_pred
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

mae = metrics.mean_absolute_error(y_test, y_pred)

print(f'Достигнуто значение MAE: {mae}')
# Best MAE: 0.1643125

# Previews MAE: 0.1649375
plt.figure(figsize=(10,10))

feat_importances = pd.Series(rfm.feature_importances_, index=X.columns)

feat_importances.nlargest(15).plot(kind='barh')
# Создадим датафрейм с валидационными данными для передачи в модель для предсказания рейтингов

# Создадим объединенный датафрейм для предсказания рейтингов, включив в него данные только из датасета kaggle_task.csv

valid_df = pd.concat([df, encoded], axis=1)

valid_df = valid_df[~valid_df['Main']]

valid_df = valid_df[train_features_set]
# Произведем предсказания

valid_y_pred = rfm.predict(valid_df)
# Применим округление

for i in range(len(valid_y_pred)):

    valid_y_pred[i] = fine_rating_pred(valid_y_pred[i])
# Создадим датасет конечного результата submission_df

submission_df = pd.DataFrame()
# Запишем в него требуемые данные

submission_df['Restaurant_id'] = df[~df['Main']]['Restaurant_id']

submission_df['Rating'] = valid_y_pred
# Проверим, что получилось

submission_df
# Сохраним результат в файл

submission_df.to_csv('submission.csv', index=False)