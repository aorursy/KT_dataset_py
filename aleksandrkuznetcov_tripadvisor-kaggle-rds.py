import pandas as pd

import numpy as np

from os.path import join

import re



import matplotlib.pyplot as plt

import seaborn as sns



# matplotlib рисовует все сразу без напоминаний

%matplotlib inline



# Удобный инструмент для разделения датасета:

from sklearn.model_selection import train_test_split



from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Для воспроизводимости результатов зададим:

# - общий параметр для генерации случайных чисел

RANDOM_SEED = 42
# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

!pip freeze > requirements.txt
# Путь к csv файлам

dfs_path = '/kaggle/input/sf-dst-restaurant-rating/'



# Название файлов, с которыми будем работать

main_ds = 'main_task.csv'

kaggle_ds = 'kaggle_task.csv'



# Путь к файлами

main_path = join(dfs_path, main_ds)

kaggle_path = join(dfs_path, kaggle_ds)



# Открываем датасеты

main_df = pd.read_csv(main_path)

kaggle_df = pd.read_csv(kaggle_path)
# Данный из 'main_task.csv'

display(main_df.info())
display(main_df.head(5))
# Данный из 'kaggle_task.csv'

display(kaggle_df.info())
display(kaggle_df.head(5))
# Два датафрейма почти идентичным по признакам. У main_df есть признак Rating, но нет признакок Name

# у kaggle_df наоборот. Добавим в kaggle_df признак Rating заполнив его 0

kaggle_df['Rating'] = 0



# Удалим признак Name

# kaggle_df = kaggle_df.drop(columns=['Name'])
kaggle_df.info()
# объединяем оба датафрейма в один для дальнейшей работы

df = pd.concat([main_df, kaggle_df], sort=False)

df.info()
df.head(10)
# Избавимся от NaN в признаке Cuisine Style

# Так как ресторан не может быть совсем без какого-либо вида кухни, предположим, что в ресторанах

# где стоит значение NaN есть хотя бы один тип кухни. Пусть он будет Local (местная)

df['Cuisine Style'] = df['Cuisine Style'].fillna('Local')
df.head(5)
# Избавимся от NaN в признаке Price Range. заполним значения NaN средним значением признакак, т.е. $$ - $$$

df['Price Range'] = df['Price Range'].fillna('$$ - $$$')
df.info()
# Избавимся от NaN в признаке Number of Reviews. Заполним значения NaN 0

df['Number of Reviews'] = df['Number of Reviews'].fillna(0)
df.info()
# Создаем словарь, в котором указываем являеться ли город столицей или нет

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
# Добавляем новый признак в датафрейм. Признак показывает находиться ли ресторан в столице или нет

df['Capital'] = df['City'].map(is_capital)
df.head(5)
# Добавим новый признак с кол-во людей в городах



cities_population = {

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



df['Population'] = df['City'].map(cities_population)
# Добавим код города (подсмотрел в чужом notebook'e)

# Переведем информацию о городах в числовой формат

# с применением sklearn.preprocessign.LabelEncoder

cities_num = LabelEncoder()

cities_num.fit(df['City'])

df['City Code'] = cities_num.transform(df['City'])
df.head(5)
df['Price Range'].value_counts()
# Преобразуем признак Price Range в числово. Предположим что люди в среднем ходят по 2 человека в рестораны.

# тогда будем считать что $ - 20 euro средний чек за двоих

#                  $$ - $$$ - 60 euro средний чек за двоих

#                      $$$$ - 100 euro средний чек за двоих



price = {

    '$': 20,

    '$$ - $$$': 60,

    '$$$$': 100

}



df['Price Range'] = df['Price Range'].map(price)
df.head(5)
# Преобразуем данные признака Cuisine Style из str в list

# Функция преобразующая строковые данные и строки Cuisine Style в данные типа список

def str_to_list(x):

    y = []

    a = ['Local']

    if x != 'Local':

        x = str(x[2:-2]).split('\', \'')

        for i in x:

            y.append(i)

        return y

    else:

        return a



df['Cuisine Style'] = df['Cuisine Style'].apply(str_to_list)
df.head(5)
# Считаем кол-во кухонь в каждом ресторане

def count_of_cuisine(data):

    if type(data) == list:

        return len(data)

    

df['Count of cuisines'] = df['Cuisine Style'].apply(count_of_cuisine)
df.head(5)
df.info()
# Создаем временный пустой датафрейм для хранения всех типов кухонь как отдельных признаков

temp_cuisines_df = pd.DataFrame()



# Функция записывающая все типы кухонь в множество

set_of_cuisines = set()

def count_style(x):

    if type(x) == list:

        for style in x:

            set_of_cuisines.add(style)

    return x



df['Cuisine Style'] = df['Cuisine Style'].apply(count_style)



# Функция заполняющая пустой датафрейм новыми признаками

def cuisines_columns(data):

    if cuisine in data:

        return 1

    return 0



for cuisine in set_of_cuisines:

    temp_cuisines_df[cuisine] = df['Cuisine Style'].apply(cuisines_columns)
temp_cuisines_df.head(5)
df.info()
# Создадим числовой признак 'ID_TA Numeric' на основе 'ID_TA'

df['ID_TA Numeric'] = df['ID_TA'].apply(lambda id_ta: int(id_ta[1:]))
df.info()
def sample_col(data):

    if data == 0:

        return 0

    else:

        return 1

    

df['Sample'] = df['Rating'].apply(sample_col)



train_df = pd.concat([df, temp_cuisines_df], axis=1)



# Убираем все не числовые признаки из датафрейма

train_df = train_df.drop(columns=['City', 'Restaurant_id', 'Cuisine Style','Reviews','URL_TA','ID_TA'])

train_df.head(5)
# Теперь выделим тестовую часть

train_data = train_df.query('Sample == 1').drop(['Sample'], axis=1)

test_data = train_df.query('Sample == 0').drop(['Sample'], axis=1)



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
sample_submission = pd.DataFrame()

sample_submission['Restaurant_id'] = kaggle_df['Restaurant_id']

sample_submission['Rating'] = kaggle_df['Rating']
predict_submission = model.predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission
sample_submission.to_csv('sample_submission.csv', index = False)  