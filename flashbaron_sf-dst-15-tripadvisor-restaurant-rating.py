# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from collections import Counter
from matplotlib import pyplot as plt
from datetime import datetime

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'
df_train = pd.read_csv(DATA_DIR+'/main_task.csv')
df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')
sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
df_train['sample'] = 1 # помечаем где у нас трейн
df_test['sample'] = 0 # помечаем где у нас тест
df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями

df = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
# Ваш код по очистке данных и генерации новых признаков
# При необходимости добавьте ячейки
display(df.head(10))
display(df.info())
cuisines_set = set()  # множество из всех кухонь датасета
cuisines_counter = Counter()  # какая кухня сколько раз встречается в датасете

# заполнение cuisines_counter и cuisines_set:
for cuisines in df['Cuisine Style']:
    if (type(cuisines) == float):
        continue
    cuisines = cuisines[2:-2]
    cuisines = cuisines.split("', '")
    for cuisine in cuisines:
        cuisines_set.add(cuisine)
        cuisines_counter[cuisine] += 1


# Добавляем столбец Cuisine_count
df['Cuisine_count'] = df['Cuisine Style'].apply(
    lambda x: 1 if (type(x) == float) else len(x[2:-2].split("', '")))

# среднее количество кухонь в ресторане
cuisines_count_mean = df['Cuisine_count'].mean()

# Добавляем столбец с признаком "рейтинг кухонь ресторана"
cuisines_ranking = pd.Series()
for cuisine in cuisines_set:
    cuisines_ranking[cuisine] = df[df['Cuisine Style'].str.contains(
        cuisine) == True]['Ranking'].median()
cuisines_ranking_mean = cuisines_ranking.median()


def get_cuisines_rating(cuisines):
    # Вычисление рейтинга ресторана относительно его кухонь
    if (type(cuisines) == float):
        return cuisines_ranking_mean * cuisines_count_mean
    cuisines = cuisines[2:-2]
    cuisines = cuisines.split("', '")
    return cuisines_ranking[cuisines].sum()


df['Cuisines_ranking'] = df['Cuisine Style'].apply(get_cuisines_rating)
def get_number_of_price_range(value):
    #   Функция форматирирует столбец из строкого в числовой
    if value == '$':
        return 1
    elif value == '$$ - $$$':
        return 2
    elif value == '$$$$':
        return 3
    else:
        return None


df['Price Range'] = df['Price Range'].apply(get_number_of_price_range)
# fig = plt.figure()
# axes = fig.add_axes([0, 0, 1, 1])

# median = df['Number of Reviews'].median()
# iqr = df['Number of Reviews'].quantile(0.75) - df['Number of Reviews'].quantile(0.25)
# perc25 = df['Number of Reviews'].quantile(0.25)
# perc75 = df['Number of Reviews'].quantile(0.75)
# print('25-й перцентиль: {},'.format(perc25), '75-й перцентиль: {},'.format(perc75)
#       , "IQR: {}, ".format(iqr),"Границы выбросов: [{f}, {l}].".format(f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR))


# axes.hist(x = df[df['Price Range'] == 2]['Number of Reviews'].loc[df['Ranking'].between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)], width=200, label = '\$\$', bins=25)
# axes.hist(x = df[df['Price Range'] == 1]['Number of Reviews'].loc[df['Ranking'].between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)], width=-200, label = '\$', bins=25)
# axes.hist(x = df[df['Price Range'] == 3]['Number of Reviews'].loc[df['Ranking'].between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)], width=80, label = '\$\$\$', bins=25)
# axes.legend(loc = 1)

# df.groupby('Price Range').mean()

# Заполнение Price Range средним значением
price_range_mean = df['Price Range'].median()
df['Price Range'] = df['Price Range'].fillna(price_range_mean)
# groupby через агрегирующую функцию суммы
df_city_groupby_med = df.groupby('City').sum()

# Добавляем столбец City_ranking
df['City_ranking'] = df['City'].apply(
    lambda x: df_city_groupby_med['Ranking'][x])

# Добавляем столбец city_price_range_sum
df['City_price_range'] = df['City'].apply(
    lambda x: df_city_groupby_med['Price Range'][x])

# Добавляем столбец number_review_sum
df['City_Number_of_Reviews'] = df['City'].apply(
    lambda x: df_city_groupby_med['Number of Reviews'][x])
# Заполняем reviews средним значением
reviews_mean = float(round(df['Number of Reviews'].mean()))  # ~125
df['Number of Reviews'] = df['Number of Reviews'].fillna(reviews_mean, axis=0)
def review_splitter(x, i):
    # Разбивает значение Reviews на две части.
    # В зависимости от i, возвращает либо даты ревью, либо сами ревью
    if type(x) == str:
        return x[2:-2].split('], [')[i]
    else:
        return None


df['Reviews_date'] = df['Reviews'].apply(review_splitter, args=(1, ))
df['Reviews_list'] = df['Reviews'].apply(review_splitter, args=(0, ))


def get_first_review_date(value):
    # Возвращает дату первого ревью в формате datetime
    if not value:
        return None
    elif (len(value.split(', ')) == 2):
        return datetime.strptime(value[1:-1].split("', '")[1], '%m/%d/%Y')
    else:
        return datetime.strptime(value[1:-1], '%m/%d/%Y')


df['First_review_date'] = df['Reviews_date'].apply(get_first_review_date)
df['First_review_date'] = df['First_review_date'].apply(lambda x: x.dayofweek)
# Заполнение столбца First_review_date средними значениями
df['First_review_date'] = df['First_review_date'].fillna(
    df['First_review_date'].mean())


def get_timedelta_from_review_date(value):
    # Возвращает дельту между первым и вторым ревью
    if (not value) or (len(value.split(', ')) == 1):
        return None
    else:
        date_min = datetime.strptime(value[1:-1].split("', '")[1], '%m/%d/%Y')
        date_max = datetime.strptime(value[1:-1].split("', '")[0], '%m/%d/%Y')
        return abs(date_max - date_min).days


df['Review_timedelta'] = df['Reviews_date'].apply(
    get_timedelta_from_review_date)
# Заполнение столбца Review_timedelta средними значениями

review_timedelta_value = df['Review_timedelta'].value_counts().mean()
df['Review_timedelta'] = df['Review_timedelta'].fillna(review_timedelta_value)
df['Reviews_list'] = df['Reviews_list'].apply(lambda x: x.lower() if (type(x) == str) else None)


def reviews_counter(text):

    # В зависимости от характера отзыва, будет возвращено значение
    # либо больше, либо меньше
    # Если значение равно нулю - значит либо данных в отзыве скорее
    # всего недостаточно, либо либо отзыв нейтральный
    if type(text) != str:
        return None
    count = 0

    bad_list = set(['bad', 'too bad', 'bad food', 'bad service', 'bad place', 'disappoint',
                    'not good', 'not good place', 'worse food', 'worse service', 'worse place',
                    'terrible', 'terrible food', 'terrible place', 'avoid',
                    'terrible service', 'horrible', 'horrible food', 'horrible place',
                    'awful', 'harmful', 'adverse'])

    good_list = set(['good', 'great', 'super', 'amazing', 'amazing food', 'amazing place',
                     'good food', 'good service', 'music', 'perfect'
                     'nice food', 'nice place', 'nice service', 'friendly',
                     'best food', 'best service', 'best place', 'love',
                     'lovely', 'lovely food', 'lovely place', 'not bad',
                     'very tasty', 'excellent', 'Beautiful', 'enjoyable'])

    for phrase in good_list:
        if phrase in text:
            count += 1
    for phrase in bad_list:
        if phrase in text:
            count -= 1
    return count


df['Reviews_rating'] = df['Reviews_list'].apply(reviews_counter)
# Какое распределение получилось в итоге для признака
df['Reviews_rating'].value_counts().sort_index().plot(
    kind='bar', grid=True, title='Распределение положительности отзывов')

# Можно наблюдать, что большинство отзывов носят положительный характер
# Заполнение пропущенных значений
df['Reviews_rating'] = df['Reviews_rating'].fillna(df['Reviews_rating'].median())
display(df.corr())
display(sns.heatmap(df.corr()))
# Создаём dummy variables для столбца 'Cuisine Style'
for cuisine in cuisines_set:
    df[cuisine] = df['Cuisine Style'].apply(
        lambda x: 1 if (cuisine in str(x)) else 0)
# df.head(10)
# Удаляем лишние столбцы
for column in df.columns:
    if ((df[column].dtype == 'O') and (df[column].name != 'Restaurant_id')):
        df = df.drop(column, axis=1)
# Х - данные с информацией о ресторанах, у - целевая переменная (рейтинги ресторанов)
train_data = df.query('sample == 1').drop(['Restaurant_id', 'sample'], axis=1)
test_data = df.query('sample == 0').drop(['Restaurant_id', 'sample'], axis=1)

X = train_data.drop(['Rating'], axis=1)
y = train_data.Rating.values
# X = df.drop(['Restaurant_id', 'Rating'], axis=1)
# y = df['Rating']
display(X.info())
display(list(X.columns))
# Загружаем специальный инструмент для разбивки:
from sklearn.model_selection import train_test_split
# Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.
# Для тестирования мы будем использовать 25% от исходного датасета.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Импортируем необходимые библиотеки:
# инструмент для создания и обучения модели
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics  # инструменты для оценки точности модели
# Создаём модель
regr = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=42)

# Обучаем модель на тестовом наборе данных
regr.fit(X_train, y_train)

# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
# Предсказанные значения записываем в переменную y_pred
y_pred = regr.predict(X_test)
print('MAE:', metrics.mean_absolute_error(
    y_test, y_pred))
test_data.sample(10)
test_data = test_data.drop(['Rating'], axis=1)
sample_submission
predict_submission = regr.predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head(10)