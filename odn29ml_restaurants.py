import pandas as pd

import numpy as np

from ast import literal_eval

from datetime import datetime as dt
# Прочитаем датасет

df = pd.read_csv('../input/sf-dst-restaurant-rating/main_task.csv')
# Заменим наименование столбцов для более удобной работы

df.columns = ['restaurant_id', 'city', 'cuisine_style', 'ranking',

              'rating', 'price_range', 'reviews_number', 'reviews', 'url_ta', 'id_ta']
# Посмотрим на информацию о датафрейме

df.info()
# Обнаружены пропуски в столбце Number of Reviews. Можно сделать предположение, что если

# в столбце reviews указан пустой список, значит количество отзывов = 0. Заменим такие пропуски.

# df[df.reviews_number.isna() & (df.reviews == '[[], []]')].loc[:,('reviews_number')] = 0

df.loc[df.reviews_number.isna() & (df.reviews == '[[], []]'),

       'reviews_number'] = 0
# Дальше мы можем предположить, что если в столбце reviews представлена информация только

# об одном отзыве, то количество отзывов = 1. Заменим такие пропуски.

# df[df.reviews_number.isna() & (df.reviews.str.match('^\[\[\'.*\'\],\s?\[\'.*\'\]\]$'))] = 1

df.loc[df.reviews_number.isna() & (df.reviews.str.match(

    '^\[\[\'.*\'\],\s?\[\'.*\'\]\]$')), 'reviews_number'] = 1
# Проверим, что получилось

df.info()
# Т.о. осталось только 28 пропусков в столбце reviews_number, заменим их средним значением

df.fillna(value={'reviews_number': df.reviews_number.mean()}, inplace=True)
# Поскольку показатель ranking зависит от общего количества ресторанов в городе, то можно

# представить его в виде относительной величины как отношение к максимальному по городу

groups_by_cities = df.groupby('city').ranking.max()

df.ranking = df.apply(lambda x: x.ranking/groups_by_cities[x.city], axis=1)
# У нас есть категориальный признак cuisine_style, создадим на его основе dummy признаки

# При этом уменьшим количество значений этого признака, переимновав такие значения в Other

df_styles = df.cuisine_style.fillna('[]').apply(lambda x: literal_eval(x))

all_cuisine_styles = []

for style in df_styles:

    all_cuisine_styles += style

cuisine_styles = set(all_cuisine_styles)

cuisine_styles_new = set()

for style in cuisine_styles:

    freq = df.cuisine_style.str.contains(style).sum()

    if freq < 100:

        df.cuisine_style = df.cuisine_style.replace(style, 'Other')

        cuisine_styles_new.add('Other')

    else:

        cuisine_styles_new.add(style)

for style in cuisine_styles_new:

    df[style] = df.cuisine_style.fillna('').apply(

        lambda x: 1 if style in x else 0)
# Создадим также dummy признаки для второго категориального признака - city

df = df.join(pd.get_dummies(df.city))
# Приведем показатель price_range к числовому значению и заменим пропуски на среднее значение

df.price_range = df.price_range.replace('$', 1)

df.price_range = df.price_range.replace('$$ - $$$', 2)

df.price_range = df.price_range.replace('$$$$', 3)

df.price_range.fillna(df.price_range.mean(), inplace=True)
# Посчитаем сколько дней прошло с момента последнего отзыва и добавим эти данные в качестве нового признака.

# При этом будем исходить из предположения, что датой составления датасета является максимальная дата отзыва

# из всего датасета. Также предположим, что если отзывов нет, эта цифра равна среднему количеству дней

# в датасете.

df.reviews = df.reviews.apply(lambda x: literal_eval(x.replace('nan', "''")))

df['review1_date'] = df.reviews.apply(lambda x: x[1][0] if len(x[1]) else '')

df['review2_date'] = df.reviews.apply(

    lambda x: x[1][1] if len(x[1]) > 1 else '')

df.review1_date = pd.to_datetime(df.review1_date)

df.review2_date = pd.to_datetime(df.review2_date)

df['last_review_date'] = df.apply(

    lambda x: x.review1_date if x.review1_date > x.review2_date else x.review2_date, axis=1)

last_date = df.last_review_date.max()

df['last_review_days'] = df.apply(lambda x: (

    last_date - x.last_review_date).days, axis=1)

df.last_review_days = df.last_review_days.fillna(df.last_review_days.mean())
# Добавим новый признак, показывающий сколько дней прошло между двумя представленными в датасете отзывами

# и заполним максимальным значением те строки, где двух отзывов нет

df['interval'] = df.apply(lambda x: abs(

    (x.review1_date - x.review2_date).days), axis=1)

df.interval.fillna(df.interval.max(), inplace=True)
# Посмотрим на распределение некоторых числовых признаков

df[['interval', 'last_review_days', 'reviews_number',

    'ranking', 'price_range']].hist(bins=5)
# for feature in ['last_review_days', 'reviews_number', 'ranking', 'interval']:

df.boxplot(column='last_review_days')
df.boxplot(column='reviews_number')
df.boxplot(column='ranking')
df.boxplot(column='interval')
df.boxplot(column='price_range')
# Можно было бы посчитать по признаку reviews_number значения более 6000 и по признаку last_review_days

# значения более 3000 выбросами, однако проведенные тесты показали, что их удаление не

# приводит к улучшению точности модели.
# Распределение признаков interval и last_review_days не выглядит нормальным, однако оно обусловлено

# большим количеством отсутствующих данных, которые мы были вынуждены заменить исходя из

# наших предположений.
# Удалим все нечисловые столбцы

df.drop(['review1_date', 'review2_date',

         'last_review_date'], axis=1, inplace=True)

for col in df:

    if df[col].dtype == np.object:

        df.drop(columns=[col], inplace=True)
# Посмотрим на корреляцию признаков.

df[['rating', 'last_review_days', 'ranking',

    'reviews_number', 'interval', 'price_range']].corr()
# Из таблицы мы видим, что силно скоррелированных признаков нет, а значит наш датасет пригоден

# для обучения модели
# Х - данные с информацией о ресторанах, у - целевая переменная (рейтинги ресторанов)

# Проведем нормализацию наших признаков

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = df.drop(['rating'], axis=1)

scaler.fit_transform(df)

y = df['rating']
# Загружаем специальный инструмент для разбивки:

from sklearn.model_selection import train_test_split
# Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.

# Для тестирования мы будем использовать 25% от исходного датасета.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
# Импортируем необходимые библиотеки:

# инструмент для создания и обучения модели

from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics  # инструменты для оценки точности модели
# Создаём модель

regr = RandomForestRegressor(n_estimators=100)



# Обучаем модель на тестовом наборе данных

regr.fit(X_train, y_train)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

y_pred = regr.predict(X_test)
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# В результате преобразований, удалось снизить отклонение предсказания более чем в раза.