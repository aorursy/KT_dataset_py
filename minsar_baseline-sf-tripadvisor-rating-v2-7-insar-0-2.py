# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели



import matplotlib.pyplot as plt

import seaborn as sns

import ast

import numpy as np

from datetime import datetime, timedelta 

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

plt.rcParams['figure.figsize'] = (10,7)
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



raw_df = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
raw_df.info()
raw_df.sample(5)
raw_df['Reviews'][1]
#скопируем сырые данные в датафрейм, с которым будем работать и переименуем столбцы

df = raw_df.copy()

df.columns = df.columns.str.replace(' ','_').str.lower()

df.info()
df['city'].value_counts().plot(kind='barh')
sns.boxplot(x='rating', y='city', data=df)
sns.boxplot(x='ranking', y='city', data=df)
top_N_cities_list = df['city'].value_counts().index[:7] # отберем топ-7 городов по кол-ву ресторанов

df['top_N_city'] = df['city'].apply(lambda x: x if x in top_N_cities_list else 'Other')

city_dummy = pd.get_dummies(df['top_N_city'])

df = pd.concat([df, city_dummy], axis = 1)
not_capitals_list = ['Barcelona', 'Milan', 'Hamburg', 'Munich', 

                          'Lyon', 'Zurich', 'Oporto', 'Geneva', 'Krakow']

df['if_capital'] = df['city'].apply(lambda x: 0 if x in not_capitals_list else 1)
city_population_dict = {'London' : 8908, 'Paris' : 2206, 'Madrid' : 3223, 'Barcelona' : 1620, 

                        'Berlin' : 6010, 'Milan' : 1366, 'Rome' : 2872, 'Prague' : 1308, 

                        'Lisbon' : 506, 'Vienna' : 1888, 'Amsterdam' : 860, 'Brussels' : 179, 

                        'Hamburg' : 1841, 'Munich' : 1457, 'Lyon' : 506, 'Stockholm' : 961, 

                        'Budapest' : 1752, 'Warsaw' : 1764, 'Dublin' : 553, 

                        'Copenhagen' : 616, 'Athens' : 665, 'Edinburgh' : 513, 

                        'Zurich' : 415, 'Oporto' : 240, 'Geneva' : 201, 'Krakow' : 769, 

                        'Oslo' : 681, 'Helsinki' : 643, 'Bratislava' : 426, 

                        'Luxembourg' : 119, 'Ljubljana' : 284}

df['city_population'] = df.apply(lambda x: city_population_dict[x['city']], axis = 1)
# считаем, что max(ranking) - и есть максимальное количество ресторанов в городе

city_rest_num = df.groupby(['city'])['ranking'].max()

df['city_rest_num'] = df['city'].apply(lambda x: city_rest_num[x])

df['city_rest_density'] = df['city_population']/df['ranking']
# видим много пропусков, чтобы не потерять эти данные - сформируем признак

df['is_nan_cuisine_style'] = pd.isna(df['cuisine_style']).astype('float64')
# судя по всему pands считывает список как строку (добавляя " в начле и конце, исправим это)

df['cuisine_style'] = df['cuisine_style'].str[2:-2].str.split("', '")

df['cuisine_style'] = df['cuisine_style'].fillna("['Other']")
# подсчитаем количество представленных типов кухонь для каждого ресторана

df['cuisine_num'] = df['cuisine_style'].str.len()
df['ranking'].hist(bins=100)
# У нас много ресторанов, которые не дотягивают и до 2500 места в своем городе, а что там по городам?

# посмотрим на топ 10 городов

for x in (df['city'].value_counts())[0:10].index:

    df['ranking'][df['city'] == x].hist(bins=100)

plt.show()
# тк в разных городах разное количество ресторанов, попробуем нормализовать ranking по городам

# и посмотреть н распределение

# для простоты решил воспользоваться формулой, а не StandardScaler

df['ranking_normalized'] = df.groupby('city')['ranking'].transform(lambda x: (x-x.mean())/x.std())

for x in (df['city'].value_counts())[0:10].index:

    df['ranking_normalized'][df['city'] == x].hist(bins=100)

plt.show()
# понятно, что 10 ресторан из 1000 и 10 ресторан из 10000 - это большая разница

# поэтому введем признак 'ranking_relative'

df['ranking_relative'] = df['ranking'] / df['city_rest_num']
df['price_range'].value_counts()
#создадим отдельный признак для значений, где был NaN

df['is_nan_price_range'] = pd.isna(df['price_range']).astype('float64')



# здесь значения - порядковые, можем их закодировать

price_range_dict = {'$': 0, '$$ - $$$': 1, '$$$$': 2}

df = df.replace({'price_range': price_range_dict})

df['price_range'] = df['price_range'].fillna(1)
# в признаке 6% пропущенных значений - скорее всего, не было комментариев

# сохраним эту информацию

df['is_nan_number_of_reviews'] = pd.isna(df['number_of_reviews']).astype('float64')



# предположим, что NaN в number_of_reviews - это отсутствие комментариев, заменим на 0

# хотя это не совсем корректно, т.к. в столбце 'reviews' у некоторых таких ресторанов есть отзывы

df['number_of_reviews'].fillna(0, inplace=True)
def return_review_date(row, mode = 'newest'):

    '''replacing nan values from text reviews cause of ast.literal_eval error

       Function returns NaN if reviews are empty

       Function returns one date if it is only one review

       Function returns newest/oldest review according to mode'''

    str_review = ast.literal_eval(str(row['reviews']).replace('nan','0'))

    if str_review == [[], []]:

        return 'NaN'

    elif len(str_review[1]) == 1:

        return(datetime.strptime(str_review[1][0],'%m/%d/%Y'))

    else:

        first_review_time = datetime.strptime(str_review[1][0],'%m/%d/%Y')

        second_review_time = datetime.strptime(str_review[1][1],'%m/%d/%Y')

        if mode == 'newest':

            return max(second_review_time, first_review_time)

        else:

            return min(second_review_time, first_review_time)
# видим, что в столбце 'reviews' есть 2 NaN, заменим их, чтобы наша функция корректно работала

df['reviews'].fillna('[[], []]', inplace=True)



#заполним даты обзоров



df['newest_review_date'] = df.apply(lambda row:return_review_date(row, mode='newest'), axis=1)

df['eldest_review_date'] = df.apply(lambda row:return_review_date(row, mode='eldest'), axis=1)
# создадим признак для тех ресторанов, у которых нет обзоров вообще

df['is_nan_reviews'] = pd.isna(df['newest_review_date']).astype('float64')
# посмотрим на дату самого свежего обзора, предположим, что тогда датасет и собирался

newest_review_date = df['newest_review_date'].max()

newest_review_date
# посмотрим на дату самого старого обзора, учтем что Tripadviser был создан в 2000 году

oldest_review_date = df['eldest_review_date'].min()

oldest_review_date
# для ресторанов без отзывов добавим минимальную дату

df['newest_review_date'].fillna(oldest_review_date, inplace=True)

df['eldest_review_date'].fillna(oldest_review_date, inplace=True)
# Посчитаем разницу в днях между обзорами

# Предполагаем, что 'сегодня' = дата самого нового обзора в датасете

df['days_to_most_recent_review'] = (newest_review_date - df['newest_review_date']).dt.days

df['days_between_oldest_newest_reviews'] = (df['newest_review_date'] - df['eldest_review_date']).dt.days
# проверим работу функции

df[['reviews', 'newest_review_date', 'eldest_review_date', 'days_to_most_recent_review',

   'is_nan_reviews', 'days_between_oldest_newest_reviews']].sample(10)
sns.heatmap(df.corr())
# на всякий случай, заново подгружаем данные

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



df = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем

df.info()
def preproc_data(df_input):

    '''includes several functions to pre-process the predictor data.'''

    

    df = df_input.copy()

    df.columns = df.columns.str.replace(' ','_').str.lower()

    

    # 'cities'

    top_N_cities_list = df['city'].value_counts().index[:7] # отберем топ-7 городов по кол-ву ресторанов

    df['top_N_city'] = df['city'].apply(lambda x: x if x in top_N_cities_list else 'Other')

    city_dummy = pd.get_dummies(df['top_N_city'])

    df = pd.concat([df, city_dummy], axis = 1)

    

    not_capitals_list = ['Barcelona', 'Milan', 'Hamburg', 'Munich', 

                          'Lyon', 'Zurich', 'Oporto', 'Geneva', 'Krakow']

    df['if_capital'] = df['city'].apply(lambda x: 0 if x in not_capitals_list else 1)

    

    city_population_dict = {'London' : 8908, 'Paris' : 2206, 'Madrid' : 3223, 'Barcelona' : 1620, 

                        'Berlin' : 6010, 'Milan' : 1366, 'Rome' : 2872, 'Prague' : 1308, 

                        'Lisbon' : 506, 'Vienna' : 1888, 'Amsterdam' : 860, 'Brussels' : 179, 

                        'Hamburg' : 1841, 'Munich' : 1457, 'Lyon' : 506, 'Stockholm' : 961, 

                        'Budapest' : 1752, 'Warsaw' : 1764, 'Dublin' : 553, 

                        'Copenhagen' : 616, 'Athens' : 665, 'Edinburgh' : 513, 

                        'Zurich' : 415, 'Oporto' : 240, 'Geneva' : 201, 'Krakow' : 769, 

                        'Oslo' : 681, 'Helsinki' : 643, 'Bratislava' : 426, 

                        'Luxembourg' : 119, 'Ljubljana' : 284}

    df['city_population'] = df.apply(lambda x: city_population_dict[x['city']], axis = 1)

    city_rest_num = df.groupby(['city'])['ranking'].max()

    df['city_rest_num'] = df['city'].apply(lambda x: city_rest_num[x])

    df['city_rest_density'] = df['city_population']/df['ranking']

    

    # 'cuisine_style'

    df['is_nan_cuisine_style'] = pd.isna(df['cuisine_style']).astype('float64')

    df['cuisine_style'] = df['cuisine_style'].str[2:-2].str.split("', '")

    df['cuisine_style'] = df['cuisine_style'].fillna("['Other']")

    

    # 'ranking'

    df['ranking_normalized'] = df.groupby('city')['ranking'].transform(lambda x: (x-x.mean())/x.std())

    df['ranking_relative'] = df['ranking'] / df['city_rest_num']

    

    # 'price_range'

    df['is_nan_price_range'] = pd.isna(df['price_range']).astype('float64')

    price_range_dict = {'$': 0, '$$ - $$$': 1, '$$$$': 2}

    df = df.replace({'price_range': price_range_dict})

    df['price_range'] = df['price_range'].fillna(1)

    

    # 'number_of_reviews'

    df['is_nan_number_of_reviews'] = pd.isna(df['number_of_reviews']).astype('float64')

    df['number_of_reviews'].fillna(0, inplace=True)

    

    # 'reviews'

    df['reviews'].fillna('[[], []]', inplace=True)

    df['newest_review_date'] = df.apply(lambda row:return_review_date(row, mode='newest'), axis=1)

    df['eldest_review_date'] = df.apply(lambda row:return_review_date(row, mode='eldest'), axis=1)

    df['is_nan_reviews'] = pd.isna(df['newest_review_date']).astype('float64')

    newest_review_date = df['newest_review_date'].max()

    oldest_review_date = df['eldest_review_date'].min()

    df['newest_review_date'].fillna(oldest_review_date, inplace=True)

    df['eldest_review_date'].fillna(oldest_review_date, inplace=True)

    df['days_to_most_recent_review'] = (newest_review_date - df['newest_review_date']).dt.days

    df['days_between_oldest_newest_reviews'] = (df['newest_review_date'] - df['eldest_review_date']).dt.days

    

    # removing object columns

    df = df.select_dtypes(exclude=['object','datetime64[ns]'])

    

    return df
df_preproc = preproc_data(df)

df_preproc.sample(10)
df_preproc.info()
# Теперь выделим тестовую часть

train_data = df_preproc.query('sample == 1').drop(['sample'], axis=1)

test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1)



y = train_data['rating'].values            # наш таргет

X = train_data.drop(['rating'], axis=1)
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

   

# Т. к. целевая переменная кратна 0.5, добавим здесь округление y_pred до 0.5

y_pred = np.round(y_pred*2)/2
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# в RandomForestRegressor есть возможность вывести самые важные признаки для модели

plt.rcParams['figure.figsize'] = (10,10)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(15).plot(kind='barh')
test_data.sample(10)
test_data = test_data.drop(['rating'], axis=1)
sample_submission
predict_submission = model.predict(test_data)
predict_submission = np.round(predict_submission*2)/2
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)