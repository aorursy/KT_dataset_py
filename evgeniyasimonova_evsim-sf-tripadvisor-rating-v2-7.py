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

import re

from datetime import datetime
# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!

RANDOM_SEED = 42
# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

!pip freeze > requirements.txt
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
print(DATA_DIR)
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

data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')
data['Price_Range_isNAN'] = pd.isna(data['Price Range']).astype('uint8')
# Далее заполняем пропуски 0, вы можете попробовать заполнением средним или средним по городу и тд...

data['Number of Reviews'].fillna(0, inplace=True)
data.nunique(dropna=False)
# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
data.head(5)
data.sample(5)
data['Price Range'].value_counts()
# Ваша обработка 'Price Range'

data.loc[data['Price Range'] == '$','Price Range'] = 1

data.loc[data['Price Range'] == '$$ - $$$','Price Range'] = 2

data.loc[data['Price Range'] == '$$$$','Price Range'] = 3

data['Price Range'].unique()
# тут ваш код на обработку других признаков

# .....
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
country = {'Paris': 'France', 'Stockholm': 'Sweden', 'London': 'Great_Britain', 'Berlin': 'Germany', 'Munich': 'Germany',

           'Oporto': 'Portugal', 'Milan': 'Italy', 'Bratislava': 'Slovakia', 'Vienna': 'Austria', 'Rome': 'Italy', 

           'Barcelona': 'Spain', 'Madrid': 'Spain', 'Dublin': 'Ireland', 'Brussels': 'Belgium', 'Zurich': 'Switzerland',

           'Warsaw': 'Poland', 'Budapest': 'Hungary', 'Copenhagen': 'Denmark', 'Amsterdam': 'Netherlands',

           'Lyon': 'France', 'Hamburg': 'Germany', 'Lisbon': 'Portugal', 'Prague': 'Czech', 'Oslo': 'Norway',

           'Helsinki': 'Finland', 'Edinburgh': 'Scotland', 'Geneva': 'Switzerland', 'Ljubljana': 'Slovenia',

           'Athens': 'Greece', 'Luxembourg': 'Luxembourg', 'Krakow': 'Poland'}

data.apply(lambda row: country[row['City']], axis=1)
сity_population = {'Paris': 2148, 'Stockholm': 975, 'London': 8982, 'Berlin': 3769, 'Munich': 1472,

           'Oporto': 214, 'Milan': 1352, 'Bratislava': 424, 'Vienna': 1897, 'Rome': 2873, 

           'Barcelona': 5575, 'Madrid': 6642, 'Dublin': 1388, 'Brussels': 174, 'Zurich': 402,

           'Warsaw': 1708, 'Budapest': 1752, 'Copenhagen': 602, 'Amsterdam': 821,

           'Lyon': 513, 'Hamburg': 1899, 'Lisbon': 504, 'Prague': 1309, 'Oslo': 681,

           'Helsinki': 631, 'Edinburgh': 482, 'Geneva': 499, 'Ljubljana': 279,

           'Athens': 664, 'Luxembourg': 613, 'Krakow': 769}

data.apply(lambda row: сity_population[row['City']], axis = 1)
def preproc_data(df_input):

    '''includes several functions to pre-process the predictor data.'''

    

    df_output = df_input.copy()

    

    # ################### 0. Добавление признаков перед их удалением ##################################### 

    # создадим столбец с количеством кухонь в ресторане

    df_output['cuisine_count'] = df_output['Cuisine Style'].str.count(',')+1

    df_output['cuisine_count'].fillna(0, inplace=True)

    # заменим диапазон цен на 1, 2 и 3

    df_output.loc[data['Price Range'] == '$','Price Range'] = 1

    df_output.loc[data['Price Range'] == '$$ - $$$','Price Range'] = 2

    df_output.loc[data['Price Range'] == '$$$$','Price Range'] = 3

    # количество ресторанов в сети 

    net_restaurant = df_output.groupby(['Restaurant_id']).Restaurant_id.count().sort_values(ascending=False)

    df_output['count_restaurant'] = df_output.apply(lambda row: net_restaurant[row['Restaurant_id']], axis=1)

          

    

    # ################### 1. Предобработка ############################################################## 

    # убираем не нужные для модели признаки

    df_output.drop(['Restaurant_id','ID_TA',], axis = 1, inplace=True)

    

    

    # ################### 2. NAN ############################################################## 

    # Далее заполняем пропуски, вы можете попробовать заполнением средним или средним по городу и тд...

    df_output['Number of Reviews'].fillna(0, inplace=True)

    # тут ваш код по обработке NAN

    # ....

    df_output['Price Range'].fillna(2, inplace=True)

    # заменим количество отзывов на среднее по городу значение

    mean_number = round(df_train.groupby('City')['Number of Reviews'].mean(),0)

    df_output['Number of Reviews'] = df_output.apply(lambda row: mean_number[row['City']] if row['Number of Reviews'] == 0 else row['Number of Reviews'], axis=1)

    # создадим столбец с разницей дней между имеющимися отзывами

    df_output['Reviews'].fillna('[[], []]', inplace=True)

    reviews = pd.DataFrame([re.findall(r'\d{2}/\d{2}/\d{4}', x) for x in df_output['Reviews']]) 

    reviews.columns = ['data1', 'data2']

    reviews.fillna('01/01/2000', inplace=True)

    reviews.data1 = pd.to_datetime(reviews.data1)

    reviews.data2 = pd.to_datetime(reviews.data2)

    df_output['delta'] = abs((reviews.data1 - reviews.data2).dt.days)

    # создадим столбец с разницей последнего отзыва и текущей датой

    today = pd.to_datetime(datetime.now())

    reviews['delta_today'] = reviews.apply(lambda row: (today - row.data1).days if row.data1 > row.data2 else (today - row.data2).days, axis=1)

    df_output['delta_today'] = reviews['delta_today']

    #попробуем использовать внешние данные для нашей модели, рассмотрим какие страны объедняют наши города

    country_city = {'Paris': 'France', 'Stockholm': 'Sweden', 'London': 'Great_Britain', 'Berlin': 'Germany', 'Munich': 'Germany',

           'Oporto': 'Portugal', 'Milan': 'Italy', 'Bratislava': 'Slovakia', 'Vienna': 'Austria', 'Rome': 'Italy', 

           'Barcelona': 'Spain', 'Madrid': 'Spain', 'Dublin': 'Ireland', 'Brussels': 'Belgium', 'Zurich': 'Switzerland',

           'Warsaw': 'Poland', 'Budapest': 'Hungary', 'Copenhagen': 'Denmark', 'Amsterdam': 'Netherlands',

           'Lyon': 'France', 'Hamburg': 'Germany', 'Lisbon': 'Portugal', 'Prague': 'Czech', 'Oslo': 'Norway',

           'Helsinki': 'Finland', 'Edinburgh': 'Scotland', 'Geneva': 'Switzerland', 'Ljubljana': 'Slovenia',

           'Athens': 'Greece', 'Luxembourg': 'Luxembourg', 'Krakow': 'Poland'}

    df_output['country'] = df_output.apply(lambda row: country_city[row['City']], axis=1)

    # посмотрим население

    сity_population = {'Paris': 2148, 'Stockholm': 975, 'London': 8982, 'Berlin': 3769, 'Munich': 1472,

           'Oporto': 214, 'Milan': 1352, 'Bratislava': 424, 'Vienna': 1897, 'Rome': 2873, 

           'Barcelona': 5575, 'Madrid': 6642, 'Dublin': 1388, 'Brussels': 174, 'Zurich': 402,

           'Warsaw': 1708, 'Budapest': 1752, 'Copenhagen': 602, 'Amsterdam': 821,

           'Lyon': 513, 'Hamburg': 1899, 'Lisbon': 504, 'Prague': 1309, 'Oslo': 681,

           'Helsinki': 631, 'Edinburgh': 482, 'Geneva': 499, 'Ljubljana': 279,

           'Athens': 664, 'Luxembourg': 613, 'Krakow': 769}

    df_output['сity_population'] = df_output.apply(lambda row: сity_population[row['City']], axis = 1)

    

    # ################### 3. Encoding ############################################################## 

    # для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

    df_output = pd.get_dummies(df_output, columns=[ 'City',], dummy_na=True)

    # тут ваш код не Encoding фитчей

    # ....

    df_output = pd.get_dummies(df_output, columns=[ 'country',], dummy_na=True)

    df_output['Cuisine Style'] = df_output['Cuisine Style'].apply(lambda x: re.findall(r"[^'][\w\s]+[^']", str(x)))

    df_output = pd.concat([df_output, 

                 pd.get_dummies(df_output.explode('Cuisine Style')['Cuisine Style'], dummy_na=True)], axis = 1)

    

    # ################### 4. Feature Engineering ####################################################

    # тут ваш код не генерацию новых фитчей

    # ....

                   

    

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
test_data
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

y_pred = np.round(y_pred * 2) / 2
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
# приведем рейтинг к кратности шага 0.5

predict_submission = list(map(lambda x: round(x * 2)/2, predict_submission))

predict_submission
sample_submission['Rating'] = predict_submission[0:10000]

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)