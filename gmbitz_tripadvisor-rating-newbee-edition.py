# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import ast

import re



import matplotlib.pyplot as plt

import seaborn as sns





from datetime import datetime



import math

from statistics import median

from statistics import mean



from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.ensemble import RandomForestRegressor 

from sklearn import metrics



# Загружаем специальный удобный инструмент для разделения датасета:

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Фиксируйте RANDOM_SEED, чтобы наши эксперименты были воспроизводимы!

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



df = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
df.info()
df.head()
# Булевый признак отстутствия данных в 'Number of Reviews'



df['Number of Reviews NaN'] = df['Number of Reviews'].isna()

df['Number of Reviews NaN'] = df['Number of Reviews NaN'].apply(lambda x: 1 if False else 0)

df['Number of Reviews not NaN'] = df['Number of Reviews NaN'].apply(lambda x: 0 if x == 1 else 1)



# Булевый признак отстутствия данных в 'Cuisine Style'



df['Cuisine Style NaN'] = df['Cuisine Style'].isna()

df['Cuisine Style NaN'] = df['Cuisine Style NaN'].apply(lambda x: 1 if False else 0)

df['Cuisine Style not NaN'] = df['Cuisine Style NaN'].apply(lambda x: 0 if x == 1 else 1)



# Булевый признак отстутствия данных в 'Price Range'



df['Price Range NaN'] = df['Price Range'].isna()

df['Price Range NaN'] = df['Price Range NaN'].apply(lambda x: 1 if False else 0)

df['Price Range not NaN'] = df['Price Range NaN'].apply(lambda x: 0 if x == 1 else 1)
# Создадим словарь, в котором будут храниться данные о количестве ресторанов в каждом городе



df['City'].value_counts()



restaurant_count = dict(df['City'].value_counts())



print(restaurant_count)
df['Restaurant Count'] = df['City'].map(restaurant_count)
df['Relative Rank'] = df['Ranking'] / df['Restaurant Count']
restaurant_chain = set()

for chain in df['Restaurant_id']:

    restaurant_chain.update(chain)

    

def find_item1(cell):

    if item in cell:

        return 1

    return 0



for item in restaurant_chain:

    df['Restaurant chain'] = df['Restaurant_id'].apply(find_item1)

df
# Смотрим какие уникальные данные по ценам присутствуют в датасете



df['Price Range'].unique()
df['Price Range'].value_counts()
# Заменяем значения цен на численные показатели



price_dict = {'$': 10, '$$ - $$$': 100, '$$$$': 1000}

df['Price Range'] = df['Price Range'].map(price_dict)
# Заполняем пропуски медианой



df['Price Range'] = df['Price Range'].fillna(df['Price Range'].median())
df.head()
df.info()
# Создадим словарь, в котором укажем принадлежность города к определенной стране



city_of_country = {

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

    'Dublin': 'Ireland' ,

    'Copenhagen': 'Denmark',

    'Athens': 'Greece',

    'Edinburgh': 'Schotland',

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
df['Country'] = df['City'].map(city_of_country)



df = pd.get_dummies(df, columns=['Country',], dummy_na=True)
# Создадим словарь, в котором укажем данные о количестве жителей городов (Google в помощь)



population = {

    'London': 7556900,

    'Paris': 2138551,

    'Madrid': 3255944,

    'Barcelona': 1621537,

    'Berlin': 3426354,

    'Milan': 1236837,

    'Rome': 2318895,

    'Prague': 1165581,

    'Lisbon': 517802,

    'Vienna': 1691468,

    'Amsterdam': 741636,

    'Brussels': 1019022,

    'Hamburg': 1739117,

    'Munich': 1260391,

    'Lyon': 472317,

    'Stockholm': 1515017,

    'Budapest': 1741041,

    'Warsaw': 1702139,

    'Dublin': 1024027,

    'Copenhagen': 1153615,

    'Athens': 664046,

    'Edinburgh': 464990,

    'Zurich': 341730,

    'Oporto': 249633,

    'Geneva': 183981,

    'Krakow': 755050,

    'Oslo': 580000,

    'Helsinki': 558457,

    'Bratislava': 423737,

    'Luxembourg': 76684,

    'Ljubljana': 272220

}
df['City Population'] = df['City'].map(population)
# Создадим словарь, в котором укажем является ли город столицей (True) или нет (False)



capitals = {

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
df['Is Сapital?'] = df['City'].map(capitals)
df['People per Restaurant'] = round(df['City Population'] / df['Restaurant Count']) 
df['Restaurants per People'] = df['Restaurant Count'] / df['City Population']
df = pd.get_dummies(df, columns=['City',], dummy_na=True)
df.info()
type(df['Cuisine Style'][0])
df['Cuisine Style quantity'] = df['Cuisine Style']



## Заменим названия кухонь в столбце Cuisine Style quantity на их количество для каждого ресторана (вместо Nan мы укажем медину кухонь)



# Найдем медиану



df['Cuisine Style quantity'] = df['Cuisine Style quantity'].fillna('one')

cuisine = df['Cuisine Style quantity'].str[2:-2].str.split("', '")

cuisine2 = []

for i in cuisine:

    cuisine2.append(len(i))

    

Cuisine_Style_median = median(cuisine2)
# Заменим названия кухонь в столбце Cuisine Style quantity на их количество для каждого ресторана



df['Cuisine Style quantity'] = df['Cuisine Style'].fillna(Cuisine_Style_median)

cuisine1_1 = df['Cuisine Style quantity'].str[2:-2].str.split("', '")

cuisine2_1 = cuisine1_1.fillna(Cuisine_Style_median)

df['Cuisine Style quantity'] = cuisine2_1



def quantity(x):

    if x == Cuisine_Style_median:

        return x

    else:

        return len(x)

    

df['Cuisine Style quantity'] = df['Cuisine Style quantity'].apply(quantity)
df['Cuisine Style Unique'] = df['Cuisine Style'].fillna('one')

df['Cuisine Style Unique'] = df['Cuisine Style Unique'].str[2:-2].str.split("', '")



cuisine1_2 = df['Cuisine Style Unique']

cuisine_full = []

for i in cuisine1_2:

    for j in i:

        cuisine_full.append(j)

cuisine_series = pd.Series(cuisine_full)

cuisine_counts = cuisine_series.value_counts()

cuisine_unique = cuisine_counts[cuisine_counts == 1]

cuisine_unique_list = list(cuisine_unique.index[:])



def unique_style(x):

    style_set = set(x)

    return not style_set.isdisjoint(cuisine_unique_list)



df['Cuisine Style Unique'] = df['Cuisine Style Unique'].apply(unique_style)
# Найдем среднее значение кухонь



Cuisine_Style_mean = mean(cuisine2)
# Отношение количества кухонь в ресторане к среднему количеству кухонь



df['Cuisine Style Quantity to Mean'] = df['Cuisine Style quantity'] / Cuisine_Style_mean
# Заполним недостающие данные значением 'Most Common' (Помним, что количество этих кухонь равно медиане среди количества кухонь всех ресторанов)



df['Cuisine Style'].fillna("['Most Common']", inplace=True)
df['Cuisine Style'] = df['Cuisine Style'].str[2:-2].str.split("', '")



# Следующий метот был нагло прогуглен на https://stackoverflow.com/questions/29034928/pandas-convert-a-column-of-list-to-dummies

# Справедливости ради я отчасти разобрался как все работает (отчасти, кек)



df_new = df['Cuisine Style']

df_new_dummy = df_new.apply(lambda x: pd.Series([1] * len(x), index=x)).fillna(0, downcast='infer')



df = pd.merge(df, df_new_dummy, left_index=True, right_index=True, how='left')
df.describe()
def make_list(list_string):

    list_string = list_string.replace('nan]', "'This is Nan']")

    list_string = list_string.replace('[nan', "['This is Nan'")

    result_list = ast.literal_eval(list_string)

    return result_list



df['Reviews'] = df['Reviews'].fillna("[[], []]")

df['Reviews'] = df['Reviews'].apply(make_list)
df['Reviews'][1][1][1]
def delta_date(date):

    if len(date[1]) == 0 or len(date[1]) == 1:

        return 0

    

    elif len(date[1]) == 2:

        date1 = datetime.strptime(date[1][0],'%m/%d/%Y')

        date2 = datetime.strptime(date[1][1],'%m/%d/%Y')

        return abs(date1 - date2).days



df['Days between reviews'] = df['Reviews'].apply(delta_date)
### Обоснованием может являтся то, что рейтигш напрямую зависит от того, насколько "свежий" последний отзыв (информация взята с https://www.tripadvisor.ru/ForRestaurants/r3998)



# Заявим текущую дату



Current_Date = '03/25/2020'



# Определим количество дней с последнего отзыва



Current_Date_dt = datetime.strptime(Current_Date,'%m/%d/%Y')



def days_since_last_review(row):

    if len(row[1]) == 0:

        date = datetime.strptime('02/01/2000','%m/%d/%Y') # Дата основания сайта

    

    elif len(row[1]) == 1:

        date = datetime.strptime(row[1][0],'%m/%d/%Y')

    

    else:

        date1 = datetime.strptime(row[1][0],'%m/%d/%Y')

        date2 = datetime.strptime(row[1][1],'%m/%d/%Y')

        date = max(date1, date2)

    

    return (Current_Date_dt - date).days



df['Days Since Last Review'] = df['Reviews'].apply(days_since_last_review)
df['Reviews Quantity'] = df['Reviews'].apply(lambda x: len(x[0]))
# ID_TA — идентификатор ресторана в базе данных TripAdvisor. Данный столбец можно превратить в численный признак



df['ID_TA Code'] = df['ID_TA'].apply(lambda x: int(x[1:]))
# Заполняем отсутствующие данные в Number of Reviews нулями



df['Number of Reviews'] = df['Number of Reviews'].fillna(0)
object_columns = [s for s in df.columns if df[s].dtypes == 'object']

df.drop(object_columns, axis = 1, inplace=True)



df_preproc = df

df_preproc.sample(10)
df_preproc.info(verbose=True)
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
# Создаём модель



model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)
# Обучаем модель на тестовом наборе данных



model.fit(X_train, y_train)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred



y_pred = model.predict(X_test)
y_pred
# Реальные рейтинги на сайте всегда кратны 0.5. Следовательно следует ввести корректировки



def real_rating(x, base = 0.5):

    return base * round(x/base)



for i in range(len(y_pred)):

    y_pred[i] = real_rating(y_pred[i])
y_pred
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# MAE Стабильный, за 5 запусков не поменял значение
# в RandomForestRegressor есть возможность вывести самые важные признаки для модели

plt.rcParams['figure.figsize'] = (10,10)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(15).plot(kind='barh')
test_data.sample(10)
test_data = test_data.drop(['Rating'], axis=1)
sample_submission
predict_submission = model.predict(test_data)
for i in range(len(predict_submission)):

    predict_submission[i] = real_rating(predict_submission[i])
predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)