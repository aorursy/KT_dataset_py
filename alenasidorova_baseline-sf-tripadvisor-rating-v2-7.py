import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline

import datetime

from sklearn.model_selection import train_test_split

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from collections import Counter

import re

from sklearn.preprocessing import StandardScaler
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
#Исследуем



data['Cuisine Style']
data['Cuisine Style'].value_counts()
data['NAN_Cuisine Style'] = pd.isna(data['Cuisine Style']).astype('float64')

data['Cuisine Style'].fillna("['Other']", inplace=True)
data.info()
data['NAN_Price Range'] = pd.isna(data['Price Range']).astype('float64')



data['Price Range'] = data['Price Range'].fillna(0)

data.info()
data['NAN_Number_of_Reviews_is'] = pd.isna(data['Number of Reviews']).astype('float64')
# Далее заполняем пропуски 0, вы можете попробовать заполнением средним или средним по городу и тд...

data['Number of Reviews'].fillna(0, inplace=True)
data.info()
data['NAN_Reviews'] = pd.isna(data['Reviews']).astype('float64')

data['Reviews'].fillna("[[], []]", inplace=True)
data.info()
data.Restaurant_id.value_counts()
def change_id(x):

    if 'id_' in str(x):

        return str(x).replace('id_', '')

    else: return x

data.Restaurant_id = data.Restaurant_id.apply(change_id)

data.Restaurant_id = pd.to_numeric(data.Restaurant_id)
data.info()
data.City.value_counts()
data['City_origin'] = data['City']
#data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
data.info()
type(data['Cuisine Style'].iloc[0])
data['Cuisine Style'] = data['Cuisine Style'].apply(lambda x: re.findall('\w+\s*\w+\s*\w+', str(x)))
type(data['Cuisine Style'].iloc[15])
data.info()
data.Ranking.value_counts()
data['Price Range'].value_counts()
#Преобразуем



price_dict = {"0": 0, "$": 1, "$$ - $$$": 2, "$$$$": 3}



data['Price Range'] = data['Price Range'].replace(to_replace=price_dict)

data['Price Range'].value_counts()
data['Number of Reviews'].value_counts()
type(data.Reviews.iloc[0])
data['Reviews'].value_counts()
def fill_na_reviews(x):

    if x == '[[], []]':

        return None

    else:

        return x

data['Reviews'] = data['Reviews'].fillna(fill_na_reviews)
data['Review_date'] = data['Reviews'].str.findall('\d+/\d+/\d+')

data['Review_date']
data.info()
data.URL_TA
data.drop(['URL_TA'], axis=1, inplace=True)
data.info()
data.ID_TA
def change_id_TA(x):

    if 'd' in str(x):

        return str(x).replace('d', '')

    else: return x

data.ID_TA = data.ID_TA.apply(change_id_TA)

data.ID_TA = pd.to_numeric(data.ID_TA)
data.info()
data.columns
cuisines = set()



for i in data['Cuisine Style']:

    for j in i:

        cuisines.add(j)
cuisines
food = {}  # создаём пустой словарь для хранения информации об ингредиентах

for item in cuisines:  # перебираем список ингредиентов

    food[item] = 0 # добавляем в словарь ключ, соответствующий очередному ингредиенту



for i in data['Cuisine Style']:   # перебираем список рецептов

    for j in i:   # и список ингредиентов в каждом рецепте

        food[j] += 1   # увеличиваем значение нужного ключа в словаре на 1
food
pop_cuisine = []

for key, value in food.items():

    if value > 3000:

        pop_cuisine.append(key)

pop_cuisine
type(data['Cuisine Style'].iloc[0])
def popular_cuisine(x):

    for element in pop_cuisine:

        if element in x:

            return 1

        else:

            continue

data['Popular_cuisine'] = data['Cuisine Style'].apply(popular_cuisine)
data['Popular_cuisine'].value_counts(dropna = False)
data['Popular_cuisine'].fillna(0, inplace = True)
data.info()
data['Count_cuisines'] = data['Cuisine Style'].apply(lambda x: len(x))
data['Count_cuisines'].value_counts(dropna=False)
data.info()
data['Review_date'] = data['Reviews'].str.findall('\d+/\d+/\d+')

data['len_of_reviews'] = data['Review_date'].apply(lambda x: x if x == None else len(x))

data['len_of_reviews'].value_counts(dropna = False)
data['len_of_reviews'].fillna(2, inplace = True)
def days_to_now(row):

    if row['Review_date'] == None:

        return None

    return datetime.datetime.now() - pd.to_datetime(row['Review_date']).max()



def days_between_reviews(row):

    if row['Review_date'] == None:

        return None

    return pd.to_datetime(row['Review_date']).max() - pd.to_datetime(row['Review_date']).min()



data['Days_to_now'] = data.apply(days_to_now, axis = 1).dt.days

data['Days_between_reviews'] = data[data['len_of_reviews']==2].apply(days_between_reviews, axis = 1).dt.days

data.info()
data['Days_between_reviews'].value_counts(dropna = False)
data['Days_to_now'] = data['Days_to_now'].fillna(data['Days_to_now'].median())

data['Days_between_reviews'] = data['Days_between_reviews'].fillna(data['Days_between_reviews'].median())

data.info()
data['City_origin'].value_counts()
Сity_population_dict = {'London' : 8908, 'Paris' : 2206, 'Madrid' : 3223, 'Barcelona' : 1620, 

                        'Berlin' : 6010, 'Milan' : 1366, 'Rome' : 2872, 'Prague' : 1308, 

                        'Lisbon' : 506, 'Vienna' : 1888, 'Amsterdam' : 860, 'Brussels' : 179, 

                        'Hamburg' : 1841, 'Munich' : 1457, 'Lyon' : 506, 'Stockholm' : 961, 

                        'Budapest' : 1752, 'Warsaw' : 1764, 'Dublin' : 553, 

                        'Copenhagen' : 616, 'Athens' : 665, 'Edinburgh' : 513, 

                        'Zurich' : 415, 'Oporto' : 240, 'Geneva' : 201, 'Krakow' : 769, 

                        'Oslo' : 681, 'Helsinki' : 643, 'Bratislava' : 426, 

                        'Luxembourg' : 119, 'Ljubljana' : 284}

data['Сity_population'] = data.apply(lambda row: Сity_population_dict[row['City_origin']], axis = 1)
data.info()
data['Сount_of_rest'] = data['City_origin'].map(data.groupby('City_origin')['Ranking'].max().to_dict())
data['Сount_of_rest'].value_counts()
data['Rest_on_person'] = data['Сity_population']/data['Сount_of_rest']
data['Rest_on_person']
data['Ranking_on_city'] = data['Ranking'] / data['Сount_of_rest']
data.info()
data.columns
correlation = data[data['sample'] == 1][['Ranking', 'Price Range','Number of Reviews', 'Rating','Popular_cuisine', 'Count_cuisines', 'len_of_reviews', 'Days_to_now', 'Days_between_reviews', 'Сity_population', 'Сount_of_rest']].corr()

plt.figure(figsize=(20, 10))

sns.heatmap(correlation, annot=True, cmap='coolwarm')
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
mn = data.groupby('City_origin')['Ranking'].mean()

st = data.groupby('City_origin')['Ranking'].std()

data['Std_Ranking'] = (data['Ranking'] - data['City_origin'].map(mn))/data['City_origin'].map(st)
data.info()
data.columns
data.drop(['Restaurant_id','NAN_Cuisine Style', 'NAN_Price Range', 'Cuisine Style', 'Reviews', 'ID_TA', 'NAN_Cuisine_Style', 'len_of_reviews','NAN_Price_Range', 'NAN_Number_of_Reviews_is','NAN_Reviews', 'City_origin', 'City', 'Review_date'], axis=1, inplace=True, errors='ignore')
data.columns
# Теперь выделим тестовую часть

train_data = data.query('sample == 1').drop(['sample'], axis=1)

test_data = data.query('sample == 0').drop(['sample'], axis=1)



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
sample_submission
predict_submission = model.predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)