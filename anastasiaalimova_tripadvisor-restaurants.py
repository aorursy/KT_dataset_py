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
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR + 'main_task.csv')

df_test = pd.read_csv(DATA_DIR + 'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR + 'sample_submission.csv')





RANDOM_SEED = 42
from datetime import timedelta



def check_population(x):

    city_popularion = {

    'Paris': 2140526,

    'Stockholm': 961609,

    'London':  8787892,

    'Berlin': 3601131,

    'Munich': 1456039,

    'Oporto': 214349,

    'Milan': 1366180,

    'Bratislava': 424428,

    'Vienna': 1840573,

    'Rome': 2872800,

    'Barcelona': 1620343,

    'Madrid': 3223334,

    'Dublin': 553165,

    'Brussels': 1198726,

    'Zurich': 402762,

    'Warsaw': 1758143,

    'Budapest': 1749734,

    'Copenhagen': 615993,

    'Amsterdam': 859732,

    'Lyon': 515695,

    'Hamburg': 1830584,

    'Lisbon': 505526,

    'Prague': 1280508,

    'Oslo': 673469,

    'Helsinki': 643272,

    'Edinburgh': 482005,

    'Geneva': 495249,

    'Ljubljana': 279631,

    'Athens': 655780,

    'Luxembourg': 602005,

    'Krakow': 769498,     

    }

    for city, population in city_popularion.items():

        if x == city:

            return population



def check_capital(x):

    capitals = [

        'Paris', 'Stockholm', 'London', 'Berlin', 'Bratislava', 'Vienna', 'Rome', 'Madrid',

        'Dublin', 'Brussels', 'Warsaw', 'Budapest', 'Copenhagen', 'Amsterdam', 'Lisbon', 'Prague', 'Oslo',

        'Helsinki', 'Edinburgh', 'Ljubljana', 'Athens', 'Luxembourg'

    ]

    if x in capitals:

        return 1

    return 0



def review_date(x):

    x = str(x).replace("nan", "None")

    x = eval(x)

    return x[1] if x else []



def check_period(x):

    if len(x) == 2:

        return x[0] - x[1]

    return timedelta()



def price(x):

    if x == '$':

        return 10

    if x == '$$ - $$$':

        return 100

    else:

        return 1000



def str_to_list(x):

     if type(x) == str:

        return eval(x)

     return []



def prepare_dataset(df_input):

    df = df_input.copy()

    city_list = df['City'].unique()

        

    df['Capital'] = df['City'].apply(check_capital)

    df['Cuisine_number'] = df['Cuisine Style'].fillna('1').str[2:-2].str.split("', '").apply(lambda x: len(x))

    last_review_series = df["Reviews"].apply(review_date).apply(lambda x: pd.to_datetime(x))

    df['Delta'] = last_review_series.apply(check_period).apply(lambda x: x.days)

    df['Delta'] = df['Delta'].fillna(df['Delta'].dropna().mean())

    df['Price'] = df["Price Range"].apply(price)

    df["Number of Reviews"] = df["Number of Reviews"].fillna(df['Number of Reviews'].mean())

    df["Ranking"] = df["Ranking"].fillna(df['Ranking'].mean())

    df["Capital"] = df["Capital"].fillna(0)

    df["Cuisine_number"] = df["Cuisine_number"].fillna(df["Cuisine_number"].mean())

    df["Price"] = df["Price"].fillna(df['Price'].mean())

    popular_cities_ser = df['City'].value_counts()

    df['Number of Restaurants'] = df['City'].apply(lambda x: popular_cities_ser.loc[x])

    df['population'] = df.City.apply(check_population)

    #добавлем еще один дамми

    price_d = pd.get_dummies(df['Price Range'])

    df = pd.concat([df, price_d], axis = 1)

    for city in city_list:

        df[city] = df['City'].apply(lambda cell: int(city in cell))

    

    #Единый спиок кухонь

    ser = df['Cuisine Style'].dropna().str[2:-2].str.split("', '")

    l = []

    for i in ser:

         for j in i:

            l.append(j)



    cuisine_list = pd.Series(l).unique()



    df['Cuisine List'] = df['Cuisine Style'].apply(str_to_list)

    for cuisine_style in cuisine_list:

        df[cuisine_style] = df['Cuisine List'].apply(lambda row: int(cuisine_style in row))



    return df.drop(['Restaurant_id', 'City', "Cuisine Style", 'Price Range', "URL_TA", "ID_TA", "Reviews", 'Cuisine List'], axis = 1)
df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем



df_preproc = prepare_dataset(data)
# Теперь выделим тестовую часть

train_data = df_preproc.query('sample == 1').drop(['sample'], axis=1) # 40000 with rating

test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1) # 10000 without rating



y = train_data.Rating.values            # наш таргет

X = train_data.drop(['Rating'], axis=1)



# выделим 20% данных на валидацию (параметр test_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели



# Создаём модель (НАСТРОЙКИ НЕ ТРОГАЕМ)

model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)



# Обучаем модель на тестовом наборе данных

model.fit(X_train, y_train)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

y_pred = model.predict(X_test)



print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
test_data = test_data.drop(['Rating'], axis=1)

predict_submission = model.predict(test_data)

sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)
plt.rcParams['figure.figsize'] = (10,10)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(15).plot(kind='barh')