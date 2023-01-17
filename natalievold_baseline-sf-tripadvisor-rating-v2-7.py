# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



from sklearn.preprocessing import LabelEncoder



import datetime as dt

from datetime import date, timedelta

import re



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
# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

!pip freeze > requirements.txt
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
df_train.info()
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем

data
data.info()
# на всякий случай, заново подгружаем данные

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем

data.sample(5)
data.info()
def preproc_data(df_input):

    '''includes several functions to pre-process the predictor data.'''

    

    df_output = df_input.copy()

    

    # ################### 1. Предобработка ############################################################## 

    # убираем не нужные для модели признаки

    #df_output.drop(['Restaurant_id','ID_TA',], axis = 1, inplace=True)

    

    

    # ################### 2. NAN ############################################################## 

    # Далее заполняем пропуски, вы можете попробовать заполнением средним или средним по городу и тд...

    df_output['Number of Reviews'].fillna(0, inplace=True)

    df_output['Reviews'].fillna(0, inplace=True)

  

     # ################### 3. Feature Engineering ####################################################

    # Создаем признак "Столица или нет"

    capitals = {'London': 1, 'Paris': 1, 'Madrid': 1, 'Barcelona': 0, 'Berlin': 1, 'Milan': 0, 'Rome': 1, 'Prague': 1, 'Lisbon': 1, 

    'Vienna': 1, 'Amsterdam': 1, 'Brussels': 1, 'Hamburg': 0, 'Munich': 0, 'Lyon': 0, 'Stockholm': 1, 'Budapest': 1, 'Warsaw': 1,

    'Dublin': 1, 'Copenhagen': 1, 'Athens': 1, 'Edinburgh': 1, 'Zurich': 0, 'Oporto': 0, 'Geneva': 0, 'Krakow': 0, 'Oslo': 1,

    'Helsinki': 1, 'Bratislava': 1, 'Luxembourg': 1, 'Ljubljana': 1}



    def dict_capital(city):

        return capitals[city]

 



    df_output['Capital'] = df_output.City.apply(dict_capital)



    # Создаем признак "Население городов"

    population = {'Paris': 2190327, 'Stockholm': 961609, 'London': 8908081, 'Berlin': 3644826, 'Munich': 1456039, 'Oporto': 237591,

                  'Milan': 1378689,'Bratislava': 432864, 'Vienna': 1821582, 'Rome': 4355725, 'Barcelona': 1620343, 'Madrid': 3223334,

                  'Dublin': 1173179,'Brussels': 179277, 'Zurich': 428737, 'Warsaw': 1758143, 'Budapest': 1752286, 'Copenhagen': 615993,

                  'Amsterdam': 857713,'Lyon': 506615, 'Hamburg': 1841179,'Lisbon': 505526, 'Prague': 1301132, 'Oslo': 673469,

                  'Helsinki': 643272,'Edinburgh': 488100,'Geneva': 200548, 'Ljubljana': 284355,'Athens': 664046, 'Luxembourg': 115227,

                  'Krakow': 769498}



    def dict_population(city):

        return population[city]

 



    df_output['Population'] = df_output.City.apply(dict_population)

    

    #В какой стране ресторан:

    city_country = {'London': 'UK','Paris': 'France','Madrid': 'Spain','Barcelona': 'Spain','Berlin': 'Germany','Milan': 'Italy',

                    'Rome': 'Italy','Prague': 'Czech','Lisbon': 'Portugalia','Vienna': 'Austria','Amsterdam': 'Nederlands','Brussels': 'Belgium ',

                    'Hamburg': 'Germany','Munich': 'Germany','Lyon': 'France','Stockholm': 'Sweden','Budapest': 'Hungary','Warsaw': 'Poland',

                    'Dublin': 'Ireland' ,'Copenhagen': 'Denmark','Athens': 'Greece','Edinburgh': 'Schotland','Zurich': 'Switzerland','Oporto': 'Portugal',

                    'Geneva': 'Switzerland','Krakow': 'Poland','Oslo': 'Norway','Helsinki': 'Finland','Bratislava': 'Slovakia','Luxembourg': 'Luxembourg',

                    'Ljubljana': 'Slovenija'}

    df_output['Country'] = df_output['City'].map(city_country)

    

    # создание признака "количество ресторанов в городе"

    df_output['Rest per City'] = df_output['City'].map(df_output.groupby(['City'])['City'].count().to_dict())

    

    # создание признака "количество ресторанов на человека в городе"

    df_output['Pop per (Rest per City)'] = df_output['City'].map(population) / df_output['Rest per City']

    

    # создание признака "количество отзывов к количеству ресторанов на человека в городе"

    df_output['Rev per (Rest per Pers)'] = df_output['Number of Reviews'] / df_output['Pop per (Rest per City)']

    

    # создание признака "относительный рэнкинг"

    df_output['Relative Ranking'] = df_output['Ranking'] / df_output['Rest per City']

    

    # создание признака "количество типов кухонь, представленных в ресторане

    df_output['Cuisine Style1'] = df_output['Cuisine Style'].str[1:-1].str.split(',').str.len().fillna(1)

        

    # создание признака "сетевой ресторан"

    restaurant_set = set()

    for chain in df_output['Restaurant_id']:

        restaurant_set.update(chain)

    def find_item(cell):

        if item in cell:

            return 1

        return 0

    for item in restaurant_set:

        df_output['Rest Chain'] = df_output['Restaurant_id'].apply(find_item)

        

    # создание признака "код города"

    cities_le = LabelEncoder()

    cities_le.fit(df_output['City'])

    df_output['City Code'] = cities_le.transform(df_output['City'])

    countries_le = LabelEncoder()

    countries_le.fit(df_output['Country'])

    df_output['Country Code'] = countries_le.transform(df_output['Country'])

        

    #Обработка данных о ценовом диапазоне ресторана:

    df_output['Price Range'] = df_output['Price Range'].str.replace(r'(\${4})','3')

    df_output['Price Range'] = df_output['Price Range'].str.replace(r'(\${2} - \${3})','2')

    df_output['Price Range'] = df_output['Price Range'].str.replace(r'((?<!\$)\$(?!\$))','1')

    df_output['Price Range'].value_counts()

    df_output['Price Range'] = df_output['Price Range'].fillna('2')

    

    # создание признака "ID URL_TA"

    df_output['URL_TA New'] = df_output['URL_TA'].apply(lambda x: float(x[20:26]))

    

    # создание признака "ID ID_TA"

    df_output['ID_TA New'] = df_output['ID_TA'].apply(lambda x: float(x[1:]))

    

    

   

    

    

    # ################### 4. Encoding ############################################################## 

    # для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

    #df_output = pd.get_dummies(df_output, columns=[ 'City',], dummy_na=True)

    # тут ваш код не Encoding фитчей

   



    df_output['Cuisine Style'].fillna("['Local']", inplace=True)

    df_output['Cuisine Style'] = df_output['Cuisine Style'].apply(lambda x: str(x)[1:-1])

    df_output['Cuisine Style'] = df_output['Cuisine Style'].apply(lambda x: x.replace("'", ""))

    cuisines = df_output['Cuisine Style'].str.get_dummies(sep=',').add_prefix('Cuisine_')

    df_output = df_output.join(cuisines)

    

    df_output = pd.get_dummies(df_output, columns=['Price Range'], dummy_na=True)

    df_output = pd.get_dummies(df_output, columns=['Capital'], dummy_na=True)



   

    

    

    # ################### 5. Clean #################################################### 

    # убираем признаки которые еще не успели обработать, 

    # модель на признаках с dtypes "object" обучаться не будет, просто выберим их и удалим

    object_columns = [s for s in df_output.columns if df_output[s].dtypes == 'object']

    df_output.drop(object_columns, axis = 1, inplace=True)

    

    return df_output
df_preproc = preproc_data(data)

df_preproc.sample(5)
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

feat_importances.nlargest(20).plot(kind='barh')
test_data.sample(10)
test_data = test_data.drop(['Rating'], axis=1)
sample_submission
predict_submission = model.predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)