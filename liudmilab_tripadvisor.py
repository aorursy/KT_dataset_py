
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline

# Загружаем специальный удобный инструмент для разделения датасета:
from sklearn.model_selection import train_test_split

# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!
RANDOM_SEED = 42
# загрузить датасет и ознакомиться с данными, форматами и особенностями хранения информации в файле

DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'
df_train = pd.read_csv(DATA_DIR+'main_task.csv')
df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')
sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет
df_train['sample'] = 1 # помечаем где у нас трейн
df_test['sample'] = 0 # помечаем где у нас тест
df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями

df = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
df.info()
df.sample(5)
# Перед как очистить данные от пропусков NAN,вынесем информацию о наличии пропуска как отдельный признак 

df['Number_of_Reviews_isNAN'] = pd.isna(df['Number of Reviews']).astype('uint8')
df['Cuisine Style_isNAN'] = pd.isna(df['Cuisine Style']).astype('uint8')
df['Price Range_isNAN'] = pd.isna(df['Price Range']).astype('uint8')
df['Reviews_isNAN'] = pd.isna(df['Reviews']).astype('uint8')
df['Reviews_isNAN'] = pd.isna(df['Reviews']).astype('uint8')

# Далее заполняем пропуски NaN в Number of Reviews на 0

df['Number of Reviews']=df['Number of Reviews'].fillna(0)
plt.rcParams['figure.figsize'] = (10,7)
#df_train['Ranking'].plot(kind = 'hist', grid = True, title = 'Рейтинг')
df['Ranking'].hist(bins=100)
df['City'].value_counts(ascending=True).plot(kind = 'barh', grid = True, title = 'Представленность городов')
df.pivot_table(values = ['Ranking'], 
               index = 'City', 
               aggfunc = 'mean').plot(kind = 'bar')
#Добавим столбец для предобработки, признак города в списке топ-туристических

tur_city = ['London', 'Paris', 'Berlin','Rome', 'Milan', 'Barcelona',  'Prague', 'Lisbon', 'Vienna', 'Madrid', 'Amsterdam']
df['Tur_City'] = df['City'].apply(lambda x: 1 if x in tur_city else 0 )
# Обработка признаков

# Добавим столбец для предобработки, диапозон цен - от 0 до 3 
def price_level (row):
    if pd.isna(row['Price Range']):
        return 0
    elif row['Price Range']=='$':
        return 1
    elif row['Price Range']=='$$ - $$$':
        return 2
    elif row['Price Range']=='$$$$':
        return 3
        
df['price_level'] = df.apply(lambda row: price_level (row), axis = 1)
df.pivot_table(values = ['Ranking', 'Number of Reviews'], 
               index = 'price_level', 
               aggfunc = 'mean').plot(kind = 'bar')
# Добавляем столбец количество представленных стилей кухонь в ресторане

df['Cuisine_count'] =df['Cuisine Style'].str[2:-2].str.split("', '").fillna('1').str.len()
df.pivot_table(values = ['Ranking', 'Number of Reviews'], 
               index = 'Cuisine_count', 
               aggfunc = 'mean').plot(kind = 'bar')
# Добавляем столбец признак города столицы

city_list = ['London', 'Paris', 'Stockholm', 'Madrid', 'Berlin', 'Rome', 'Prague', 'Lisbon', 'Vienna', 'Amsterdam', 'Budapest', 'Warsaw', 'Dublin', 'Copenhagen', 
             'Athens', 'Edinburgh', 'Oslo', 'Helsinki', 'Bratislava', 'Ljubljana', 'Brussels', 'Luxembourg']

df['City_Сapital'] = df['City'].apply(lambda x: 1 if x in city_list else 0 )
# Добавляем столбец показатель разницы дней отзывов ресторане

df['Date_1'] = df['Reviews'].str.findall(r'\d+\W\d+\W\d+').str.get(0)
df['Date_1'] =pd.to_datetime(df['Date_1'], errors='coerce')
df['Date_2'] = df['Reviews'].str.findall(r'\d+\W\d+\W\d+').str.get(1)
df['Date_2'] =pd.to_datetime(df['Date_2'], errors='coerce')
df['Data_difference'] = (df['Date_1'] - df['Date_2'])
df['Data_difference'] = df['Data_difference'].astype('timedelta64[D]').fillna(0)
# Добавляем категорийный признак городов ресторанов по модели One-Hot Encoding

df['City_temp'] = df['City']
df = pd.get_dummies(df, columns=['City'], dummy_na=True)
df = df.rename(columns = {'City_temp': 'City'})
# Добавляем столбец признак наличия негативного отзыва о ресторане

df['Reviews'] = df['Reviews'].str[2:-2].str.split(" , ").fillna('0')
def not_food (cell):
     for i in cell:
        if 'not' in i.lower():
            return 0
        else:
            return 1

df['Not_Reviews'] = df['Reviews'].apply(not_food)

# Добавляем категорийный признак стили кухонь ресторанов

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer

df['Cuisine Style'] = df['Cuisine Style'].astype(str).str[2:-2].str.split("', '").fillna(0) 
mlb = MultiLabelBinarizer()
df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('Cuisine Style')) , index=df.index, columns=mlb.classes_))
# Удалим все столбцы формата object
df.drop(['Restaurant_id', 'City','Price Range', 'Reviews', 'URL_TA', 'ID_TA', 'Date_1', 'Date_2'], axis='columns', inplace = True)
# Разделение датафрейма

train_data = df.query('sample == 1').drop(['sample'], axis=1)
test_data = df.query('sample == 0').drop(['sample'], axis=1)

# Х - данные с информацией о ресторанах, у - целевая переменная (рейтинги ресторанов)
y = train_data.Rating.values            
X = train_data.drop(['Rating'], axis=1)

# Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.
# Для тестирования мы будем использовать 25% от исходного датасета.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)


# Создаём модель

# Импортируем необходимые библиотеки:
from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
from sklearn import metrics # инструменты для оценки точности модели
regr = RandomForestRegressor(n_estimators=100)

# Обучаем модель на тестовом наборе данных
regr.fit(X_train, y_train)

# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
# Предсказанные значения записываем в переменную y_pred
y_pred = regr.predict(X_test)
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются
# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
test_data = test_data.drop(['Rating'], axis=1)

predict_submission = regr.predict(test_data)
predict_submission
# Заменим рейтинг на наш спрогнозированный рейтинг

sample_submission['Rating'] = predict_submission

# Проверим, что получилось

sample_submission.head(10)
# Округлим Рейтинг до десятых

sample_submission.round(1).head(10)
# Запишем датасет в файл cvs

sample_submission.to_csv('submission.csv', index=False)