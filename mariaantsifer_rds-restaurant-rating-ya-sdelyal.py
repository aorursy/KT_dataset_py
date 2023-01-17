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
import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



from sklearn.model_selection import train_test_split



RANDOM_SEED = 42



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
# Заполнить пустые строки

data['Number_of_Reviews_isNAN'] = data['Number of Reviews'].isna()

data['Number of Reviews'].fillna(0, inplace=True)



data['Cuisine_Style_isNAN'] = data['Cuisine Style'].isna()

data['Cuisine Style'].fillna("['Unknown']", inplace=True)



data['Price_Range_isNAN'] = data['Price Range'].isna()

data['Price Range'].fillna('$$ - $$$', inplace=True)
# Добавить столбец с информацией о количестве кухонь

CScount = data['Cuisine Style'].str[2:-2].str.split("', '").str.len().fillna(1)

data['Cuisine Styles Amount'] = CScount
#Является ли город столицей или нет

not_capitals = ['Barcelona','Milan','Hamburg','Munich','Lyon','Oporto']

data['Capital'] = data['City'].apply(lambda x: 0 if x in not_capitals else 1)
#Разница во времени между последним и предпоследним отзыовм

#Разница во времени между последним и предпоследним отзыовм

data['Last_Date'] = data['Reviews'].str.findall(r'\d\d/\d\d/\d\d\d\d').str.get(0)

data['Last_Date'] = pd.to_datetime(data['Last_Date'], errors='coerce')

data['Prelast'] = data['Reviews'].str.findall(r'\d\d/\d\d/\d\d\d\d').str.get(1)

data['Prelast'] = pd.to_datetime(data['Prelast'], errors='coerce')

data['Time Difference in Reviews'] = (data['Last_Date'] - data['Prelast'])

data['Time Difference in Reviews'] = data['Time Difference in Reviews'].astype('timedelta64[D]').fillna(0)
# Переведу значения цен в числа

def new_range(r):

    if r =='$':

        return 1

    if r =='$$ - $$$':

        return 2

    if r =='$$$$':

        return 3

new_range('$')

data['Price Range'] = data['Price Range'].apply(new_range)
data['City_Name'] = data['City']

data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
data.info()
data.head()
# Удалим все столбцы типа object

df_preproc = data.drop(['Restaurant_id','City_Name','Cuisine Style','Price Range','Reviews','URL_TA','ID_TA', 'Last_Date', 'Prelast'], axis = 1)
df_preproc.info()
# Выделим тестовую часть

train_data = df_preproc.query('sample == 1').drop(['sample'], axis=1)

test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1)



y = train_data.Rating.values            # наш таргет

X = train_data.drop(['Rating'], axis=1)
# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных

# выделим 20% данных на валидацию (параметр test_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
# Импортируем необходимые библиотеки:

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели
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
test_data.sample(10)
test_data = test_data.drop(['Rating'], axis=1)
sample_submission
predict_submission = model.predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)