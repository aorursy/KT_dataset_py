# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

import re

from datetime import datetime

from functools import reduce

%matplotlib inline



# Загружаем специальный удобный инструмент для разделения датасета:

from sklearn.model_selection import train_test_split



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# фиксируйте RANDOM_SEED, чтобы эксперименты были воспроизводимы

RANDOM_SEED = 42



# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы

!pip freeze > requirements.txt
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
df_train.info()
df_train.head()
df_test.info()
df_test.head()
sample_submission.info()
sample_submission.head()
#Для корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True)

data.info()
data.sample(10)
# Необходимо обработать данные

data = data.drop_duplicates()

data.info()
data.nunique()
# Рассмотрим вначале признаки City и Price Range

# Так как поле City содержит небольшое количество уникальных значений, можно добавить dummies переменные

data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)



# Поле Price Range содержит всего 3 уникальных значения. Данный признак можно преобразовать к численному виду

dict_price = {'$$$$': 3,'$$ - $$$': 2,'$': 1}

data['Price Range'] = data['Price Range'].replace(to_replace=dict_price).fillna(0)



data.sample(5)
# Рассмотрим признак "Cuisine Style" и создадим новый признак, показывающий количество типов кухни в ресторане

# Если в поле нет данных, установим значение 1, так как хотя бы одна кухня точно присутствует в ресторане

data['Cuis_style_count'] = data['Cuisine Style'].fillna('net_dannyh')

data['Cuis_style_count'] = data['Cuis_style_count'].apply(lambda x: len(re.sub(r"[\[, \], ']", " ", x).strip().split()))



data.sample(5)
# Далее рассмотрим признак Number of Reviews и заполним NaN значения нулями

data['Number of Reviews'] = data['Number of Reviews'].fillna(0)



# Осталось преобразовать признак Reviews

# Создадим новый признак, показывающий разница в днях последних двух комментариев. В противном случае 0

data['Reviews_razn'] = data['Reviews'].apply(lambda x: re.findall(r'\d{2}/\d{2}/\d{4}', str(x)))

data['Reviews_razn'] = data['Reviews_razn'].apply(

    lambda x: reduce(lambda a,b : (datetime.strptime(a,'%m/%d/%Y') - datetime.strptime(b,'%m/%d/%Y')).days, x)

    if len(x)==2 else 0).astype(int)



# Создадим новый признак, показывающий отношение Ranking на Number of Reviews

data['Ranking_by_rev'] = round(data['Ranking'] / data['Number of Reviews'], 2)



data.sample(5)
df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
def preproc_data(df_input):

    

    df_output = df_input.copy()

    

    # Предобработка данных

    df_output = df_output.drop_duplicates()

    df_output.drop(['Restaurant_id','ID_TA',], axis = 1, inplace=True)

    

    # Заполнение NaN значений.

    df_output['Number of Reviews'] = df_output['Number of Reviews'].fillna(0)

    df_output['Cuisine Style'] = df_output['Cuisine Style'].fillna('net_dannyh')

    df_output['Price Range'] = df_output['Price Range'].fillna(0)

    

    #Создаем dummies признаки

    df_output = pd.get_dummies(df_output, columns=[ 'City',], dummy_na=True)

    

    # Feature Engineering

    # 1. "Price Range"

    dict_price = {'$$$$': 3,'$$ - $$$': 2,'$': 1}

    df_output['Price Range'] = df_output['Price Range'].replace(to_replace=dict_price).fillna(0)

    

    # 2. "Cuisine Style count"

    df_output['Cuis_style_count'] = df_output['Cuisine Style'].apply(lambda x: len(re.sub(r"[\[, \], ']", " ", x).strip().split()))

    

    # 3. "Reviews difference days"

    df_output['Reviews_razn'] = df_output['Reviews'].apply(lambda x: re.findall(r'\d{2}/\d{2}/\d{4}', str(x)))

    df_output['Reviews_razn'] = df_output['Reviews_razn'].apply(

        lambda x: reduce(lambda a,b : (datetime.strptime(a,'%m/%d/%Y') - datetime.strptime(b,'%m/%d/%Y')).days, x)

        if len(x)==2 else 0).astype(int)

    

    # 4. "Ranking by Reviews"

    df_output['Ranking_by_rev'] = round(df_output['Ranking'] / data['Number of Reviews'], 2).fillna(0)

    

    # Убираем ненужные признаки

    object_columns = [s for s in df_output.columns if df_output[s].dtypes == 'object']

    df_output.drop(object_columns, axis = 1, inplace=True)

    

    return df_output
df_preproc = preproc_data(data)

df_preproc.sample(10)
df_preproc.info()
# Теперь выделим тестовую часть

train_data = df_preproc.query('sample == 1').drop(['sample'], axis=1)

test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1)



y = train_data.Rating.values

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