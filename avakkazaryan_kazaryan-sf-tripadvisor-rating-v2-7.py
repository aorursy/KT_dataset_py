# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import collections



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
data.sample(5)
data.Reviews[1]
# Для примера я возьму столбец Number of Reviews

data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')
data['Number_of_Reviews_isNAN']
# Далее заполняем пропуски 0, вы можете попробовать заполнением средним или средним по городу и тд...

data['Number of Reviews'].fillna(0, inplace=True)
data.nunique(dropna=False)
def city_population(x):

    if x == 'Paris':

        x = int(2148327)

    elif x == 'Helsinki':

        x = int(655281)

    elif x == 'Edinburgh':

        x = int(488100)

    elif x == 'London':

        x = int(8908081)

    elif x == 'Bratislava':

        x = int(437725)

    elif x == 'Lisbon':

        x = int(505526)

    elif x == 'Budapest':

        x = int(1752286)

    elif x == 'Stockholm':

        x = int(961609)

    elif x == 'Rome':

        x = int(2870500)

    elif x == 'Milan':

        x = int(1378689)

    elif x == 'Munich':

        x = int(1471508)

    elif x == 'Hamburg':

        x = int(1841179)

    elif x == 'Prague':

        x = int(1301132)

    elif x == 'Vienna':

        x = int(1897491)

    elif x == 'Dublin':

        x = int(1173179)

    elif x == 'Barcelona':

        x = int(1636762)    

    elif x == 'Brussels':

        x = int(179277)

    elif x == 'Madrid':

        x = int(3266126)    

    elif x == 'Oslo':

        x = int(673469)

    elif x == 'Amsterdam':

        x = int(872757)    

    elif x == 'Berlin':

        x = int(3644826)

    elif x == 'Lyon':

        x = int(506615)

    elif x == 'Athens':

        x = int(664046)    

    elif x == 'Warsaw':

        x = int(1790658)

    elif x == 'Oporto':

        x = int(237591)    

    elif x == 'Krakow':

        x = int(779115)

    elif x == 'Copenhagen':

        x = int(615993)    

    elif x == 'Luxembourg':

        x = int(602005)

    elif x == 'Zurich':

        x = int(428737)

    elif x == 'Geneva':

        x = int(200548)

    elif x == 'Ljubljana':

        x = int(284355)

    return x
data['City_population'] = data['City'].apply(city_population)
data.head()
# плотность населения

def city_density(x):

    if x == 'Paris':

        x = int(20781)

    elif x == 'Helsinki':

        x = int(3035)

    elif x == 'Edinburgh':

        x = int(1830)

    elif x == 'London':

        x = int(5173)

    elif x == 'Bratislava':

        x = int(1169)

    elif x == 'Lisbon':

        x = int(4883)

    elif x == 'Budapest':

        x = int(3351)

    elif x == 'Stockholm':

        x = int(4800)

    elif x == 'Rome':

        x = int(2232)

    elif x == 'Milan':

        x = int(7700)

    elif x == 'Munich':

        x = int(4500)

    elif x == 'Hamburg':

        x = int(2320)

    elif x == 'Prague':

        x = int(2700)

    elif x == 'Vienna':

        x = int(16000)

    elif x == 'Dublin':

        x = int(4588)

    elif x == 'Barcelona':

        x = int(15779)    

    elif x == 'Brussels':

        x = int(5384)

    elif x == 'Madrid':

        x = int(5390)    

    elif x == 'Oslo':

        x = int(1645)

    elif x == 'Amsterdam':

        x = int(4908)    

    elif x == 'Berlin':

        x = int(3809)

    elif x == 'Lyon':

        x = int(10000)

    elif x == 'Athens':

        x = int(17040)    

    elif x == 'Warsaw':

        x = int(3372)

    elif x == 'Oporto':

        x = int(6900)    

    elif x == 'Krakow':

        x = int(2328)

    elif x == 'Copenhagen':

        x = int(4400)    

    elif x == 'Luxembourg':

        x = int(242)

    elif x == 'Zurich':

        x = int(4700)

    elif x == 'Geneva':

        x = int(12000)

    elif x == 'Ljubljana':

        x = int(1712)

    return x
data['city_density'] = data['City'].apply(city_density)
data.head()
#Создадем новый датафрейм с количеством ресторанов в каждом городе (по данным имеющегося датафрейма) 

df2 = data['City'].value_counts().rename_axis('City').to_frame(name='Rest Count')

#Добавим количество ресторанов в исходный датафрейм

data = data.merge(df2, on='City', how='left')

#Новый ранг - отношение ранга к количеству ресторанов в городе

data['New Rank'] = data['Ranking'] / data['Rest Count']
data.head()
# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
data.info()
data.sample(5)
data['Price Range'].value_counts()
#Заполняем пустые значения колонки Price Range самой распространенной ценовой категорией

data['Price Range'] = data['Price Range'].fillna('$$ - $$$')

#Заменяем значения на числа

data['Price Range'] = data['Price Range'].replace('$', '1')

data['Price Range'] = data['Price Range'].replace('$$ - $$$', '2')

data['Price Range'] = data['Price Range'].replace('$$$$', '3')
# Cuisine Style и Dummie

# Создадим признак для ресторанов где тип кухни не упомянут

data['cuisine_isna'] = pd.isna(data['Cuisine Style']).astype('uint8')
#Очистим строки от всего лишнего, включая пробелы и оставим только разделители-запятые

data['Cuisine Style'] = data.apply(lambda x: x['Cuisine Style'].replace('[','').replace(']','').replace("'",'').replace(' ','') 

                                   if type(x['Cuisine Style']) != float else x['Cuisine Style'], axis = 1)



#Разберем признак который представлен строкой, на дамми-переменные по разделителю

styles = data['Cuisine Style'].str.get_dummies(',').sum().sort_values(ascending = False)

styles_drop = [x for x in styles.index if styles[x] < 100] # изначально ограничимся только признаками которые имеют больше 100 ресторанов



#Присоединим получившийся датафрейм новых признаков 

data = data.join(data['Cuisine Style'].str.get_dummies(',').drop(styles_drop, axis = 1), how = 'left')
data.sample()
# посмотрим на содепжимое колонки:

data.Reviews[1]
#Видно, что здесь есть интересующие нас даты

#Добавляем 3 новых столбца: время между последним комментарием в датасете и последнем в ресторане,

#аналогично с предпоследним

#время между комментариями

import datetime as dt

pattern = re.compile('\d{2}/\d{2}/\d{4}')

reviews=[]

for i in data['Reviews']:

    reviews.append(re.findall(pattern, str(i)))

rev=pd.DataFrame(reviews).dropna()

rev.columns=['date1', 'date2']

rev['date1'] = pd.to_datetime(rev['date1']) 

rev['date2'] = pd.to_datetime(rev['date2']) 

rev['dd']=rev['date1']-rev['date2']

data['date1'] = rev['date1'].max() - rev['date1']

data['date1'] = data['date1'].apply(lambda x: x.days)

data['date1'] = data['date1'].fillna(0)

data['date1'] = data['date1'].apply(lambda x: int(x))

data['date2'] = rev['date1'].max() - rev['date2']

data['date2'] = data['date2'].apply(lambda x: x.days)

data['date2'] = data['date2'].fillna(0)

data['date2'] = data['date2'].apply(lambda x: int(x))

data['dd']=data['date2']-data['date1']
data.head()
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
data
data.info()

# убираем строковые признаки

data = data.drop(['Restaurant_id', 'Cuisine Style', 'Price Range', 'Reviews', 'URL_TA', 'ID_TA'], axis = 1)
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