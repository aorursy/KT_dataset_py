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
# функция для
# задания количества жителей в зависимости от города
# данные взяты из википедии
def City_population(x):
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
# запишем значения в столбец 'City_population'
data['City_population'] = data['City'].apply(City_population)
data['City_population']
# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na
data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
data.head(5)
data.sample(5)
data['Price Range'].value_counts()
# Ваша обработка 'Price Range'
# функция для
# замены $ на 1, $$ - $$$ на 2, $$$$ на 3
# замены пропуска на наболее часто встречающееся значение $$ - $$$, т.е. на 2
def change_price_range(x):
    if x == '$':
        x = 1
    elif x == '$$ - $$$':
        x = 2
    elif x == '$$$$':
        x = 3
    else: # если пустое значение
        x = 2
    return x
# запишем измененные значения в столбец 'Price_Range_dig'
data['Price_Range_figure'] = data['Price Range'].apply(change_price_range)
data['Price_Range_figure'].value_counts()
# тут ваш код на обработку других признаков
# .....
# поработаем с данными в столбце 'Cuisine Style'

# заполним пустые значения самой популярной кухней 'European'
# ('Vegetarian Friendly' более популярна, но маловероятно, что в ресторане только вегетарианская кухня)
data['Cuisine Style'] = data['Cuisine Style'].fillna(value="['European']")

# выполним очистку от лишних символов
data['Cuisine Style'] = data['Cuisine Style'].apply(lambda x: x.replace('[', ''))
data['Cuisine Style'] = data['Cuisine Style'].apply(lambda x: x.replace(']', ''))

# подсчитаем кол-во кухонь в каждом ресторане 
# и запишем эту информацию в столбец 'Cuisine_Style_quantity'
data['Cuisine_Style_quantity'] = 0
n = 0
for i in data['Cuisine Style']:
    m = 0
    for j in i.split(', '):
        m += 1
    data['Cuisine_Style_quantity'][n] = m
    n += 1
# поработаем с данными в столбце 'Reviews'

# достаточно много ресторанов без отзывов. введем столбец, где 1 - есть отзывы, 0 - не отзывов.

data['Reviews_quantity'] = data['Reviews'].apply(lambda x: 0 if (x=='[[], []]') else 1)
data['Reviews_quantity'].unique()
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
data.info()
# убираем строковые признаки
data = data.drop(['Restaurant_id', 'Cuisine Style', 'Price Range', 'Reviews', 'URL_TA', 'ID_TA'], axis = 1)
data.sample()
data.info()
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
