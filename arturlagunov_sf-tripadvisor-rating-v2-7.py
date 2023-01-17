#Установим библиотеку для десериализации плохого json, стандартная плохо обрабатывает одинарные кавычки.
!pip install demjson
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json
import datetime
import math

import demjson

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
#Подготовим наборы данных, переименуем колонки.

DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/' 

def rename_columns(data_frame):
    new_column_names = {}
    for column in data_frame.columns:
        new_column_name = column.lower()
        new_column_name = new_column_name.replace(' ','_')
        new_column_names[column] = new_column_name.strip()
    new_column_names['Number of Reviews'] = 'reviews_number'
    data_frame.rename(new_column_names,axis=1,inplace=True)

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')
rename_columns(df_train)

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')
rename_columns(df_test)

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
rename_columns(df_test)

df_train.info()
df_train.head(5)
df_test.info()
df_test.head(5)
sample_submission.head(5)
sample_submission.info()
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет
df_train['sample'] = 1 # помечаем где у нас трейн
df_test['sample'] = 0 # помечаем где у нас тест
df_test['rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями

data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
data.info()
data.sample(5)
data.reviews[1]
## Начальная обработка

#Скопируем исходные данные, все преобразования будем делать с копией исходных данных.
data1 = data.copy()

#Удалим дубликаты в столбце id_ta, для дальнейшего деления данных, их всего 20.
data1.drop_duplicates(subset=['id_ta'],inplace=True)
#Создадим переменную для хранения множества признаков готовых к обработке
numeric_columns = set(['ranking','price_range','reviews_number','sample','rating'])

# Далее заполняем пропуски 0, вы можете попробовать заполнением средним или средним по городу и тд...
data1['reviews_number'].fillna(0, inplace=True)
data1.nunique(dropna=False)
data1['price_range'].value_counts()
#price_range -  качественный признак, который можно отнести к ранговой переменной, так как мы знаем, что есть
# связь между значениями по возрастанию (убыванию) значений, однако не можем точно сказать на сколько отличаются
# конкретные значения, по которым строился ранг. Заменим строковые значения выражающие ранг на числа.
def func1(x):
    if x == '$':
        return 1
    elif x == '$$ - $$$':
        return 2
    elif x == '$$$$':
        return 3
    else:
        return x

data1['price_range'] = data1['price_range'].apply(func1)

#Данные о ценовой группе содержат довольно много пропусков (13886 / 40000 * 100) 35% датафрейма.
#Заполним пропуски средним рангом по городу.

city_price_range = data1.groupby('city',as_index=False)['price_range'].mean()
city_price_range.rename({'price_range':'city_price_range'},axis=1,inplace=True)
data1 = pd.merge(data1,city_price_range,'left','city')
mean_price_range = round(data1['price_range'].mean(),0)
def func2(row):
    if pd.isna(row['price_range']):
        if row['city_price_range'] == 0:
            return mean_price_range
        else:
            return round(row['city_price_range'],0)
    else:
        return row['price_range']
data1['price_range'] = data1.apply(func2,axis=1)

data1['price_range'].value_counts()
data1.head(5)
numeric_columns.add('price_range')
# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na
cities = data1['city'].value_counts().reset_index()['index']
for city in cities:
    numeric_columns.add('city_'+city)
data1 = pd.get_dummies(data1, columns=['city'], dummy_na=True)
data1.sample(5)
# Для проверки посчитаем количество строк в исходном датафрейме, после удаления дублей по id
len(data1)
#Так как связь между рестораном и кухонными стилями один ко многим, вынесем стили в отдельный датафрейм, десериализуем вложенные массивы json.
cuisine_styles = data1[['id_ta','cuisine_style']]
cuisine_styles['cuisine_style'] = cuisine_styles['cuisine_style'].apply(lambda x: demjson.decode(x) if not pd.isna(x) else [])
cuisine_styles = cuisine_styles.set_index(['id_ta'])
cuisine_styles = cuisine_styles.explode('cuisine_style')
cuisine_styles.reset_index(inplace=True)
cuisine_styles.head(10)
data1.drop('cuisine_style',axis=1,inplace=True)
cuisine_styles.head(5)
#Вынесем отзывы в отдельный датафрейм, десериализуем вложенные массивы json.
reviews = pd.DataFrame({'id_ta':data1['id_ta'], 'review':data1['reviews']})
reviews = reviews.set_index('id_ta')
reviews['review'] = reviews['review'].apply(lambda x:demjson.decode(x.replace('nan','"nan"')) if not pd.isna(x) else [])
reviews['review'] = reviews['review'].apply(lambda x:list(zip(*x)))
reviews = reviews.explode('review')
reviews['text'] = [(x[0] if type(x) != float else math.nan) for x in reviews['review']]
reviews['date'] = [(x[1] if type(x) != float else math.nan) for x in reviews['review']]
reviews['date'] = reviews['date'].apply(lambda x: datetime.datetime.strptime(x,'%m/%d/%Y').date() if  type(x) is str else x)
reviews.drop('review',inplace=True,axis=1)
reviews.reset_index(inplace=True)
data1.drop('reviews',axis=1,inplace=True)
reviews.head(5)
#Рассчитаем количество кухонь в ресторане и добавим новый признак
cusine_style_count = cuisine_styles.groupby('id_ta',as_index=False)['cuisine_style'].count()
cusine_style_count.rename({'cuisine_style':'cusine_style_count'},inplace=True,axis=1)
#Пропуски заполним средним количеством кухонь
mean_count_cuisine_styles = cuisine_styles.fillna('none').groupby('id_ta')['cuisine_style'].count().sort_values(ascending=False).mean()
cusine_style_count['cusine_style_count'] = cusine_style_count['cusine_style_count'].apply(lambda x:round(mean_count_cuisine_styles,0) if x==0 else x)
data1 = pd.merge(data1,cusine_style_count,'left','id_ta')
numeric_columns.add('cusine_style_count')
data1.head(5)
#Добавим признак наличия вегитаринской кухни в ресторане
vegetarian = cuisine_styles[(cuisine_styles['cuisine_style']=='Vegetarian Friendly')|(cuisine_styles['cuisine_style']=='Vegan Options')]
vegetarian.rename({'cuisine_style':'vegetarian'},axis=1,inplace=True)
vegetarian.drop_duplicates(subset=['id_ta'],inplace=True)
vegetarian['vegetarian'] = vegetarian['vegetarian'].apply(lambda x: 1)
data1 = pd.merge(data1,vegetarian,'left','id_ta')
data1['vegetarian'].fillna(0,inplace=True)
numeric_columns.add('vegetarian')
data1.head(5)
#Добавим признак актуальности отзывов равный количеству дней от последнего отзыва до текущей даты
mm_reviews = reviews.groupby('id_ta')['date'].agg(['min','max']).reset_index()
mm_reviews['delta'] = mm_reviews['max'] - mm_reviews['min']
mm_reviews['relevance_of_reviews'] = mm_reviews['max'].apply(lambda max: max if  pd.isna(max)\
else (datetime.datetime.now().date() - max).days)
data1 = pd.merge(data1,mm_reviews[['id_ta','relevance_of_reviews']],'left','id_ta')
data1['relevance_of_reviews'].fillna(data1['relevance_of_reviews'].mean(),inplace=True)
numeric_columns.add('relevance_of_reviews')

data1.head(5)
result = data1[list(numeric_columns)]
result.head()
#Сверим количество строк в результирующем датафрейме с исходным количеством, так как было много операций соединения.
len(result)
plt.rcParams['figure.figsize'] = (10,7)
df_train['ranking'].hist(bins=100)
df_train['city'].value_counts(ascending=True).plot(kind='barh')
df_train['ranking'][df_train['city'] =='London'].hist(bins=100)
# посмотрим на топ 10 городов
for x in (df_train['city'].value_counts())[0:10].index:
    df_train['ranking'][df_train['city'] == x].hist(bins=100)
plt.show()
df_train['rating'].value_counts(ascending=True).plot(kind='barh')
df_train['ranking'][df_train['rating'] == 5].hist(bins=100)
df_train['ranking'][df_train['rating'] < 4].hist(bins=100)
plt.rcParams['figure.figsize'] = (30,30)
sns.heatmap(result.corr(),annot=True)
df_preproc = result
df_preproc.sample(10)
df_preproc.info()
# Теперь выделим тестовую часть
train_data = df_preproc.query('sample == 1').drop(['sample'], axis=1)
test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1)

y = train_data.rating.values            # наш таргет
X = train_data.drop(['rating'], axis=1)
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
test_data = test_data.drop(['rating'], axis=1)
sample_submission
predict_submission = model.predict(test_data)
predict_submission
sample_submission['rating'] = predict_submission
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head(10)
