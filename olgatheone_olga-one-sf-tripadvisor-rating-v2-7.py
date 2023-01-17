# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline

from collections import Counter
import collections as co
from datetime import datetime, timedelta

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
#Переименовываю, чтобы не исправлять свои имена в коде
df = data.copy()
data.info()
df.sample(5)
df.rename(
    columns={'Price Range': 'price_range', 'Cuisine Style': 'cuisine_style', 'Number of Reviews': 'n_reviews'}, inplace=True)
# n_reviews заменили пропуски средним по колонке
df['n_reviews'].fillna(df['n_reviews'].mean(), inplace=True)

# Пропуски в price_range заполним самым частым значением
df['price_range'].fillna(df['price_range'].mode()[0], inplace=True)
# n_reviews заменили пропуски средним по колонке
df['Reviews'].fillna('[[], []]', inplace=True)
df['Reviews'].head()
# формируется колонка со списком дат отзывов
df['Reviews1'] = df['Reviews'].apply(lambda x: x.split(", ['", 1)[-1][0:-2])
# формируется колонка с максимальной датой
def custom_r(r):
    maxi = datetime.strptime('01/01/1000', '%m/%d/%Y')
    ss = r.replace("'", '')
    if (ss != "") and (ss != '[[], ['):
        dd = ss.split(',')
        for yy in dd:
            jj = datetime.strptime(yy.strip(), '%m/%d/%Y')
            if maxi < jj:
                maxi = jj
    return maxi


df['Reviews2'] = df.apply(lambda x: custom_r(x['Reviews1']), axis=1)
# формируется колонка с минимальной датой
def custom_r(r):
    mini = datetime.strptime('01/01/3000', '%m/%d/%Y')
    ss = r.replace("'", '')
    if (ss != "") and (ss != '[[], ['):
        dd = ss.split(',')
        for yy in dd:
            jj = datetime.strptime(yy.strip(), '%m/%d/%Y')
            if jj < mini:
                mini = jj
    return mini


df['Reviews3'] = df.apply(lambda x: custom_r(x['Reviews1']), axis=1)
# формируется колонка с разницей между максимаьной и минимаьной датой отзыва
df['Reviews4'] = df['Reviews2']-df['Reviews3']
# формируется колонка с разницей между максимаьной и минимаьной датой отзыва в числовом формате
df['Reviews4'] = df.apply(lambda x: x.Reviews4.days, axis=1)
# убираю вспомогательные значения
df['Reviews2'] = df.apply(lambda x: None if x.Reviews2.strftime(
    "%Y") == '1000' else x.Reviews2, axis=1)
# убираю вспомогательные значения
df['Reviews3'] = df.apply(lambda x: None if x.Reviews3.strftime(
    "%Y") == '3000' else x.Reviews3, axis=1)
# Заменила пропуски на выборочный элеент (пропусков очень мало)
df['Reviews2'].fillna(df['Reviews2'][1], inplace=True)
df['Reviews3'].fillna(df['Reviews3'][1], inplace=True)
# формируется колонка с максимальной датой в числовом формате
df['Reviews5'] = df.apply(lambda x: float(x.Reviews2.asm8), axis=1)
# формируется колонка с минимальной датой в числовом формате
df['Reviews6'] = df.apply(lambda x: float(x.Reviews3.asm8), axis=1)
df.describe()
# City - оставим 10 часто встречающихся городов, остальные города заменем на 'other'
all_cities = df.City.value_counts().index
top_cities = list(df.City.value_counts().sort_values(
    ascending=False).head(11).index)
cities_to_throw_away = list(set(all_cities) - set(top_cities))
df.loc[df['City'].isin(cities_to_throw_away), 'City'] = 'other'
# создали переменные dummies
dum = pd.get_dummies(df.City, drop_first=True)
# присоединили к данным
df = pd.concat([df, dum], axis=1)
# убрали лишние символы, а пропуски теперь выглядят, как строка 'a'
df['cuisine_style'] = df['cuisine_style'].map(lambda x: str(x)[1:][:-1])
# составили список кухонь
cuisine_list = []
for some in df['cuisine_style']:
    for in_list in some.split(', '):
        if in_list != 'a':
            in_list = in_list[1:-1]
        cuisine_list.append(in_list)
# cuisine_list
# посчитали количество упоминаний каждой кухни (пропуск 'a' считаем кухней), пусть это тоже буднет рейтинг
list_rating = co.Counter(cuisine_list).most_common()
# перевели в формат = лист в листе
res = [list(ele) for ele in list_rating]
# посчитали количество упоминаний каждой кухни, первые 10
list_rating_dum = co.Counter(cuisine_list).most_common(10)
# перевели в формат = лист в листе
res_dum = [list(ele) for ele in list_rating_dum]
# формируем новый признак cuisine_rating_dum для переменной
def cuisine_rating_dum(rating):
    num = ''
    list_x = rating.replace("'", '')
    list_x = list_x.split(',')
    for yy in list_x:
        yy = yy.strip()
        result = [element for element in res_dum if element[0] == yy]
        if result == []:
            num = 'other'
        else:
            num = result[0][0]
            break
    return num


df['cuisine_rating_dum'] = df.apply(
    lambda x: cuisine_rating_dum(x['cuisine_style']), axis=1)
# создали переменные dum из cuisine_rating_dum
dum = pd.get_dummies(df.cuisine_rating_dum)
# присоединили к данным
df = pd.concat([df, dum], axis=1)
def cuisine_rating(rating):
    num = 0
    list_x = rating.replace("'", '')
    list_x = list_x.split(',')
    for yy in list_x:
        yy=yy.strip()
        result = [element for element in res if element[0] == yy]
        num+=result[0][1]
    return num

df['cuisine_rating']=df.apply(
    lambda x: cuisine_rating(x['cuisine_style']), axis=1)
# добавили новый признак price_range_num
def price_range_num(price_range):
    if price_range == '$':
        return 1
    elif price_range == '$$ - $$$':
        return 2
    elif price_range == '$$$$':
        return 3
    
df['price_range_num'] = df['price_range'].apply(price_range_num)
# создали переменные
dum = pd.get_dummies(df.price_range)
# присоединили к данным
df = pd.concat([df, dum], axis=1)
df['n_reviews'].value_counts().hist(bins=100)
#есть выбросы
sns.boxplot(df['n_reviews'])
IQR = df.n_reviews.quantile(0.75) - df.n_reviews.quantile(0.25)
perc25 = df.n_reviews.quantile(0.25)
perc75 = df.n_reviews.quantile(0.75)
print("Границы выбросов absences: [{f}, {l}].".format(
    f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR))
# сделаем новый признак n_reviews_bool
df['n_reviews_bool'] = df.n_reviews.between(
    perc25 - 1.5*IQR, perc75 + 1.5*IQR)
# сделаем новый признак n_reviews_new
xmean = df['n_reviews'].mean()
df['n_reviews_new'] = df.apply(
    lambda x: x.n_reviews if x.n_reviews_bool == True else xmean, axis=1)
# преобразуем новый признак n_reviews_bool в числовой
def n_reviews_bool(booll):
    if booll == True:
        return 1.0
    else:
        return 2.0

df['n_reviews_bool'] = df['n_reviews_bool'].apply(n_reviews_bool)
df.describe()
# Убираем нечисловые признаки
df = df.drop(['City', 'cuisine_rating_dum', 'cuisine_style', 'price_range',
              'Reviews', 'Reviews1', 'Reviews2', 'Reviews3', 'URL_TA', 'ID_TA'], axis=1)
#уберу Ranking, Reviews6, other
df = df.drop(['Ranking','other','Reviews6'], axis=1)
df['Rating'].value_counts(ascending=True).plot(kind='bar')
df.info()
df.drop(['sample'], axis=1).corr()
df = df.drop(['Restaurant_id'], axis=1)
# Из подготовленного датафрейма формируем копию и на ней работаем
df_preproc = df.copy()
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
feat_importances.nlargest(15).plot(kind='barh')
test_data.sample(10)
test_data = test_data.drop(['Rating'], axis=1)
predict_submission = model.predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head(10)
