# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



from sklearn.preprocessing import LabelEncoder



from collections import Counter

import re

from datetime import datetime



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

DATA_DIR1 = '/kaggle/input/europe-datasets/'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')

df_europe_median_income = pd.read_csv(DATA_DIR1+'/median_income_2016.csv')

df_europe_life_satisfaction = pd.read_csv(DATA_DIR1+'/life_satisfaction_2013.csv')

df_europe_leisure_satisfaction = pd.read_csv(DATA_DIR1+'/leisure_satisfaction_2013.csv')
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
data.columns
# переименуем столбцы для удобства обращения к ним

data.columns = ['restaurant_id', 'city', 'cuisine_style', 'ranking', 'price_range', 'number_of_reviews', 'reviews', 'URL_TA', 'ID_TA','sample','Rating']
# Для примера я возьму столбец Number of Reviews

data['Number_of_Reviews_isNAN'] = pd.isna(data['number_of_reviews']).astype('uint8')
data['Number_of_Reviews_isNAN']
# заполним NaN средним значением по городу

data.loc[data['number_of_reviews'].isnull(),['number_of_reviews']] = data.groupby(['city'])['number_of_reviews'].transform('mean')
# посмотрим на столбец cuisine_style

# в нем много пустых значений, но если ресторан работает, значит он предлагает какую-то кухню - "Other"

data['cuisine_style_isNAN'] = pd.isna(data['cuisine_style']).astype('uint8')

data['cuisine_style'] = data['cuisine_style'].fillna('Other')
# посмотрим на столбец price_range

data['price_range_isNAN'] = pd.isna(data['price_range']).astype('uint8')

data['price_range'] = data['price_range'].fillna('Other')
# заполним два пустых значения 'reviews'

data['reviews'] = data['reviews'].fillna("[[], []]")
# убедимся, что больше пустых ячеек нет

data.info()
data.nunique(dropna=False)
# создадим фнкцию для добавления новой колонки из словаря

# """функция для добавления новой колонки из словаря"""

def new_column(x,y):

    for item in y:

        if x == item:

            result = y[item]   

    return result
# создадим список уникальных названий городов

city_names = data.city.value_counts().index.to_list()



# добавим в датасет число туристов для каждого города

tourists = [19233000,17560200,5440100,6714500,5959400,6481300,10065400,8948600,3539400,6410300,8354200,3942000,1450000,4066600,6000000,2604600,3822800,2850000,5213400,3069700,5728400,1660000,2240000,2341300,1150000,2732000,2140000,1240000,1290056,1139000,1127904]

city_tourists = dict(zip(city_names,tourists))

data['city_tourists'] = data.city.apply(lambda x: new_column(x, city_tourists))



# добавим в датасет население для каждого города

population = [8787892,2140000,3223334,1620343,3601131,1366180,2872800,1280508,506654,1840573,859732,1198726,1830584,1456039,515695,961609,1749734,1758143,553165,615993,655780,482005,402762,214349,499480,766739,673469,643272,424428,613894,279631]

city_population = dict(zip(city_names,population))

data['city_population'] = data.city.apply(lambda x: new_column(x, city_population))



# добавим в датасет население + туристы для каждого города

data['city_people'] = data['city_tourists'] + data['city_population']



# создадим столбец отношения рэнкинга к числу туристов в городе

data['ranking_ratio'] = data['ranking']/data['city_tourists'] 



# создадим столбец - количество ресторанов в городе

number_of_restaurants = data.groupby(['city'])['restaurant_id'].count().reset_index()

number_of_restaurants = dict(zip(number_of_restaurants['city'].to_list(),number_of_restaurants['restaurant_id'].to_list()))

data['number_of_restaurants'] = data.city.apply(lambda x: new_column(x, number_of_restaurants))



# создадим столбец отношения рэнкинга к числу ресторанов в городе

data['ranking_rest_ratio'] = data['ranking']/data['number_of_restaurants']



# создадим столбец отношения числа отзывов к числу туристов в городе

data['number_of_reviews_rest_ratio'] = data['number_of_reviews']/data['city_tourists'] 



# создадим столбец с количеством ресторанов со звездой Micheline в каждом городе

city_names = data.city.value_counts().index.to_list()

micheline_stars = [66,118,15,20,16,15,15,2,8,15,11,21,11,13,20,13,6,2,2,11,4,4,11,5,11,1,4,6,0,12,6]

city_micheline_stars = dict(zip(city_names,micheline_stars))

data['micheline'] = data.city.apply(lambda x: new_column(x, city_micheline_stars))/data['number_of_restaurants']

# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

#data = pd.get_dummies(data, columns=[ 'city',], dummy_na=True)
data.head(5)
data.sample(5)
data['price_range'].value_counts()
# обработаем признак 'price_range'

encoder = LabelEncoder()

data['price_range'] = encoder.fit_transform(data['price_range'].dropna())
# cоздадим dummy-переменные для 'price_range'

#data = pd.get_dummies(data, columns=[ 'price_range',], dummy_na=True)
# обработка признака 'cuisine_style'



# создадим признак, показывающий количество кухонь, предлагаемых в каждом ресторане

def column_clean(s):

    #"""функция для "очистки" колонки от лишних симоволов """

    result = str(s)

    result = result[1:-1].split(',')

    return result



# удалим лишние кавычки и заменим кириллицу на латинницу



data['cuisine_style'] = data['cuisine_style'].map(

    lambda x: x.lstrip("'").rstrip("'"))

data['cuisine_style'] = data['cuisine_style'].map(

    lambda x: x.replace('е', 'e').replace('а', 'a'))

data['cuisine_style'] = data['cuisine_style'].map(

    lambda x: x.lstrip("[").rstrip("]"))



cuisine_style = data['cuisine_style'].apply(column_clean)



# создадим список уникальных кухонь, которые встречаются в датафрейме

cuisine_list = []

for item in cuisine_style:

    for name in item:

        name = name.lstrip(" '").rstrip("' ")

        if name not in cuisine_list and name!='the':

            cuisine_list.append(name)



# посчитаем сколько кухонь предлагает каждый ресторан

cuisine_count = [len(item) if item is not None else 1 for item in cuisine_style]



# создадим новый признак "количество кухонь" 

data['cuisine_amount'] = cuisine_count
# cоздадим dummy-переменные для 'cuisine_style'

#dummies = data['cuisine_style'].str.get_dummies(sep=',')

#data = pd.concat([data,dummies], axis=1)
# обработка признака 'review'



# разделим данные в столбце на два списка

data[['review_text','review_date']] = pd.DataFrame(data['reviews'].str.split("],",1).to_list())

data[['review_text']] = data['review_text'].apply(lambda x: x[2:])

data[['review_date']] = data['review_date'].apply(lambda x: x[2:-2])



# попробуем проанализировать текст отзывов

words = {'will come back':4,"WOW":4,"fantastic":4,"Fantastic":4,"delightful":3,"great":3,"Great":3,"Delightful":3,"yummi":3,"Yummi":3,"nice":2,"nice":2,'good':1, 'Good':1,"Average":0,"average":0,"bad":-1,'Bad':-1,"disgusting":-2,"avoid":-3,"Avoid":-3,"worst":-4,"Worst":-4}

import re



def review_analysis_column(x,y):

    result = 0

    for word in y:

        if re.search(r'\b%s\b' % word, x) is not None:

            result = y[word]

    return result



data['review_analysis'] = data.review_text.apply(lambda x: review_analysis_column(x, words))

data['review_analysis'].value_counts()
# разделим даты отзывов на два признака

data[['first_review_date','second_review_date']] = pd.DataFrame(data['review_date'].str.split(", ",1).to_list())



#заполним пропуски 

data[['second_review_date']] = data['second_review_date'].fillna('0')



# переведем даты в формат datetime





def date_time(s):

    #"""функция для перевода в формат datetime"""

    date1 = re.search(r'\d{2}/\d{2}/\d{4}',s)

    if date1 is not None:

        return (datetime.strptime(date1.group(),'%m/%d/%Y').date())

    else:

        return None



#создадим столбец 'time_difference' - разница между отзывами    

data[['first_review_date']] = data['first_review_date'].apply(date_time)

data[['second_review_date']] = data['second_review_date'].apply(date_time)

time_difference = abs(data['first_review_date'] - data['second_review_date'])

data['time_difference'] = time_difference.dropna().apply(lambda x: abs(np.timedelta64(x).astype(int)))

data[['time_difference']] = data['time_difference'].fillna(0)



# создадим новый признак "text_review_len"

data['review_text_len'] = data['review_text'].apply(lambda x: len(x))
#переведем столбец ID_TA в цифровой формат

data[['ID_TA']] = data['ID_TA'].dropna().apply(lambda x: int(x[1:]))

data[['ID_TA']] = data['ID_TA'].fillna(0)                                      
data.info()
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

sns.heatmap(data.drop(['sample'], axis=1).corr(),annot = True)
# на всякий случай, заново подгружаем данные

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем

data.info()
def new_column(x, y):

    '''функция для создания новой колонки'''

    for item in y:

        if x == item:

            result = y[item]

    return result





def column_clean(s):

    '''функция для очистки колонки'''

    result = str(s[1:-1].split(','))

    return result





def review_analysis_column(x, y):

    '''функция для анализа текста'''

    result = 0

    for word in y:

        if re.search(r'\b%s\b' % word, x) is not None:

            result = y[word]

    return result





def date_time(s):

    '''функция для перевода в формат datetime'''

    date1 = re.search(r'\d{2}/\d{2}/\d{4}', s)

    if date1 is not None:

        return (datetime.strptime(date1.group(), '%m/%d/%Y').date())

    else:

        return None

    

    

def preproc_data(df_input):

    '''includes several functions to pre-process the predictor data.'''

    df_output = df_input.copy()

    

    ################### 1. Предобработка ############################################################## 

    

    # переименуем столбцы для удобства обращения к ним

    df_output.columns = ['restaurant_id', 'city', 'cuisine_style', 'ranking', 'price_range', 'number_of_reviews', 'reviews', 'URL_TA', 'ID_TA','sample','Rating']

    # убираем ненужный для модели признак

    df_output.drop(['URL_TA'], axis = 1, inplace=True)

    

    ################### 2. NAN ##############################################################

        

    # Далее заполняем пропуски, вы можете попробовать заполнением средним или средним по городу и тд...

    df_output['Number_of_Reviews_isNAN'] = pd.isna(df_output['number_of_reviews']).astype('uint8')

    # заполним пустые ячейки в столбце 'number_of_reviews' средним значением по городу

    df_output.loc[df_output['number_of_reviews'].isnull(),['number_of_reviews']] = df_output.groupby(['city'])['number_of_reviews'].transform('mean')

    # заполним пустые ячейки в столбце 'price_range' значением "Other"

    df_output['price_range_isNAN'] = pd.isna(df_output['price_range']).astype('uint8')

    df_output['price_range'] = df_output['price_range'].fillna('Other')

    # заполним пустые ячейки в столбце 'cuisine_style' значением "Other"

    df_output['cuisine_style_isNAN'] = pd.isna(df_output['cuisine_style']).astype('uint8')

    df_output['cuisine_style'] = df_output['cuisine_style'].fillna('Other')

    # заполним пустые ячейки в столбце 'reviews' значением "[[], []]"

    df_output['reviews'] = df_output['reviews'].fillna("[[], []]")



    ################### 3. Feature Engineering ####################################################

    

    # создадим новые признаки, исходя из числа туристов, населения количества и качества ресторанов

    

    # создадим список уникальных названий городов

    city_names = df_output.city.value_counts().index.to_list()

    # добавим в датасет число туристов для каждого города

    tourists = [19233000,17560200,5440100,6714500,5959400,6481300,10065400,8948600,3539400,6410300,8354200,3942000,1450000,4066600,6000000,2604600,3822800,2850000,5213400,3069700,5728400,1660000,2240000,2341300,1150000,2732000,2140000,1240000,1290056,1139000,1127904]

    city_tourists = dict(zip(city_names,tourists))

    df_output['city_tourists'] = df_output.city.apply(lambda x: new_column(x, city_tourists))

    # добавим в датасет население для каждого города

    population = [8787892,2140000,3223334,1620343,3601131,1366180,2872800,1280508,506654,1840573,859732,1198726,1830584,1456039,515695,961609,1749734,1758143,553165,615993,655780,482005,402762,214349,499480,766739,673469,643272,424428,613894,279631]

    city_population = dict(zip(city_names,population))

    df_output['city_population'] = df_output.city.apply(lambda x: new_column(x, city_population))

    # добавим в датасет общее число жителей + туристы

    df_output['city_people'] = df_output['city_population'] + df_output['city_tourists']

    # создадим столбец отношения рэнкинга к числу туристов в городе

    df_output['ranking_ratio'] = df_output['ranking']/df_output['city_tourists']

    # создадим столбец - количество ресторанов в городе

    number_of_restaurants = df_output.groupby(['city'])['restaurant_id'].count().reset_index()

    number_of_restaurants = dict(zip(number_of_restaurants['city'].to_list(),number_of_restaurants['restaurant_id'].to_list()))

    df_output['number_of_restaurants'] = df_output.city.apply(lambda x: new_column(x, number_of_restaurants))

    # создадим столбец отношения рэнкинга к числу ресторанов в городе

    df_output['ranking_rest_ratio'] = df_output['ranking']/df_output['number_of_restaurants']

    # создадим столбец отношения числа отзывов к числу туристов в городе

    df_output['number_of_reviews_rest_ratio'] = df_output['number_of_reviews']/df_output['city_tourists'] 

    # создадим столбец отношения ресторанов со звездой Micheline к общему числу ресторанов

    city_names = df_output.city.value_counts().index.to_list()

    micheline_stars = [66,118,15,20,16,15,15,2,8,15,11,21,11,13,20,13,6,2,2,11,4,4,11,5,11,1,4,6,0,12,6]

    city_micheline_stars = dict(zip(city_names,micheline_stars))

    df_output['micheline_ratio'] = df_output.city.apply(lambda x: new_column(x, city_micheline_stars))/df_output['number_of_restaurants']

    

    # извлечем новые признаки из стоблца 'cuisine_style'

    

    # создадим признак, показывающий количество кухонь, предлагаемых в каждом ресторане

    # удалим лишние кавычки и заменим кириллицу на латинницу

    df_output['cuisine_style'] = df_output['cuisine_style'].map(

                                lambda x: x.lstrip("'").rstrip("'"))

    df_output['cuisine_style'] = df_output['cuisine_style'].map(

                                lambda x: x.replace('е', 'e').replace('а', 'a'))

    df_output['cuisine_style'] = df_output['cuisine_style'].map(

                                lambda x: x.lstrip("[").rstrip("]"))

    cuisine_style = df_output['cuisine_style'].apply(column_clean)

    # создадим список уникальных кухонь, которые встречаются в датафрейме

    cuisine_list = []

    for item in cuisine_style:

        for name in item:

            name = name.lstrip(" '").rstrip("' ")

            if name not in cuisine_list and name!='the':

                cuisine_list.append(name)

    # посчитаем сколько кухонь предлагает каждый ресторан

    cuisine_count = [len(item) if item is not None else 1 for item in cuisine_style]

    # создадим новый признак "количество кухонь" 

    df_output['cuisine_amount'] = cuisine_count

    

    # извлечем новые признаки из стоблца 'reviews'

    

    # разделим данные в столбце на два - текст отзыва и даты отзыва

    df_output[['review_text','review_date']] = pd.DataFrame(df_output['reviews'].str.split("],",1).to_list())

    df_output[['review_text']] = df_output['review_text'].apply(lambda x: x[2:])

    df_output[['review_date']] = df_output['review_date'].apply(lambda x: x[2:-2])

    # попробуем создать новый признак на основе того какие слова встречаются в тексте

    words = words = {'will come back':4,"delightful":3,"Delightful":3,"nice":2,"nice":2,'good':1, 'Good':1,"Average":0,"average":0,"bad":-1,'Bad':-1,"disgusting":-2,

                     "avoid":-3,"Avoid":-3,"worst":-4,"Worst":-4}

    df_output['review_analysis'] = df_output.review_text.apply(lambda x: review_analysis_column(x, words))

    # создадим новый признак количество символов в отзыве "text_review_len"

    df_output['review_text_len'] = df_output['review_text'].apply(lambda x: len(x))

    

    # разделим даты отзывов на два признака

    df_output[['first_review_date','second_review_date']] = pd.DataFrame(df_output['review_date'].str.split(", ",1).to_list())

    # переведем даты в формат datetime

    df_output[['first_review_date']] = df_output['first_review_date'].apply(date_time)

    df_output[['second_review_date']] = df_output['second_review_date'].dropna().apply(date_time)

    # создадим столбец 'time_difference' - временная разница между отзывами    

    time_difference = abs(df_output['first_review_date'] - df_output['second_review_date'])

    df_output['time_difference'] = time_difference.dropna().apply(lambda x: abs(np.timedelta64(x).astype(float)))

    df_output[['time_difference']] = df_output['time_difference'].fillna(0)

       

    # переведем столбец 'ID_TA' в цифровой формат

    df_output[['ID_TA']] = df_output['ID_TA'].dropna().apply(lambda x: int(x[1:]))

    df_output[['ID_TA']] = df_output['ID_TA'].fillna(0)                                      



    ################### 3. Encoding ##############################################################

    # закодируем признак 'city'

    df_output = pd.get_dummies(df_output, columns=[ 'city',], dummy_na=True)

    # закодируем признак 'price_range'

    encoder = LabelEncoder()

    df_output['price_range'] = encoder.fit_transform( df_output['price_range'].dropna())

    df_output = pd.get_dummies(df_output, columns=[ 'price_range',], dummy_na=True)

    # закодируем признак 'cuisine_style'

    dummies = df_output['cuisine_style'].str.get_dummies(sep=',')

    df_output = pd.concat([df_output,dummies], axis=1)

        

    ################### 5. Clean #################################################### 

    # убираем признаки, которые ухудшают качество предсказания

    df_output.drop(['number_of_restaurants','city_tourists','city_population'],axis = 1,inplace=True)

    

    # модель на признаках с dtypes "object" обучаться не будет, просто выберим их и удалим

    object_columns = [s for s in df_output.columns if df_output[s].dtypes == 'object']

    df_output.drop(object_columns, axis = 1, inplace=True)

    

    return df_output
df_preproc = preproc_data(data)

df_preproc.sample(10)
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

y_pred = [np.round(x*2)/2 for x in y_pred]
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

predict_submission = [np.round(x*2)/2 for x in predict_submission]
predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)