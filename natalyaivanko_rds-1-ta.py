import pandas as pd

import re

import datetime as dt

import numpy as np

from sklearn.model_selection import train_test_split

# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!

RANDOM_SEED = 42

# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

!pip freeze > requirements.txt

# Импортируем необходимые библиотеки:

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')



# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



df = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
# заменяю название колонок

df = df.rename(columns={'Cuisine Style':'Cuisine_Style','Price Range':'Price_Range','Number of Reviews': 'Number_of_Reviews'})
# обработка пропусков

df['Number_of_Reviews_isNAN'] = pd.isna(df['Number_of_Reviews']).astype('uint8')

df['Cuisine_Style_isNAN'] = pd.isna(df['Cuisine_Style']).astype('uint8')

df['Price_Range_isNAN'] = pd.isna(df['Price_Range']).astype('uint8')
# проставляю численность жителей в городах

citizen=pd.DataFrame({

    'City':['Amsterdam', 'Athens', 'Barcelona', 'Berlin', 'Bratislava', 'Brussels', 'Budapest', 'Copenhagen', 'Dublin', 'Edinburgh', 'Geneva', 'Hamburg', 'Helsinki', 'Krakow', 'Lisbon', 'Ljubljana', 'London', 'Luxembourg', 'Lyon', 'Madrid', 'Milan', 'Munich', 'Oporto', 'Oslo', 'Paris', 'Prague', 'Rome', 'Stockholm', 'Vienna', 'Warsaw', 'Zurich'],

    'Citizen':[860,656,1620,3601,424,1199,1750,616,553,430,495,1831,643,767,506,342,8788,602,516,3223,1366,1456,250,673,2141,1281,2873,962,1841,1758,403]})

df=df.merge(citizen, on='City', how='left')

df = df.rename(columns={'Citizen_x': 'Citizen'})



# Number of Reviews - заполняю пустые значения

df['Number_of_Reviews']=df['Number_of_Reviews'].fillna(0)



# пересчитываю кол-во отзывов пропорционально кол-ву жителей

df['Number_of_Reviews_cit']=df['Number_of_Reviews']/df['Citizen']
# ввожу признак столица = 1, не столица = 0

capital=['Amsterdam', 'Athens', 'Berlin', 'Bratislava', 'Brussels', 'Budapest', 'Copenhagen', 'Dublin', 'Edinburgh', 'Helsinki', 'Lisbon', 'Ljubljana', 'London', 'Luxembourg', 'Madrid', 'Oslo', 'Paris', 'Prague', 'Rome', 'Stockholm', 'Vienna', 'Warsaw']

df['Capital']=df['City'].apply(lambda x: 1 if x in(capital) else 0)
# пересчитываю рэнкинг относительно максимального значения в каждом городе

df['Ranking_n'] = df['Ranking'] / df['City'].map(df.groupby(['City'])['Ranking'].max())
# подсчитываю количество типов кухонь, если не заполнено, пишу 2(медиана)

df['Cuisine_Count']=df['Cuisine_Style'].apply(lambda x: 2 if type(x)==float else len(x.split(',')))
# Ranking на 8 групп относительно срезов в городе (ухудшает)

# ran=pd.DataFrame(columns=['Q25', 'Median', 'Q75', 'Max'])

# ran['Q25']=df.groupby(['City'])['Ranking'].quantile(0.25)

# ran['Median']=df.groupby(['City'])['Ranking'].median()

# ran['Q75']=df.groupby(['City'])['Ranking'].quantile(0.75)

# ran['Max']=df.groupby(['City'])['Ranking'].max()

# df = df.merge(ran, on='City', how='left')

# df['Ranking_nn']=0



# conditions = [

#     df['Ranking'] < df['Q25']/2, 

#     (df['Ranking'] >= df['Q25']/2)&(df['Ranking'] < df['Q25']),

#     (df['Ranking'] >= df['Q25'])&(df['Ranking'] < (df['Median']-df['Q25'])/2+df['Q25']),

#     (df['Ranking'] >= (df['Median']-df['Q25'])/2+df['Q25'])&(df['Ranking'] < df['Median']),

#     (df['Ranking'] >= df['Median'])&(df['Ranking']<(df['Q75']-df['Median'])/2+df['Median']),

#     (df['Ranking'] >= (df['Q75']-df['Median'])/2+df['Median'])&(df['Ranking'] < df['Q75']),

#     (df['Ranking'] >= df['Q75'])&(df['Ranking']<(df['Max']-df['Q75'])/2+df['Q75']),

#     df['Ranking'] >= (df['Max']-df['Q75'])/2+df['Q75']]



# choices = [1,2,3,4,5,6,7,8]



# df['Ranking_nn'] = np.select(conditions, choices, default=np.nan)

# df = df.drop(['Q25', 'Median', 'Q75', 'Max'], axis = 1)

# подсчитываю количество дней между датами в отзывах

pattern = re.compile('\d{2}/\d{2}/\d{4}')

def list_of_dates(Reviews):

    dates = []

    if type(Reviews) is str:

        dates = pattern.findall(Reviews)

    return dates

df['dates'] = df['Reviews'].apply(list_of_dates)

df['dd'] = df['dates'].apply(lambda x: abs((pd.to_datetime(str(x[0]))-pd.to_datetime(str(x[1]))).days) if len(x) == 2 else 0)

df = df.drop(['dates'], axis = 1)
# проставляю признак принадлежности к сети из id

df['Net_c']=df['Restaurant_id'].map(df.groupby(['Restaurant_id'])['Restaurant_id'].count())

df['Net']=df['Net_c'].apply(lambda x: 1 if x>1 else 0)

df = df.drop(['Net_c'], axis = 1)
# Price Range -  преревожу диапазон в числовое обозначение и рассчитываю средний уровень

def pr(rec):

    if rec == '$':

        return 0

    elif rec == '$$ - $$$':

        return 1

    elif rec == '$$$$':

        return 2

    else:

        return 1

df['Price_Group']=df['Price_Range'].apply(pr)

df['Price_m'] = df['City'].map(df.groupby('City')['Price_Group'].mean())
# пример из baseline - разнесла города и ценовой диапазон в отдельные колонки

df = pd.get_dummies(df, columns=['City'], dummy_na=True)

df = pd.get_dummies(df, columns=['Price_Group'], dummy_na=True)
# Кухни - вынесла в отдельные колонки признак наличия трех топовых типов кухонь

def cu1(rec):

    if type(rec)==float:

        return 0

    elif 'Vegetarian Friendly' in rec:

        return 1

    else:

        return 0

df['Veget']=df['Cuisine_Style'].apply(cu1)

def cu2(rec):

    if type(rec)==float:

        return 0

    elif 'European' in rec:

        return 1

    else:

        return 0

df['Eur']=df['Cuisine_Style'].apply(cu2)

def cu3(rec):

    if type(rec)==float:

        return 0

    elif 'Mediterranean' in rec:

        return 1

    else:

        return 0

df['Med']=df['Cuisine_Style'].apply(cu3)

def cu4(rec):

    if type(rec)==float:

        return 0

    elif 'Vegetarian Friendly' in rec or 'European' in rec or 'Mediterranean'in rec:

        return 0

    else:

        return 1

df['Oth']=df['Cuisine_Style'].apply(cu4)
# убираю ненужные для модели признаки

df = df.drop(['Restaurant_id', 'Cuisine_Style', 'Price_Range', 'Reviews', 'URL_TA', 'ID_TA'], axis = 1)
# Теперь выделим тестовую часть

train_df = df.query('sample == 1').drop(['sample'], axis=1)



y = train_df.Rating.values            # наш таргет

X = train_df.drop(['Rating'], axis=1)



# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
# Создаём модель (НАСТРОЙКИ НЕ ТРОГАЕМ)

model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)
model.fit(X_train, y_train)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

y_pred = model.predict(X_test)
# округляю полученные значения рейтингов

def round_d(rec):

    if rec <0.25:

        return 0

    elif 0.25<=rec<0.75:

        return 0.5

    elif 0.75<=rec<1.25:

        return 1

    elif 1.25<=rec<1.75:

        return 1.5

    elif 1.75<=rec<2.25:

        return 2

    elif 2.25<=rec<2.75:

        return 2.5

    elif 2.75<=rec<3.25:

        return 3

    elif 3.25<=rec<3.75:

        return 3.5

    elif 3.75<=rec<4.25:

        return 4

    elif 4.25<=rec<4.75:

        return 4.5

    else:

        return 5

    

for i in range(y_pred.size):

    y_pred[i]=round_d(y_pred[i])
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# Обучаю модель на всех исходных данных

model.fit(X, y)



# Запускаю на данных по заданию

test_df = df.query('sample == 0').drop(['sample'], axis=1)

test_df = test_df.drop(['Rating'], axis=1)



predict_submission = model.predict(test_df)



# округляю полученные значения рейтингов

def round_d(rec):

    if rec <0.25:

        return 0

    elif 0.25<=rec<0.75:

        return 0.5

    elif 0.75<=rec<1.25:

        return 1

    elif 1.25<=rec<1.75:

        return 1.5

    elif 1.75<=rec<2.25:

        return 2

    elif 2.25<=rec<2.75:

        return 2.5

    elif 2.75<=rec<3.25:

        return 3

    elif 3.25<=rec<3.75:

        return 3.5

    elif 3.75<=rec<4.25:

        return 4

    elif 4.25<=rec<4.75:

        return 4.5

    else:

        return 5

    

sample_submission['Rating'] = predict_submission

sample_submission['Rating']=sample_submission['Rating'].apply(round_d)



# выгружаю в файл

sample_submission.to_csv('submission.csv', index=False)