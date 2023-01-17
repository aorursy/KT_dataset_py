import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#библиотеки

import matplotlib.pyplot as plt

%matplotlib inline
#инициализация

train_ds = 'main_task.csv'

kaggle_ds = 'kaggle_task.csv'

#загрузка данных

#спасибо коллегам, принято решение не сохранять модель и использовать ее на проверочном датасете, а объединить

#датасеты с последующим разделением при обучении& 

#объединение датасетов позволит иметь гораздо больший набор данных и не иметь проблем с количеством признаков

mf = pd.read_csv(train_ds)

kf = pd.read_csv(kaggle_ds)

# Ваш код по очистке данных и генерации новых признаков

# При необходимости добавьте ячейки

mf.head()
#Анализ тренировочного ДС

mf.info()
#Анализ проверочного ДС

kf.info()
# В целом, первичный анализ можно провести непосредственно на самой платформе, используя визуализацию датасетов

# Нам нужно получить одинаковую структуру наборов данных, поэтому нужно будет добавить столбец в тестовый ДС  и 

# добавить признак датасета в оба набора

mf['Main'] = True

kf['Main'] = False

mf['Name'] = 'Dummy' # В этой задаче название для ресторана нам не важно

mf.info()
kf.info()
#Дальнейшие действия уже будем проводить с объединенным набором данных.

df = pd.concat([mf, kf], sort=False)

df.info()
#у нас есть большое количество незаполненных полей ценового диапазона, попробуем их заполнить. Предпологаем, что 

# есть зависимость между страной, городом и ценовым диапазоном.

#но спера нам надо будет разобратьсяс принадлежностью к стране, плюс мы должны будем определить является ли город

# столицей, что тоже может быть влияет на оценку

europe_capitals = pd.read_csv('eu_capitals.csv')

europe_capitals.head()
df  = df.merge(europe_capitals, on = 'City', how = 'left')

df.head()
#  проверяем, что с данными все ок

df.info()







#Теперь займемся признаком столицы - где поле Country isNan - это не столица 

mask = pd.isna(df['Country'])

df['isCap'] = df[~mask]['Country'].apply(lambda x: True)

df['isCap'] = df['isCap'].fillna(False)

df.head()
df.info()
#Теперь заполним недостающие данные по странам

#есть приготовленный датасет

w_c = pd.read_csv('europe_cities.csv', sep=';')

w_c.head()
#ОБъединим два набора данных с последующей проверкой

df = df.drop(['Country'], axis=1)

df  = df.merge(w_c, on = 'City', how = 'left')

df.info()

df.head()
res_count = {

    'Paris': 17593,

    'Stockholm': 3131,

    'London': 22366,

    'Berlin': 8110, 

    'Munich': 3367,

    'Oporto': 2060, 

    'Milan': 7940,

    'Bratislava': 1331,

    'Vienna': 4387, 

    'Rome': 12086,

    'Barcelona': 10086,

    'Madrid': 11562,

    'Dublin': 2706,

    'Brussels': 3703,

    'Zurich': 1901,

    'Warsaw': 3210,

    'Budapest': 3445, 

    'Copenhagen': 2637,

    'Amsterdam': 4189,

    'Lyon': 2833,

    'Hamburg': 3501, 

    'Lisbon': 4985,

    'Prague': 5850,

    'Oslo': 1441, 

    'Helsinki': 1661,

    'Edinburgh': 2248,

    'Geneva': 1753,

    'Ljubljana': 647,

    'Athens': 2814,

    'Luxembourg': 759,

    'Krakow': 1832       

}
df['Restaurants Count'] = df['City'].map(res_count)
df.info()
df.head()
df['Relative Ranking'] = df['Ranking'] / df['Restaurants Count']

df.info()
#Теперь цены, проверим доступные значения

df['Price Range'].unique()
#для определения  распределения цены проведем небольшую трансформацию данных

price_range = {'$' : 1,

                '$$ - $$$' : 2,

                '$$$$' : 3

              }

df['Price'] = df['Price Range'].map(price_range)

df.head()
#опять же, спасибо коллегам, добавим признак того, что у ресторана отсутствовало значение ценового диапазона

df['NoPrice'] =  pd.isna(df['Price Range'])

df.head()
df.info()
#теперь посмотрим медиану и среднее

df[['Price', 'isCap', 'Country']].groupby([ 'Country','isCap']).median()
df[['Price', 'isCap', 'Country']].groupby([ 'Country','isCap']).mean().sort_values(by = 'Price')
#у нас есть два варианта - проставить везде средний диапазон, исходя из данных по медиане, или же попробовать 

# поиграть со средними значениями. Например, для части ресторанов можно проставить минимальный диапазон 

# для тех, у которых среднее менее 1.7-1.8

# пока остановимся на медиане





df['Price Range'].fillna('$$ - $$$', inplace=True)

df['Price'].fillna(2, inplace=True)

df.head()
df.info()
#Далее начинаем работать с количеством кухонь

def cStyle_clean(s):

    new_st = str(s)[1:-1]

    new_st = new_st.replace('\'', '')

    new_st = new_st.replace(', ',',')

    return new_st



df['Cuisine Style'] = df[df['Cuisine Style'].notna()]['Cuisine Style'].apply(cStyle_clean)

df['Cuisine Count'] = df[df['Cuisine Style'].notna()]['Cuisine Style'].apply(lambda x:len(x.split(',')))

df.head()
df.info()
#проверим количество кухонь по городам с учетом столиц

cc_mean_cap = df[(df['Cuisine Style'].notna()) & (df.isCap == True)]['Cuisine Count'].mean()

print('isCap', cc_mean_cap)

cc_mean_ncap = df[(df['Cuisine Style'].notna()) & (df.isCap == False)]['Cuisine Count'].mean()

print('is no Cap', cc_mean_ncap)

#df['Cuisine Count'] = df['Cuisine Count'].fillna(cc_mean)

#df

#с учетом этих значений, заполним 

df['Cuisine Count'] = df['Cuisine Count'].fillna(3)

df.info()
#Заполним пропущенные отзывы

df['Reviews'] = df['Reviews'].fillna('[[], []]')

df.info()
#получим доступный список кухонь

cs_list = []

def adder(row):

    l = row.split(',')

    for s in l:

        cs_list.append(s)



#запускаем обработку     

df[df['Cuisine Style'].notna()]['Cuisine Style'].apply(adder)

cs_list  = pd.Series(cs_list).value_counts().index

cs_list
#теперь функция которая будет обрабатывать комментарии

def make_CStyle(row):

    cStyle = ''

    if pd.isnull(row['Cuisine Style']):

        

        for word in cs_list:

            #print(word)

            if word.upper() in row['Reviews'].upper():

                cStyle +=','

                cStyle +=word

        if cStyle == '':

            return None

        return cStyle[1:]

    return row['Cuisine Style']





df['Cuisine Style'] = df.apply(make_CStyle, axis = 1)

df.info()

#для остальных заполним просто по умолчанию

df['Cuisine Style'].fillna('NotDef', inplace=True)

df.info()
# Теперь заполним более корректно данные по ревью

capital_review = df[df['isCap']]['Number of Reviews'].median()

not_capital_review = df[~df['isCap']]['Number of Reviews'].median()

print(capital_review, not_capital_review )



#функция заполнения количества отзывов

def fill_num_reviews(row):

    if pd.isnull(row['Number of Reviews']):

        if row['isCap'] == 1:

#             return capital_review

           return 0

        if row['isCap'] ==0:

#             return not_capital_review

            return 0

    return row['Number of Reviews']

df['Number of Reviews'] = df.apply(fill_num_reviews, axis=1)
df.info()
#Теперь займемся временем

import re



from datetime import datetime, timedelta



pattern = '\d\d/\d\d/\d\d\d\d'

def getMaxReviewDate(string):

    # находим список значений дат 

    date_list = re.findall(pattern, string)

   # print(date_list)

    #осталось нормальное количество данных - теперь роазберемся с ними

    if len(date_list) == 0:

        return 0

#         return int(datetime.strptime('01/01/2014', '%m/%d/%Y').strftime("%s"))

        #return None

    if len(date_list) == 1:

        return int(datetime.strptime(date_list[0], '%m/%d/%Y').strftime("%s"))

        #return pd.to_datetime(date_list[0])

    dt = [] 

    dt.append(int(datetime.strptime(date_list[0], '%m/%d/%Y').strftime("%s")))

    #dt.append(pd.to_datetime(date_list[0]))

    dt.append(int(datetime.strptime(date_list[1], '%m/%d/%Y').strftime("%s")))

    #dt.append(pd.to_datetime(date_list[1]))

    dt = sorted(dt)

    return dt[0]

df['max_review_date'] = df['Reviews'].apply(getMaxReviewDate)

df.head()

def getInterval(string):

        # находим список значений дат 

    date_list = re.findall(pattern, string)

   # print(date_list)

    #осталось нормальное количество данных - теперь роазберемся с ними

    if len(date_list) == 0:

        return 0

    if len(date_list) == 1:

        return 0

    dt = [] 

    dt.append(int(datetime.strptime(date_list[0], '%m/%d/%Y').strftime("%s")))

   # dt.append(pd.to_datetime(date_list[0]))

    dt.append(int(datetime.strptime(date_list[1], '%m/%d/%Y').strftime("%s")))

    #dt.append(pd.to_datetime(date_list[1]))

    dt = sorted(dt)

    return (pd.to_datetime(dt[1]) - pd.to_datetime(dt[0])).days

df['interval'] = df['Reviews'].apply(getInterval)
mean_MRD = df[df['max_review_date'] != 0]['max_review_date'].median()



df['max_review_date'] = df['max_review_date'].fillna(mean_MRD)
#df['max_review_date'] = pd.to_datetime(df['max_review_date'])

#df['interval'] = pd.to_datetime(df['interval'])
df.head()
df.info()
#теперь будем создавать дамми-колонки -  по цене

df = pd.concat([df, pd.get_dummies(df['Price Range'])], axis=1)
#дамми по городам

df = pd.concat([df, pd.get_dummies(df['City'])], axis=1)
#дамми по странам

df = pd.concat([df, pd.get_dummies(df['Country'])], axis=1)
# а теперь самое интересно - дамми по кухням

#df = pd.concat([df, pd.get_dummies(df['Cuisine Style'])], axis=1)

#dummies = df['Cuisine Style'].str.get_dummies(sep=',')

df = pd.concat([df, df['Cuisine Style'].str.get_dummies(sep=',')], axis=1)

#df = pd.concat([df, pd.get_dummies(df['Cuisine Style'])], axis=1)
# Предварительная очистка данных

temp = df.drop(['Cuisine Style','Name', 'City', 'Price Range', 'Reviews', 'URL_TA', 'ID_TA', 'Country'], axis = 1)

temp.head(1)
temp = temp.fillna(1 )

temp.head(1)
# Х - данные с информацией о ресторанах, у - целевая переменная (рейтинги ресторанов)

main = temp[temp['Main']]

main.info()
kaggle = temp[~temp['Main']]

kaggle = kaggle.drop(['Restaurant_id', 'Rating'], axis = 1)

kaggle.info()
X = main

y = main['Rating']

X = X.drop(['Restaurant_id', 'Rating'], axis = 1)

y.head()
X.head()
X.info()
# Загружаем специальный инструмент для разбивки:

from sklearn.model_selection import train_test_split
# Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.

# Для тестирования мы будем использовать 25% от исходного датасета.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 56)
# Импортируем необходимые библиотеки:

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели
# Создаём модель

regr = RandomForestRegressor(n_estimators=630,  random_state=56, n_jobs = -1)



# Обучаем модель на тестовом наборе данных

#regr.fit(X_train, y_train)

regr.fit(X, y)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

#y_pred = regr.predict(X_test)



y_pred = regr.predict(kaggle)

import numpy as np

y_pred = np.round(y_pred*2)/2
# формируем финальный результат

kaggle = temp[~temp['Main']]

kaggle = kaggle[['Restaurant_id']]

kaggle['Rating'] = y_pred

kaggle.to_csv('solution.csv', index = False) 
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

#print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
#с добавленным relative ranking  - MAE: 0.172625 n_estimators=500,  random_state=56

#MAE: MAE: 0.175375 - без учета кухонь

# с учетом кухонь MAE: 0.172875 n_estimators=500,  random_state=56, n_jobs = -1
plt.figure(figsize=(20,20))

feat_importances = pd.Series(regr.feature_importances_, index=X.columns)

feat_importances.nlargest(40).plot(kind='barh')



