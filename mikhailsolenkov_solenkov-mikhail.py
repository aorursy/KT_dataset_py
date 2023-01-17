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
import re

import datetime



#первая базовая табличка

df_1 = pd.read_csv('../input/sf-dst-restaurant-rating/main_task.csv')



#вторая табличка из Kaggle без столбца рейтинга

df_2 = pd.read_csv('../input/sf-dst-restaurant-rating/kaggle_task.csv')



# Добавим признак 'Main' для отделения основной выборки от валидационной

df_1['Main'] = True

df_2['Main'] = False



# объединим в одну табличку

df = pd.concat([df_1, df_2])



# Очистим столбец Restaurant_id (сделаем только из цифр)

df.Restaurant_id = df.Restaurant_id.apply(lambda x : float(x.split('_')[1])) 



# Очистим ID_TA

df.ID_TA = df.ID_TA.apply(lambda x : float(x[1:]))





# Заполним отсутствие отзывов "0"

df['Number of Reviews'] = df['Number of Reviews'].fillna(0)



# Сделаем рейтинг по затрачиваемым бабкам ['cash_range']

def cashtonumber(item):

    if item == '$':

        return 10

    elif item == '$$ - $$$':

        return 100

    elif item == '$$$$':

        return 1000

    else:

        return 10  #игрался с параметром - на точность не повлиял, каким его не ставь (0, 10, 100, 1000)

df['cash_range'] = df['Price Range'].apply(cashtonumber)



# добавим столбец - относится город к столицам или нет ['capitals']

capitals = ['Paris', 'Stockholm', 'London', 'Berlin', 'Bratislava', 'Vienna', 'Rome', 'Madrid', 'Dublin', 'Brussels',

            'Warsaw', 'Budapest', 'Copenhagen', 'Amsterdam', 'Lisbon', 'Prague', 'Oslo','Helsinki', 'Edinburgh', 

            'Ljubljana', 'Athens','Luxembourg']

df['is_capital'] = df['City'].apply(lambda x : 1 if x in capitals else 0)



# добавим кол-во кухонь предлагаемых в ресторане

# заполним пропуски

df['Cuisine Style'] = df['Cuisine Style'].fillna("['Usual']")



df['kol-vo kuchon'] = df['Cuisine Style'].str[2:-2].str.split("', '").str.len().fillna(1)



#нужно добавить думисы по типу кухонь

df['kuchni'] = df['Cuisine Style'].str[2:-2].str.split("', '")



from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

mlb = MultiLabelBinarizer()



# взято отсюда https://ru.stackoverflow.com/questions/928443/%D0%9A%D0%B0%D0%BA-%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0%D0%B5%D1%82-sklearn-preprocessing-multilabelbinarizer

#df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('Col2')), index=df.index, columns=mlb.classes_))



# делаем табличку думисов - по типам кухонь

new_df_kuchni = pd.DataFrame(mlb.fit_transform(df.pop('kuchni')), index=df.index, columns=mlb.classes_)



# делаем табличку думисов - по городам

# забиваем гвозди микроскопом. Делаем из городов списки городов, например из 'Paris' тип str получаем ['Paris'] тип list

# потому что MultiLabelBinarizer() работает со списками (хз чем работать со строками)

city_list = []

for i in df['City']:

    city_list.append([i])

    

df['city_list'] = city_list  

new_df_cities = pd.DataFrame(mlb.fit_transform(df['city_list']), index=df.index, columns=mlb.classes_)



# добавим время отзывов и разницу между ними

    #сначала новые столбцы с датами отзывов

pattern = re.compile('\d{2}/\d{2}/\d{4}')

reviews=[]

for i in df['Reviews']:

    try:

        reviews.append(re.findall(pattern, i))

    except:

        reviews.append(['01/01/1970', '01/01/1970'])



date=pd.DataFrame(reviews)  #отдельный датафрейм для удобства

date = date.fillna(0)  # заполним пропуски нулями

date.columns=['date1', 'date2']

date['date1'] = pd.to_datetime(date['date1'])

date['date2'] = pd.to_datetime(date['date2']) 



df['time_between'] = (date['date1'] - date['date2']).apply(lambda x : float(x.days))  

# время от последнего отзыва



last_time_review = []

for i in date['date1']:

    last_time_review.append(float((datetime.datetime.today() - i).days))

    

df['last_time_review'] = last_time_review

# Добавим население из Google (честно заимствовано у коллег)

population = {

    'London': 8908081,

    'Paris': 2240621,

    'Madrid': 3155360,

    'Barcelona': 1593075,

    'Berlin': 3326002,

    'Milan': 1331586,

    'Rome': 2870493,

    'Prague': 1272690,

    'Lisbon': 547733,

    'Vienna': 1765649,

    'Amsterdam': 825080,

    'Brussels': 144784,

    'Hamburg': 1718187,

    'Munich': 1364920,

    'Lyon': 496343,

    'Stockholm': 1981263,

    'Budapest': 1744665,

    'Warsaw': 1720398,

    'Dublin': 506211 ,

    'Copenhagen': 1246611,

    'Athens': 3168846,

    'Edinburgh': 476100,

    'Zurich': 402275,

    'Oporto': 221800,

    'Geneva': 196150,

    'Krakow': 756183,

    'Oslo': 673469,

    'Helsinki': 574579,

    'Bratislava': 413192,

    'Luxembourg': 576249,

    'Ljubljana': 277554

}

df['Population'] = df['City'].map(population)
#кол-во ресторанов в выборке

count = df['City'].value_counts()

df['restaurant_in_city'] = df['City'].apply(lambda x : count[x])
# добавим ранг между данными ресторанами

df['range_in_count'] = df['Ranking'] / df['restaurant_in_city']
# добавим ранг за деньги

df['range_for_money'] = df['range_in_count']/df['cash_range']

df['rank_for_money'] = df['Ranking']/df['cash_range']
# добавим кол-во отзывов на ранг УХУДШИЛО

df['reviews_in_range'] = df['Number of Reviews']/df['range_in_count']

df['reviews_in_rank'] = df['Number of Reviews']/df['Ranking']
# плотность населения на один ресторан

df['person_per_restaurant'] = df['Population'] / df['restaurant_in_city']
# Добавим координаты городов. Используем стороннюю библиотеку geopy

#!pip install geopy

#geolocator = Nominatim()



# ниже образец кода для одного города

#location = geolocator.geocode("Paris")

#print((location.latitude, location.longitude))



cities = ['Paris', 'Stockholm', 'London', 'Berlin', 'Munich', 'Oporto', 'Milan', 'Bratislava', 'Vienna', 'Rome',

          'Barcelona', 'Madrid','Dublin', 'Brussels', 'Zurich', 'Warsaw', 'Budapest', 'Copenhagen', 'Amsterdam', 

          'Lyon', 'Hamburg', 'Lisbon', 'Prague', 'Oslo', 'Helsinki', 'Edinburgh', 'Geneva', 'Ljubljana', 'Athens',

          'Luxembourg', 'Krakow']

def city_location(city):

    location = geolocator.geocode(city)

    return (location.latitude, location.longitude)



#for i in cities:

#    print("'",i,"'",":",city_location(i),",", sep = "")

#для создания словаря пришлось просто построчно печатать данные, походу ресурсоемкая библиотека 

# хотел столбец через apply заполнить ))

locations = {'Paris':[48.8566969, 2.3514616],

'Stockholm':[59.3251172, 18.0710935],

'London':[51.5073219, -0.1276474],

'Berlin':[52.5170365, 13.3888599],

'Munich':[48.1371079, 11.5753822],

'Oporto':[41.1494512, -8.6107884],

'Milan':[45.4668, 9.1905],

'Bratislava':[48.1516988, 17.1093063],

'Vienna':[48.2083537, 16.3725042],

'Rome':[41.8933203, 12.4829321],

'Barcelona':[41.3828939, 2.1774322],

'Madrid':[40.4167047, -3.7035825],

'Dublin':[53.3497645, -6.2602732],

'Brussels':[50.8436709, 4.3674366933879565],

'Zurich':[47.3723941, 8.5423328],

'Warsaw':[52.2337172, 21.07141112883227],

'Budapest':[47.48138955, 19.14607278448202],

'Copenhagen':[55.6867243, 12.5700724],

'Amsterdam':[52.3727598, 4.8936041],

'Lyon':[45.7578137, 4.8320114],

'Hamburg':[53.5437641, 10.0099133],

'Lisbon':[38.7077507, -9.1365919],

'Prague':[50.0874654, 14.4212535],

'Oslo':[59.9133301, 10.7389701],

'Helsinki':[60.1674098, 24.9425769],

'Edinburgh':[55.9533456, -3.1883749],

'Geneva':[46.2017559, 6.1466014],

'Ljubljana':[46.0499803, 14.5068602],

'Athens':[37.9839412, 23.7283052],

'Luxembourg':[49.8158683, 6.1296751],

'Krakow':[50.0469432, 19.997153435836697] 

}



df['city_latitude'] = df['City'].apply(lambda x : locations[x][0])

df['city_longitude'] = df['City'].apply(lambda x : locations[x][1])
# площадь города, квадратных километров (ручками из Гугл) парсить пока не умею, а библиотеки не нашел

square = {

    'London': 1572,

    'Paris': 105.4,

    'Madrid': 607,

    'Barcelona': 100.4,

    'Berlin': 891.68,

    'Milan': 181.67,

    'Rome': 1287.36,

    'Prague': 500,

    'Lisbon': 100.05,

    'Vienna': 414.75,

    'Amsterdam': 219.4,

    'Brussels': 32.61,

    'Hamburg': 755.09,

    'Munich': 310.71,

    'Lyon': 47.87,

    'Stockholm': 188,

    'Budapest': 525.14,

    'Warsaw': 517,

    'Dublin':  318,

    'Copenhagen': 86.4,

    'Athens': 412,

    'Edinburgh': 118,

    'Zurich': 91.88,

    'Oporto': 41.66,

    'Geneva': 15.93,

    'Krakow': 327,

    'Oslo': 454,

    'Helsinki': 715.48,

    'Bratislava': 368,

    'Luxembourg': 2586.4,

    'Ljubljana': 163.8

}

df['city_square'] = df['City'].map(square)
#добавим относительные показатели - плотность населения и кол-во ресторанов на 1 кв. км

df['people_na_km'] = df['Population']/df['city_square']

df['restaurants_na_km'] = df['restaurant_in_city']/df['city_square']
# заполним пустоту отзывов

df['Reviews'] = df['Reviews'].fillna('empty')
#сделаем по review списки слов



review_list = []

symbols = ['[',']',",","'",'"','!','/', '0','1','2','3','4','5','6','7','8','9' ]



for i in df['Reviews']:

    for symbol in symbols:

        i = i.replace(symbol, '')

        i = i.lower()

    review_list.append(i.split(' '))

    

df['review_list'] = review_list



# предположим, что если в отзыве есть хорошие слова, то +1 бал, если плохие, то -1 бал

# пока считал употребляемость слов - комп подзавис капитально. Списки из тренировочного файла

good_words = ['good', 'delicious', 'best', 'nice', 'amazing', 'excellent', 'great', 'lovely', 'fantastic',

              'tasty', 'perfect', 'wonderful', 'awesome', 'cosy', 'cozy', 'pleasant', 'cool', 'fabulous',

             'beautiful', 'super ', 'love', 'cute', 'favourite', 'fine', 'pretty']

bad_words = ['bad', 'but', 'average', 'expensive', 'disappointing', 'terrible', 'worst', 'nothing', 'horrible',

            'rude', 'awful']



def to_word_rating (word_list): # передаем список слов

    if type(word_list) == list:

        count = 0

        for i in word_list:

            if i in good_words:

                count += 1

            elif i in bad_words:

                count -= 1

        return count

    else:

        return 0

        

df['word_rating'] = df['review_list'].apply(to_word_rating)

df = pd.concat([df, new_df_kuchni], axis=1) # убрать "закомментированность" перед запуском

df = pd.concat([df, new_df_cities], axis=1) # убрать "закомментированность" перед запуском
# оставляем в табличке для теста только рабочие столбцы 

df_for_test = df.drop(['City', 'Cuisine Style', 'Rating', 'Price Range', 'Reviews', 'URL_TA',

                      'city_list', 'review_list'], axis = 1)
# Готовим данные

X = df_for_test[df_for_test['Main']]

y = df[df['Main']]['Rating']



# Загружаем специальный инструмент для разбивки:

from sklearn.model_selection import train_test_split



# Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.

# Для тестирования мы будем использовать 20% от исходного датасета.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Импортируем необходимые библиотеки:

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели
# Создаём модель

regr = RandomForestRegressor(n_estimators=100)



# Обучаем модель на тестовом наборе данных

regr.fit(X_train, y_train)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

y_pred = regr.predict(X_test)
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# Округлим результаты работы модели:

def round_to_polovina(row):

    return (round(row*2.0)/2)



new_round = np.vectorize(round_to_polovina)

y_pred_round = new_round(regr.predict(X_test))

print('MAE:', metrics.mean_absolute_error(y_test, y_pred_round))
# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

kaggle_df = df_for_test[~df_for_test['Main']]

y_pred = regr.predict(kaggle_df)
submission = pd.DataFrame({

        "Restaurant_id": df_for_test[~df_for_test['Main']]['Restaurant_id'],

        "Rating": y_pred

    })

submission['Restaurant_id']=submission['Restaurant_id'].astype('str')



submission['Rating'] = submission['Rating'].apply(round_to_polovina)

submission.to_csv('submission.csv', index=False)

submission