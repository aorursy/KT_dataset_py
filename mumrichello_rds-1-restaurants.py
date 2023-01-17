import pandas as pd

import numpy as np



import ast



from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer



from datetime import datetime

import time



from bs4 import BeautifulSoup    

import requests



def get_list(list_string):

    result_list = ast.literal_eval(list_string)

    return result_list



#Использовалась для добычи кол-ва отзывов для каждого ресторана с TripAdvisor

# def totalRestNum(adress):

#     response=requests.get(adress)

#     page=BeautifulSoup(response.text, 'html.parser')

#     total=page.find('span',class_="header_popularity popIndexValidation")

#     total=str(total)

#     b=(total.find('</b> of'))

#     c=(total.find('<a href='))

#     if (b==-1 or c==-1): return(-1)

#     else:

#         zzz=int(total[b+8:c].replace(',',''))

#         return(zzz)



# url='https://www.tripadvisor.com'



# allCityRests={}





# for index, row in df.iterrows():

#     if row['City'] not in allCityRests.keys():

#         pp=totalRestNum(url+row['URL_TA'])

#         if pp!=-1:

#             allCityRests[row['City']]=pp

            

# print(allCityRests)





CURRENT_DATE = pd.to_datetime('14/03/2020')



RANDOM_SEED=42



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')



# df_train = pd.read_csv('main_task.csv')

# df_test=pd.read_csv('kaggle_task.csv')



# sample_submission = pd.read_csv('sample_submission.csv')



df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



df = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем



# display(df)

df['Number of Reviews Was NAN'] = df['Number of Reviews'].isna()

df['Number of Reviews'].fillna(1,inplace=True)



start_time = datetime.now()



df['Reviews'].fillna('[[], []]',inplace=True)



y = df['Rating']



# Ранжирование по цене



def priceRange(s):

    if s=='$': return(1)

    elif s=='hz': return(10)

    elif s=='$$ - $$$': return(100)

    else: return(1000)

    

df['Price Range'].fillna('hz',inplace=True)

df['Price_Int']=df['Price Range'].apply(priceRange)



# Ранжирование по столицам



def cityVal(s):

    if s in ['London','Paris','Madrid','Barcelona','Berlin','Rome','Prague','Lisbon','Vienna','Amsterdam','Brussels',

             'Stockholm','Budapest','Warsaw','Dublin','Copenhagen','Athens','Edinburgh','Oslo','Helsinki','Bratislava',

             'Luxembourg','Ljubljana']: return(1)

    else: return(0)



df['isCapital']=df.City.apply(cityVal)



# Ранжирование по кол-ву кухонь



df['Cuisine Style'].fillna('Something',inplace=True)

aa={}

df['Cuisine Style']=df['Cuisine Style'].str[2:-2].str.split("', '")

for i in df['Cuisine Style']:

    for j in range(len(i)):

        if i[j] not in aa.keys(): 

            aa[i[j]]=1

        else:

            aa[i[j]]+=1



def cusNum(a):

    return(len(a))

    

df['Cuisine Count']=df['Cuisine Style'].apply(cusNum)

 

# Создание всех кухонь как отдельный признак



aa={k: v for k, v in sorted(aa.items(), key=lambda item: item[1],reverse = True)}



for i in aa.keys():

    df[i]=0



for index, row in df.iterrows():

    for j in row['Cuisine Style']:

        if j in df:

            row[j]=1

            df.at[index,j] = 1



# Присваивание городам номера

cities_le = LabelEncoder()

cities_le.fit(df['City'])

df['City Code'] = cities_le.transform(df['City'])



df['cheapCapital']=df['isCapital']*(df['Price_Int'])





# Попытка ранжирования по положительности/отрицательности отзывов

def goodRev(s):

    if s=='[[], []]': return(0)

    else:

        b=(s.split("'], ['"))[0][3:].split("', '")

        retkol=0

        for a in b:

            for c in a.split():

                if c.lower() in goodWords:

                    retkol+=1

        return(retkol)



def badRev(s):

    if s=='[[], []]': return(0)

    else:

        b=(s.split("'], ['"))[0][3:].split("', '")

        retkol=0

        for a in b:

            for c in a.split():

                if c.lower() in badWords:

                    retkol+=1

        return(retkol)



goodWords=['good','unique','nice','delicious','friend','firends','best','expcerience','wonderful','atmosphere','old'

           ,'jewel','hidden','lovely','excellent','great','relaxed']

badWords = ['pricey','price','bad','worst','but','awfull','average','never','?','seriously','wasting','waste','trash']



df['revGood']=df['Reviews'].apply(goodRev)

df['revBad']=df['Reviews'].apply(badRev)



#Создание признаков по кол-ву ресторанов, населению и странам ресторанов



allCityRests={'Paris': 16707, 'Helsinki': 1482, 'Edinburgh': 1857, 'London': 19417, 'Bratislava': 1202, 

              'Lisbon': 4695, 'Budapest': 2917, 'Stockholm': 2893, 'Rome': 10567, 'Milan': 7018, 'Munich': 3022,

              'Hamburg': 3149, 'Prague': 5236, 'Vienna': 3954, 'Dublin': 2310, 'Barcelona': 9325, 'Brussels': 59,

              'Oslo': 1306, 'Madrid': 10921, 'Amsterdam': 3857, 'Berlin': 6921, 'Lyon': 2704, 'Athens': 2449, 

              'Warsaw': 3049, 'Oporto': 1904, 'Krakow': 1620, 'Copenhagen': 2335, 'Zurich': 1805, 'Geneva': 1678,

              'Luxembourg': 718, 'Ljubljana': 584}



city_population = {'London': 8173900,'Paris': 2240621,'Madrid': 3155360,'Barcelona': 1593075,'Berlin': 3326002,'Milan': 1331586,

    'Rome': 2870493,'Prague': 1272690,'Lisbon': 547733,'Vienna': 1765649,'Amsterdam': 825080,'Brussels': 144784,'Hamburg': 1718187,

    'Munich': 1364920,'Lyon': 496343,'Stockholm': 1981263,'Budapest': 1744665,'Warsaw': 1720398,'Dublin': 506211 ,'Copenhagen': 1246611,

    'Athens': 3168846,'Edinburgh': 476100,'Zurich': 402275,'Oporto': 221800,'Geneva': 196150,'Krakow': 756183,'Oslo': 673469,

    'Helsinki': 574579,'Bratislava': 413192,'Luxembourg': 576249,'Ljubljana': 277554}



city_country = {'London': 'UK','Paris': 'France','Madrid': 'Spain','Barcelona': 'Spain','Berlin': 'Germany','Milan': 'Italy',

    'Rome': 'Italy','Prague': 'Czech','Lisbon': 'Portugalia','Vienna': 'Austria','Amsterdam': 'Nederlands','Brussels': '144784 ',

    'Hamburg': 'Germany','Munich': 'Germany','Lyon': 'France','Stockholm': 'Sweden','Budapest': 'Hungary','Warsaw': 'Poland',

    'Dublin': 'Ireland' ,'Copenhagen': 'Denmark','Athens': 'Greece','Edinburgh': 'Schotland','Zurich': 'Switzerland','Oporto': 'Portugalia',

    'Geneva': 'Switzerland','Krakow': 'Poland','Oslo': 'Norway','Helsinki': 'Finland','Bratislava': 'Slovakia','Luxembourg': 'Luxembourg',

    'Ljubljana': 'Slovenija'}



df['Population'] = df['City'].map(city_population)

df['Country'] = df['City'].map(city_country)



countries_le = LabelEncoder()

countries_le.fit(df['Country'])

df['Country Code'] = countries_le.transform(df['Country'])



df['Restaurants Count'] = df['City'].map(allCityRests)

df['Relative Ranking']=df['Ranking']/df['Restaurants Count']



df['People Per Restaurant'] = df['Population'] / df['Restaurants Count']



df['ID_TA Numeric'] = df['ID_TA'].apply(lambda id_ta: int(id_ta[1:]))

df['URL_TA Numeric'] = df['URL_TA'].apply(lambda x: float(x[20:26]))



# Сначала заменим nan в строковых представлениях списков отзывов и их дат на строку с датой по-умолчанию

# а также заменим пустой список вида [] на список с незаполненными элементами вида ['01/01/2000', '01/01/2000']

def nan_to_default_date(list_string):

    try:

        list_string = list_string.replace('[nan', "['01/01/2000'")

        list_string = list_string.replace('nan]', "'01/01/2000']")

        list_string = list_string.replace('[]', "['01/01/2000', '01/01/2000']")

    except:

        list_string = "[['None Review', 'None Review'], ['01/01/2000', '01/01/2000']]"

    return list_string



df['Reviews'] = df['Reviews'].apply(nan_to_default_date)



# Преобразуем данные в столбце 'Reviews' к типу list

last_review = []

last_review_date = []

prelast_review = []

prelast_review_date = []

for reviews in df['Reviews']:

    reviews_list = get_list(reviews)

    if len(reviews_list) == 2:

        if (len(reviews_list[0]) == 2 and len(reviews_list[1]) == 2):

            last_review.append(reviews_list[0][0])

            last_review_date.append(reviews_list[1][0])

            prelast_review.append(reviews_list[0][1])

            prelast_review_date.append(reviews_list[1][1])

        elif (len(reviews_list[0]) == 1 and len(reviews_list[1]) == 1):

            last_review.append(reviews_list[0][0])

            last_review_date.append(reviews_list[1][0])

            prelast_review.append('None review')

            prelast_review_date.append('01/01/2000')

        else:

            print(reviews_list)

            

            

# Добавим новые признаки:

# - последний отзыв

df['Last Review'] = last_review

# - дата последнего отзыва

df['Last Review Date'] = last_review_date

# - предпоследний отзыв

df['Prelast Review'] = prelast_review

# - дата предпоследнего отзыва

df['Prelast Review Date'] = prelast_review_date



# Переведем даты в формат datetime

df['Last Review Date'] = pd.to_datetime(df['Last Review Date'])

df['Prelast Review Date'] = pd.to_datetime(df['Prelast Review Date'])



df['Days Between Reviews'] = (df['Last Review Date'] - df['Prelast Review Date'])

def get_days(timedelta):

    return timedelta.days

df['Days Between Reviews'] = df['Days Between Reviews'].apply(get_days)





df['Days Since Last Review'] = df['Last Review Date'].apply(lambda date: CURRENT_DATE - date)

df['Days Since Last Review'] = df['Days Since Last Review'].apply(get_days)



df=df.drop(['Restaurant_id','City','Cuisine Style','Price Range','Reviews','URL_TA','ID_TA','Country',

            'Last Review','Prelast Review','Last Review Date','Prelast Review Date'],axis=1)



# Загружаем специальный инструмент для разбивки:

from sklearn.model_selection import train_test_split



# Теперь выделим тестовую часть

train_data = df.query('sample == 1').drop(['sample'], axis=1)

test_data = df.query('sample == 0').drop(['sample'], axis=1)



y = train_data.Rating.values            # наш таргет

X = train_data.drop(['Rating'], axis=1)



# Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.

# Для тестирования мы будем использовать 25% от исходного датасета.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_SEED)



print("calc time is ", datetime.now() - start_time)

# display(X_train)

# Импортируем необходимые библиотеки:

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели

start_time = datetime.now()

# Создаём модель

# regr = RandomForestRegressor(n_estimators=100)

regr = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)

# Обучаем модель на тестовом наборе данных

regr.fit(X_train, y_train)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

y_pred = regr.predict(X_test)



#Функция округления.. вторая показала себя чуть лучше :-)

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

    

def round_d2(rec):

    if rec <0.25:

        return 0

    elif 0.25<rec<=0.75:

        return 0.5

    elif 0.75<rec<=1.25:

        return 1

    elif 1.25<rec<=1.75:

        return 1.5

    elif 1.75<rec<=2.25:

        return 2

    elif 2.25<rec<=2.75:

        return 2.5

    elif 2.75<rec<=3.25:

        return 3

    elif 3.25<rec<=3.75:

        return 3.5

    elif 3.75<rec<=4.25:

        return 4

    elif 4.25<rec<=4.75:

        return 4.5

    else:

        return 5

    

for i in range(y_pred.size):

    y_pred[i]=round_d2(y_pred[i])



# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))



print("calc time is ", datetime.now() - start_time)



test_data = test_data.drop(['Rating'], axis=1)



predict_submission = regr.predict(test_data)



sample_submission['Rating'] = predict_submission

sample_submission['Rating'] = sample_submission['Rating'].apply(round_d2)

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)
