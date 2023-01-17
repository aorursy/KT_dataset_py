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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

import matplotlib.pyplot as plt

import seaborn as sns 

import ast

%matplotlib inline

from bs4 import BeautifulSoup

import requests as req

import urllib

from multiprocessing.dummy import Pool as ThreadPool

import json



# Загружаем специальный удобный инструмент для разделения датасета:

from sklearn.model_selection import train_test_split

import os

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))




# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!

RANDOM_SEED = 42
# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

!pip freeze > requirements.txt
df_train = pd.read_csv('/kaggle/input/kaggle-sf-dst-through-1/main_task.csv/main_task.csv')

df_test = pd.read_csv('/kaggle/input/kaggle-sf-dst-through-1/kaggle_task.csv')

sample_submission = pd.read_csv('/kaggle/input/kaggle-sf-dst-through-1/sample_submission.csv/sample_submission.csv')




df_train.info()
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями

data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем

# удаляем Name есть только в тесте





#разгребает страницу трипэдвайзера



def TA_link_parser(rest_id, resp):

    soup = BeautifulSoup(resp.text, 'lxml')

    #тут поправить полезет ошибка сделать через ифы

    try:

        if len(soup.find_all("div", {"class": "restaurants-detail-overview-cards-RatingsOverviewCard__ranking--17CmN"})) > 1:

            spec_ranking = soup.find_all("div", {"class": "restaurants-detail-overview-cards-RatingsOverviewCard__ranking--17CmN"})[0].get_text()

            ranking = soup.find_all("div", {"class": "restaurants-detail-overview-cards-RatingsOverviewCard__ranking--17CmN"})[1].get_text()

        else :

            spec_ranking = None

            ranking = soup.find_all("div", {"class": "restaurants-detail-overview-cards-RatingsOverviewCard__ranking--17CmN"})[0].get_text()

    except:

        spec_ranking = None

        ranking = None

    try:

        num_rev = soup.find("a", {"class": "restaurants-detail-overview-cards-RatingsOverviewCard__ratingCount--DFxkG"}).get_text()

    except:

        num_rev = None

    try:

        header = soup.find("div", {"class": "header_links"}).get_text().split(",")

    except:

        pass

    try:

        price_range = header[0]

    except:

        price_range = None

    try:

        cousin_style = header[1:]

    except:

        cousin_style = None

    try:

        eting_time = soup.find_all("div", {"class": "restaurants-detail-overview-cards-DetailsSectionOverviewCard__tagText--1OH6h"})[2].get_text()

    except:

        eting_time = None

    category_rank = {}

    try:

        category_list = soup.find_all("span", {"class": "restaurants-detail-overview-cards-RatingsOverviewCard__ratingText--1P1Lq"})

        category_rank_list = soup.find_all("span", {"class": "restaurants-detail-overview-cards-RatingsOverviewCard__ratingBubbles--1kQYC"})

        for num, category in enumerate(category_list):

            category = category.get_text()

            rank = category_rank_list[num]

            rank = str(rank)[-18:-16]

            category_rank[category] = rank

    except:

        pass

    #считаю кто сколько поставил на английском языке (как вызвать все не понял)

    stars = {}

    try:

        stars_rev = soup.find_all("div", {"class": "ui_checkbox item"})



        for num, i in enumerate(stars_rev):

            i = i.get_text()

            #print(i)

            if num == 0:

                emoushen = "Excellent"

                em_count = i[len(emoushen):]

                stars[emoushen] = em_count

            elif num == 1:

                emoushen = "Very good"

                em_count = i[len(emoushen):]

                stars[emoushen] = em_count

            elif num == 2:

                emoushen = "Average"

                em_count = i[len(emoushen):]

                stars[emoushen] = em_count

            elif num == 3:

                emoushen = "Poor"

                em_count = i[len(emoushen):]

                stars[emoushen] = em_count

            elif num == 4:

                emoushen = "Terrible"

                em_count = i[len(emoushen):]

                stars[emoushen] = em_count

    except:

        pass

    

    return {rest_id : [spec_ranking, ranking, num_rev, price_range, cousin_style, eting_time, category_rank, stars]}







# сам парсер



def log_json(log_type, info):

    file_neme = log_type + "_log_parsera_TA_14_16.json"

    log = json.dumps(info)

    with open(file_neme, 'a') as file:

        file.write(log)

        



        

def parser_TA(rest_id):

    try:

        link = rest_dict[rest_id]

        #print(link)

        resp = req.get(link)

        #print(resp)

        response = TA_link_parser(rest_id, resp)

        log_json("good", response)

        #print(resp)

        #soup = BeautifulSoup(resp.text, 'lxml')

        #mydivs = soup.find("div", {"class": "restaurants-detail-overview-cards-RatingsOverviewCard__ranking--17CmN"})

        return response

    except:

        #bad_req.append({rest_id: link})

        response = {rest_id: link}

        log_json("error", response)

        return response

    

# запуск парсера !!! не запускать иначе очень болго ждать!!!

start = datetime.datetime.now()

pool = ThreadPool(15)

#results = pool.map(parser_TA, list(rest_dict.keys()))

print((datetime.datetime.now() - start).total_seconds())

#время обработки всего датасета в 15 потоков около 3 часов можно ускорить с помощью acinho но не разобрался
### этот код разгребает то что было наворочено в предидущем и приводит в порядок в годный для импорта в датафрейм словарь

result_dict = {"ID_TA": [], "special_city_rank": [], "city_rank": [], "TA_num_rew": [], "TA_price_range": [],

               "TA_cusine_style": [], "services": [], "category_marks": [], "en_rew_marks": []}

def result_parser(info):

    for key, val in info.items():

        result_dict["ID_TA"].append(key)

        result_dict["special_city_rank"].append(val[0])

        result_dict["city_rank"].append(val[1])

        result_dict["TA_num_rew"].append(val[2])

        result_dict["TA_price_range"].append(val[3])

        result_dict["TA_cusine_style"].append(val[4])

        result_dict["services"].append(val[5])

        result_dict["category_marks"].append(val[6])

        result_dict["en_rew_marks"].append(val[7])

        

#for info in result:

#    result_parser(info)

    

#data_ta = pd.DataFrame(result_dict)

#data_ta = data_ta.drop_duplicates(["ID_TA"])

#data_ta.to_csv("A_pars_15_12_v2.csv")
data_TA = pd.read_csv('/kaggle/input/ta-all-data/TA_all_data.csv', delimiter=',')
def special_city_rank(info):

    try:

        info = info.replace("#", "")

        info = info.replace(",", "")

        info = info.split(" ")

        return int(info[0])/int(info[2]), info[3]

    except:

        return None, None

    

data_TA["TA_special_rank"], data_TA["TA_special_rank_cusine"] = zip(*data_TA.special_city_rank.apply(special_city_rank))

# удаляем колонку по кторой отработали

data_TA = data_TA.drop(["special_city_rank"], axis = 1)
def city_rank(info):

    try:

        info = info.replace("#", "")

        info = info.replace(",", "")

        info = info.split(" ")

        return int(info[0])/int(info[2]), info[2]

    except:

        return None, None

    

data_TA["TA_rank"], data_TA["TA_num_rest_in_city"] = zip(*data_TA.city_rank.apply(city_rank))

# удаляем колонку по кторой отработали

data_TA = data_TA.drop(["city_rank"], axis = 1)

# подсчет сколько типов кухонь предлагает ресторан по данным трипэдвайзера

def cuisine_count(lst):

    try:

        lst =  lst[1:-1].split(",")

        return len(lst)

    except:

        return 1

    

data_TA['TA_cuisine_count'] = data_TA["TA_cusine_style"].apply(cuisine_count)

data_TA = data_TA.drop(["TA_cusine_style"], axis = 1)
#подсчет сервисов

data_TA['TA_services_count'] = data_TA["services"].apply(cuisine_count)

data_TA = data_TA.drop(["services"], axis = 1)
#исправление отзывов

def TA_num_rew(info):

    try:

        info = info.split(" ")[0]

    except:

        pass

    try:

        info = info.replace(",", "")

    except:

        pass

    try:

        info = int(info)

    except:

        pass

    return info



data_TA["TA_num_rew"] = data_TA.TA_num_rew.apply(TA_num_rew)
# чистим цену от косяков парсера

def TA_price_range(info):

    if info in ["$$ - $$$", "$", "$$$$"]:

        return info

    

data_TA["TA_price_range"] = data_TA.TA_price_range.apply(TA_price_range)
# парсим словарь категорий

def category_marks(info):

    info = ast.literal_eval(info)

    category_dict = {'Food': 0, 'Service': 0, 'Value': 0, 'Atmosphere': 0}

    for category in info:

        if category in category_dict:

            category_dict[category] = info[category]

    return tuple(category_dict.values())



data_TA["Food"], data_TA["Service"], data_TA["Value"], data_TA["Atmosphere"] = zip(*data_TA.category_marks.apply(category_marks))

data_TA = data_TA.drop(["category_marks"], axis = 1)
# паршу en_rew_marks

def en_rew_marks(info):

    info = ast.literal_eval(info)

    category_dict = {'Excellent': 0, 'Very good': 0, 'Average': 0, 'Poor': 0, 'Terrible': 0}

    for category in info:

        if category in category_dict:

            category_dict[category] = info[category]

    return tuple(category_dict.values())

data_TA["Excellent"], data_TA["Very_good"], data_TA["Average"], data_TA["Poor"], data_TA["Terrible"] = zip(*data_TA.en_rew_marks.apply(en_rew_marks))

data_TA = data_TA.drop(["en_rew_marks"], axis = 1)
data_TA.Food = data_TA.Food.astype('int64')

data_TA.Service = data_TA.Food.astype('int64')

data_TA.Value = data_TA.Food.astype('int64')

data_TA.Atmosphere = data_TA.Food.astype('int64')

data_TA.Excellent = data_TA.Food.astype('int64')

data_TA.Very_good = data_TA.Food.astype('int64')

data_TA.Average = data_TA.Food.astype('int64')

data_TA.Poor = data_TA.Food.astype('int64')

data_TA.Terrible = data_TA.Food.astype('int64')
#собираем весь ДФ воедино

data = data.merge(data_TA, how = "left", on = "ID_TA")
data.info()
data.fillna(value=pd.np.nan, inplace=True)

data.TA_num_rest_in_city = data.TA_num_rest_in_city.astype('float64')
# дельта между отзывами



def last_Reviews(Reviews):

    #print(Reviews)

    try:

        Reviews = Reviews.replace('nan', 'None')

        Reviews = ast.literal_eval(Reviews)

        Reviews = Reviews[1]

        rev_date=[]

    except:

        pass

        #print(Reviews)

    try:

        for i in Reviews:

            #print(i)

            i = datetime.datetime.strptime(i, '%m/%d/%Y')

            rev_date.append(i)

    except:

        pass

    try:

        date_last = max(rev_date)

        date_first = min(rev_date)

        return (date_last - date_first).days

    except:

        return None

    

data["delta_Reviews"] = data.Reviews.apply(last_Reviews)

data["delta_Reviews"].fillna(data["delta_Reviews"].max(), inplace = True)
# обработка 'Price Range'

data.loc[data["Price Range"] == "$","Price Range"] = 1

data.loc[data["Price Range"] == "$$ - $$$","Price Range"] = 2

data.loc[data["Price Range"] == "$$$$","Price Range"] = 3

data.loc[data["TA_price_range"] == "$","TA_price_range"] = 1

data.loc[data["TA_price_range"] == "$$ - $$$","TA_price_range"] = 2

data.loc[data["TA_price_range"] == "$$$$","TA_price_range"] = 3
#взаимная замена нанов в прайсрендж

data['Price Range'].fillna(data['TA_price_range'], inplace=True)

data['TA_price_range'].fillna(data['Price Range'], inplace=True)
# поскольку собиралось в разное время учтем это

(data['Number of Reviews']/data['TA_num_rew']).mean()



# все по аналогии с ценой

data['Number of Reviews'].fillna((data['TA_num_rew'])*((data['Number of Reviews']/data['TA_num_rew']).mean()), inplace=True)

data['TA_num_rew'].fillna((data['Number of Reviews'])/((data['Number of Reviews']/data['TA_num_rew']).mean()), inplace=True)

# добавляем количество туристов по городам

tourists_traf = {"Paris": 17560200, "Stockholm": 2604600, "London": 19233000, "Berlin": 5959400, "Munich": 4066600,

                "Oporto": 2341300, "Milan": 6481300, "Bratislava": 1500000, 'Vienna': 6410300, 'Rome': 10065400, 

                "Barcelona": 6714500, "Madrid": 5440100, "Dublin": 5213400, "Brussels": 3942000, "Zurich": 2240000,

                'Warsaw': 2850000, "Budapest": 3822800, "Copenhagen": 3069700, "Amsterdam": 8354200, "Lyon": 2963598, 

                "Hamburg": 1450000, "Lisbon": 3539400, "Prague": 8948600, "Oslo": 1263920, "Helsinki": 1268366, 

                 "Edinburgh": 1660000, "Geneva": 1150000, "Ljubljana": 520241, "Athens": 5728400, "Luxembourg": 690320,

                "Krakow": 1382864}



data["Vizitors"] = data.City.apply(lambda x: tourists_traf[x])

df_test["Vizitors"] = df_test.City.apply(lambda x: tourists_traf[x])

df_train["Vizitors"] = df_train.City.apply(lambda x: tourists_traf[x])



#Население

Cityzens = {'Amsterdam': 741636,

      'Athens': 664046,

      'Barcelona': 1621537,

      'Berlin': 3426354,

      'Bratislava': 423737,

      'Brussels': 1019022,

      'Budapest': 1741041,

      'Copenhagen': 1153615,

      'Dublin': 1024027,

      'Edinburgh': 464990,

      'Geneva': 183981,

      'Hamburg': 1739117,

      'Helsinki': 558457,

      'Krakow': 755050,

      'Lisbon': 517802,

      'Ljubljana': 272220,

      'London': 7556900,

      'Luxembourg': 119215,

      'Lyon': 472317,

      'Madrid': 3255944,

      'Milan': 1236837,

      'Munich': 1260391,

      'Oporto': 249633,

      'Oslo': 580000,

      'Paris': 2138551,

      'Prague': 1165581,

      'Rome': 2318895,

      'Stockholm': 1515017,

      'Vienna': 1691468,

      'Warsaw': 1702139,

      'Zurich': 341730}



data["Cityzens"] = data.City.apply(lambda x: Cityzens[x])

df_test["Cityzens"] = df_test.City.apply(lambda x: Cityzens[x])

df_train["Cityzens"] = df_train.City.apply(lambda x: Cityzens[x])
# считаем кол-во ресторанов в городе

data['Num_rest_in_City'] = np.NaN

# делаем так из за того что трипэдвайзер периодически отдавал кол-во ресторанов в локации

data['Num_rest_in_City'].fillna(data.groupby(['City'])['TA_num_rest_in_city'].transform('max'), inplace = True)

data = data.drop(["TA_num_rest_in_city"], axis = 1)
#рейтинг ресторана среди ресторанов в городе

data["Ranking_in_City"] = data.Ranking/data.Num_rest_in_City
# заменяем ТА_ранк где пусто на общий по городу

data['TA_rank'].fillna(data['Ranking_in_City'], inplace=True)
#  генерим признаки из оставшихся нанов

data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')

data['Cuisine_Style_isNAN'] = pd.isna(data['Cuisine Style']).astype('uint8')

data['Cuisine_Style_isNAN'] = pd.isna(data['Cuisine Style']).astype('uint8')

data['Price_Range_isNAN'] = pd.isna(data['Price Range']).astype('uint8')

data['Reviews_isNAN'] = pd.isna(data['Reviews']).astype('uint8')
# где все же остались нули меняем на 2 как самое частое значение

data['Price Range'].fillna(2, inplace=True)

data['TA_price_range'].fillna(2, inplace=True)
# остальное заменяем нулями благо там мало осталось

data['Number of Reviews'].fillna(0, inplace=True)

data['TA_num_rew'].fillna(0, inplace=True)
# заменяем где не спарсили TA_special_rank на Ranking_in_City

data['TA_special_rank'].fillna(data["Ranking_in_City"], inplace=True)
# подсчет сколько типов кухонь предлагает ресторан

def cuisine_count(lst):

    try:

        lst =  lst[1:-1].split(",")

        return len(lst)

    except:

        return 1

    

data['cuisine_count'] = data["Cuisine Style"].apply(cuisine_count)
# переведем кухни в категориальные признаки

#Создание словоря кухонь и их популярности

Cuisine=data["Cuisine Style"].dropna().unique()

cus_count = {}

for lst in Cuisine:

    #lst =  lst[1:-1].split(",")

    lst = ast.literal_eval(lst)

    for i in lst:

        #i = i[1:-1]

        #i = "".join(it.dropwhile(lambda x: not x.isalpha(), i))

        if i not in cus_count:

            cus_count[i] = 1

        else:

            cus_count[i]+=1



#cоздаем словарь из городов, кухни и места по кухни

Cusine_city_dict = {}



def Cusine_city_dict_add(City, cusine_style, rank):

    #print(City, cusine_style, rank)

    try:

        cusine_style = ast.literal_eval(cusine_style)

        #print(cusine_style)

        if len(cusine_style) != 0:

            if City not in Cusine_city_dict:

                Cusine_city_dict[City] = {}

                for cousin in cusine_style:

                    Cusine_city_dict[City][cousin] = [rank]

            else:

                for cousin in cusine_style:

                    if cousin not in Cusine_city_dict[City]:

                        #print(Cusine_city_dict[City][cousin])

                        Cusine_city_dict[City][cousin] = [rank]

                    else:

                        Cusine_city_dict[City][cousin].append(rank)

    except:

        #print(City, cusine_style, rank)

        pass

        

#создаю словарь из которого в последствии сделаю датафрейм, обращаюсь из функции к глобальной переменной возможно лучше через класс

cusine_list = []

            

def cusine_popular(city, cusine_style, rank):

    empty_cus_count = {}

    for cusine in cus_count:

        empty_cus_count[cusine] = 1

    

    try:

        #lst =  lst[1:-1].split(",")

        cusine_style = ast.literal_eval(cusine_style)

        for cusine in cusine_style:

            #i = i[1:-1]

            #i = "".join(it.dropwhile(lambda x: not x.isalpha(), i))

            city_cousin_tmp_list = Cusine_city_dict[city][cusine]

            city_cousin_tmp_list.sort()

            tmp_rank = city_cousin_tmp_list.index(rank)/len(city_cousin_tmp_list)

            empty_cus_count[cusine] = tmp_rank

        cusine_list.append(empty_cus_count)

        #return list(empty_cus_count.values())

    except:

        #print(len(list(empty_cus_count.values())))

        cusine_list.append(empty_cus_count)

        #return list(empty_cus_count.values())

# заполняю словарь    

x = data.apply(lambda x: Cusine_city_dict_add(x["City"], x["Cuisine Style"], x["Ranking"]), axis=1)

x = data.apply(lambda x: cusine_popular(x["City"], x["Cuisine Style"], x["Ranking"]), axis=1)

df1 = pd.DataFrame(cusine_list)

data = pd.merge(pd.DataFrame(data),pd.DataFrame(df1),left_index=True,right_index=True)
# рейтинг по АНГЛ отзывам

data["rating_by_Reviews"] = (data["Excellent"]*5 + data["Very_good"]*4 + data["Average"]*3+ data["Poor"]*2 +data["Terrible"]*1)/(data["Excellent"] + data["Very_good"] + data["Average"] + data["Poor"] +data["Terrible"]+1)
# рейтинг по категориям в одно

data["rating_by_category"] = (data["Food"] + data["Service"] + data["Value"]+ data["Atmosphere"])/4
data["rating_by_category_rew"] = (data["rating_by_category"]/10+data["rating_by_Reviews"])/2

data["delta_rew"] = (data["TA_num_rew"] - data["Number of Reviews"])/data["TA_num_rew"]

data['delta_rew'].fillna(data['delta_rew'].mean(), inplace=True)
# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)




# убираем не нужные для модели признаки

data.drop(['Restaurant_id','ID_TA',], axis = 1, inplace=True)

# убираем признаки которые еще не успели обработать

data.drop([ 'URL_TA','Reviews','Cuisine Style','TA_special_rank_cusine'], axis = 1, inplace=True)

# убираем Ranking как не корректную метрику

data.drop([ 'Ranking',], axis = 1, inplace=True)

# пробую убрать данные в которых отзывы

data.drop([ "Excellent","Very_good", "Average", "Poor", "Terrible"], axis = 1, inplace=True)

# убираю ТА ранк как дублирующийся

data.drop([ 'TA_rank',], axis = 1, inplace=True)

data.drop(["Food", "Service", "Value", "Atmosphere"], axis = 1, inplace=True)




# Теперь выделим тестовую часть

train_data = data.query('sample == 1').drop(['sample'], axis=1)



y = train_data.Rating.values            # наш таргет

X = train_data.drop(['Rating'], axis=1)



# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)



train_data.shape, X.shape, X_train.shape, X_test.shape
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

plt.rcParams['figure.figsize'] = (10,5)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')




test_data = data.query('sample == 0').drop(['sample'], axis=1)

test_data = test_data.drop(['Rating'], axis=1)
predict_submission = model.predict(test_data)




sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head()