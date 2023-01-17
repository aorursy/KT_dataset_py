import pandas as pd

import re

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



%matplotlib inline



# Объединяем вместе тренировочный и боевой датасет

restaurants_test = pd.read_csv('../input/sf-dst-restaurant-rating/kaggle_task.csv') #Боевой датасет

restaurants_train = pd.read_csv('../input/sf-dst-restaurant-rating/main_task.csv')  #Тренировочный датасет

restaurants_train['Test'] = 0

restaurants_test['Test'] = 1



# Объединяем датасеты в один общий

restaurants = restaurants_test.append(restaurants_train, sort=False).reset_index(drop=True)
is_capital={'London':1, 'Paris':1,'Madrid':1,'Barcelona':0,'Berlin':1,'Milan':0,'Rome':1,'Prague':1,'Lisbon':1,

            'Vienna':1, 'Amsterdam':1,'Brussels':1,'Hamburg':0,'Munich':0,'Lyon':0,'Stockholm':1,'Budapest':1,'Warsaw':1,

            'Dublin':1,'Copenhagen':1, 'Athens':1,'Edinburgh':1,'Zurich':0,'Oporto':0,'Geneva':0,'Krakow':0,'Oslo':1,

            'Helsinki':1,'Bratislava':1,'Luxembourg':1, 'Ljubljana':1}



population = {'Paris': 2190327, 'Stockholm': 961609, 'London': 8908081, 'Berlin': 3644826, 'Munich': 1456039, 'Oporto': 237591,

              'Milan': 1378689,'Bratislava': 432864, 'Vienna': 1821582, 'Rome': 4355725, 'Barcelona': 1620343, 

              'Madrid': 3223334,'Dublin': 1173179,'Brussels': 179277, 'Zurich': 428737, 'Warsaw': 1758143, 'Budapest': 1752286,

              'Copenhagen': 615993,'Amsterdam': 857713,'Lyon': 506615, 'Hamburg': 1841179,'Lisbon': 505526, 'Prague': 1301132, 

              'Oslo': 673469,'Helsinki': 643272,'Edinburgh': 488100,'Geneva': 200548, 'Ljubljana': 284355,'Athens': 664046, 

              'Luxembourg': 115227,'Krakow': 769498}
# 1. Restaurant_id переводим в число

#restaurants['Restaurant_id'] = restaurants['Restaurant_id'].apply(lambda x: int(str(x)[3:]))



# 2. Price Range заменим числовой шкалой и заполним пропуски средним ценовым диапазоном

restaurants['Price Range'] = restaurants['Price Range'].map({'$':1,'$$ - $$$':2,'$$$$':3}).fillna(2)
# 3. Cuisine Style - трансформируем из строки в список строк и сразу добавим новый признак - кол-во кухонь в ресторане



# Функция осуществляющая конвертацию в список кухонь

def create_cuisine_list(row):



    if str(row) != 'nan':

        cusines = row.split(sep='\'')



        #Чистим список с кухнями от мусора

        for cuisine in cusines:

            if cuisine[0]=='[':

                cusines.remove(cuisine)

            elif cuisine[0]==',':

                cusines.remove(cuisine)

            elif cuisine[0]==']':

                cusines.remove(cuisine)

        return cusines

    else:

        return []

    

restaurants['Cuisine Style'] = restaurants['Cuisine Style'].apply(create_cuisine_list)



# Новый признак количество кухонб в конкретном ресторане

restaurants['# of Cusine'] = restaurants['Cuisine Style'].apply(lambda x: len(x) if len(x)>0 else 1)

# 4. Разбоор  поля с отзывами и их датами

def create_list_of_reviews(x):

    review_pattern = re.compile("[A-Z][A-Za-z\s \.]+")

    reviews = review_pattern.findall(x)

    for review in reviews:

        pass

    return reviews



# Дата последнего отзыва

def create_list_of_dates_last(x):

    date_pattern = re.compile("\d\d/\d\d/\d\d\d\d")

    dates = date_pattern.findall(x)

    if len(dates)==2:

        return max(pd.to_datetime(dates[0]),pd.to_datetime(dates[1]))

    elif len(dates)==1:

        return pd.to_datetime(dates[0])

    else:

        return None



# Дата предпоследнего отзыва

def create_list_of_dates_perv(x):

    date_pattern = re.compile("\d\d/\d\d/\d\d\d\d")

    dates = date_pattern.findall(x)

    if len(dates)==2:

        return min(pd.to_datetime(dates[0]), pd.to_datetime(dates[1]))

    elif len(dates)==1:

        return pd.to_datetime(dates[0])

    else:

        return None



# Список отзывов

restaurants['Reviews'].fillna('[[], []]', inplace = True)

restaurants['List of reviews'] = restaurants['Reviews'].apply(create_list_of_reviews)



# Добавляем столбец последенго отзыва

restaurants['Last_review_date'] = restaurants['Reviews'].apply(create_list_of_dates_last)



# Добавляем столбец с предыдущим отзывом

restaurants['Perv_review_date'] = restaurants['Reviews'].apply(create_list_of_dates_perv)



# Переводим даты в числовой формат (чичло дней с 1970 года) и заполняем пропуски средними значениями

restaurants['Last_review_date'] =(restaurants['Last_review_date'] - pd.to_datetime('1970-01-01')).dt.days

restaurants['Last_review_date'].fillna(restaurants['Last_review_date'].median(), inplace = True)

restaurants['Perv_review_date'] =(restaurants['Perv_review_date'] - pd.to_datetime('1970-01-01')).dt.days

restaurants['Perv_review_date'].fillna(restaurants['Perv_review_date'].median(), inplace = True)



# Разница в днх между двумя последними отзывами

restaurants['Dates delta'] = (restaurants['Last_review_date'] - restaurants['Perv_review_date'])



# Заполняем пропуски в количестве отзывов 0

restaurants['Number of Reviews'].fillna(0 ,inplace = True)



# Добавление  нового признака - население города

restaurants['City Population'] = restaurants['City'].apply(lambda x:population[x])



# Удаляем лишние столбцы

restaurants.drop(['Reviews', 'URL_TA', 'ID_TA'], axis = 1, inplace = True)



#restaurants['Is Capital'] = restaurants['City'].apply(lambda x:is_capital[x])

# Функция голосования по ключевым словам

def ReviewsAnalyzer(reviews):

    res=0

    for review in reviews:

        review=review.lower()

        res+=review.count('good')

        res+=review.count('excellent')

        res+=review.count('nice')

        res+=review.count('great')

        res+=review.count('nice')

        res+=review.count('gem')

        res+=review.count('lovely')

        res+=review.count('delicious')

        res+=review.count('best')

        res+=review.count('perfect')

        res+=review.count('amazing')

        res+=review.count('love')

        res+=review.count('tasty')

        res+=review.count('clean')

        res+=review.count('fresh')

        res+=review.count('yamm')

        res+=review.count('bellyful')

        res+=review.count('pleasant')

        res-=review.count('bad')

        res-=review.count('awful')

        res-=review.count("don\\'t")

        res-=review.count('nothing')

        res-=review.count('wast')

        res-=review.count('trash')

        res-=review.count('worst')

        res-=review.count('overpriced')

        res-=review.count('disappoint')

        res-=review.count('loud')

        res-=review.count('nothing')

        res-=review.count('worth')

        res-=review.count('tasteless')

        res-=review.count('nasty')

        res-=review.count('disgus')

        res-=review.count('dirt')

        res-=review.count("not")

    return res



restaurants['Reviews'] = restaurants['List of reviews'].apply(ReviewsAnalyzer)

restaurants.drop(['List of reviews'], axis = 1, inplace = True)
# Подсчёт статистики встречаемости кухни нужно было для ответы на вопросы модуля

# import operator

# different_cusines =[]

# for cusines in restaurants['Cuisine Style']:

#     for cusine in cusines:

#         if cusine not in different_cusines:

#             different_cusines.append(cusine)



# # Делаем словарь где каждой кухни соответствует список позиций в датасете, где эта кухня встретилась

# cusines_stat={cusine : 0 for cusine in different_cusines}



# for i in range(restaurants.first_valid_index(), restaurants.last_valid_index()):

#     for cusine in restaurants['Cuisine Style'][i]:

#         cusines_stat[cusine]+=1





# sorted_cusines = sorted(cusines_stat.items(), key=operator.itemgetter(1))

# sorted_cusines



        
# Эксперименты с тем как наличие той или иной кухни влияет на качество модели, но так и не удалось выявить зависимости

def find_item(cell):

    if item in cell:

        return 1

    return 0



for item in ['Italian']:

     restaurants[item] = restaurants['Cuisine Style'].apply(find_item)



# Эксперимент с городами как оказалось расположение в ресторан в Риме влияет на итоговую оценку

for city in ['Rome']:

    restaurants[city] = restaurants['City'].apply(lambda x: 1 if x==city else 0)

#среднее значение количества кухонь в ресторанах конкретного города

restaurants['Mean_Cuisine_Quantity'] = restaurants['City'].map(restaurants.groupby('City')['# of Cusine'].mean())
# Отношение Ranking к кол-ву ресторанов в городе

restaurants = restaurants.merge(restaurants.City.value_counts().to_frame(name = '# of Rest in City'), how='left', left_on='City', right_index=True)

restaurants['Rank/Rest_in_city'] = restaurants.Ranking / restaurants['# of Rest in City']
#Отношение Ranking к количеству населения города

restaurants['Rank/Population'] = restaurants.Ranking / restaurants['City Population']
restaurants
fig, ax = plt.subplots(figsize=(10,10)) 

sns.heatmap(restaurants.corr(), cmap='coolwarm', ax=ax, )
# Удаляем оставшиеся категориальные столбцы

restaurants.drop(['City','Cuisine Style'], axis = 1, inplace = True)                 
train_data = restaurants.query('Test == 0').drop(['Test'], axis=1)

test_data = restaurants.query('Test == 1').drop(['Test'], axis=1)

y = train_data.Rating.values           

X = train_data.drop(['Restaurant_id','Rating'], axis=1)

X_test = test_data.drop(['Restaurant_id','Rating'], axis=1)
# Разбиваем датафрейм на части, необходимые для обучения и тестирования модели

# Х - данные с информацией о ресторанах, у - целевая переменная (рейтинги ресторанов)

#X = restaurants_train.drop(['Restaurant_id', 'Rating'], axis = 1)

#y = restaurants_train['Rating']

# Загружаем специальный инструмент для разбивки:

from sklearn.model_selection import train_test_split

# Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.

# Для тестирования мы будем использовать 25% от исходного датасета.

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#Создаём, обучаем и тестируем модель

# Импортируем необходимые библиотеки:

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели

# Создаём модель

regr = RandomForestRegressor(n_estimators=100)



# Обучаем модель на тестовом наборе данных

#regr.fit(X_train, y_train)

regr.fit(X,y)

# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

#y_pred = regr.predict(X_test)



# Округляем результат до 0.5

def round05(row):

    return (round(row*2,0)/2)



new_round = np.vectorize(round05)

y_pred_round = new_round(regr.predict(X_test))



# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

#print('MAE:', metrics.mean_absolute_error(y_test, y_pred_round))
plt.rcParams['figure.figsize'] = (10,5)

feat_importances = pd.Series(regr.feature_importances_, index=X.columns)

feat_importances.nlargest(20).plot(kind='barh')
# Сохранение результатов

#test_data['Rating'] = y_pred_round

#test_data[['Restaurant_id','Rating']].to_csv('solution.csv', index = False)
#test_data