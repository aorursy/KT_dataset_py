import pandas as pd

import pandas_profiling

import numpy as np

from sklearn.model_selection import train_test_split 

from sklearn.ensemble import RandomForestRegressor 

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from collections import Counter

import datetime

import os

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline

import re

import math

import copy

from IPython.display import display

pd.options.mode.chained_assignment = None
import my_module as my
RANDOM_SEED = 42

!pip freeze > requirements.txt

CURRENT_DATE = pd.to_datetime('15/06/2020')
path_to_file = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(path_to_file+'main_task.csv')

df_test = pd.read_csv(path_to_file+'kaggle_task.csv')

pd.set_option('display.max_columns', 200)

display(df_train.head(2))

display(df_test.head(2))
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['Sample'] = 1 # помечаем где у нас трейн

df_test['Sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



df = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
# предварительный анализ данных с помощью библиотеки pandas_profiling закомментирован так как отображается не корректно

# pandas_profiling.ProfileReport(df_train)
df['code_Restaurant_id'] = df['Restaurant_id'].apply(lambda x: float(x[3:]))
# в переменной 9283 (23.2%) пропущенных значений 

# сохраним эту информацию

df['NAN_Cuisine Style'] = pd.isna(df['Cuisine Style']).astype('float64') 



# заполним пропуски значением 'Other'

df['Cuisine Style'] = df['Cuisine Style'].fillna("['Other']")



# закодируем значения в переменной до их преобразования

le = LabelEncoder()

le.fit(df['Cuisine Style'])

df['code_Cuisine Style'] = le.transform(df['Cuisine Style'])
# проведем обработку значений переменной

df['Cuisine Style'] = df['Cuisine Style'].str.findall(r"'(\b.*?\b)'") 



temp_list = df['Cuisine Style'].tolist()



def list_unrar(list_of_lists):

    result=[]

    for lst in list_of_lists:

      result.extend(lst)

    return result



temp_counter=Counter(list_unrar(temp_list))
# сформируем список достаточно уникальных кухонь и сформируем на его основе новый признак

list_of_unique_Cuisine = [x[0] for x in temp_counter.most_common()[-16:]]

df['unique_Cuisine_Style'] = df['Cuisine Style'].apply(lambda x: 1 if len(set(x) & set(list_of_unique_Cuisine))>0  else 0).astype('float64')
for cuisine in temp_counter:

    df[cuisine] = df['Cuisine Style'].apply(lambda x: 1 if cuisine in x else 0 ).astype('float64')



# генерируем новый признак кол-во кухонь в ресторане

df['count_Cuisine_Style'] = df['Cuisine Style'].apply(lambda x: len(x)).astype('float64')
my.four_plot_with_log('count_Cuisine', df[df['Sample'] == 1].count_Cuisine_Style)
# видимо в значениях есть нули, это значит что это не Other, а просто не заполненные. проверим

my.describe_without_plots('count_Cuisine_Style', df[df['Sample'] == 1].count_Cuisine_Style)
# в переменной очень много пропусков 13886 (34.7%)

# сохраним информацию о пропусках чтобы не потерять

df['NaN_Price Range'] = pd.isna(df['Price Range']).astype('float64') 



# заполним значения в переменной по словарю

dic_value_Price = {'$':1,'$$ - $$$':2,'$$$$':3}

df['Price_Range']=df['Price Range'].map(lambda x: dic_value_Price.get(x,x))



# 18412 ресторанов это более 70% из заполненной информации имеют средний параметр цены

# поэтому заполняем пропуски двойкой (2)

df['Price_Range'] = df['Price_Range'].fillna(2)
# проверим полученный критерий

my.describe_with_hist('Price Range', df[df['Sample'] == 1].Price_Range)
# в переменной 2543 (6.4%) пропущенных значений 

# сохраним эту информацию

df['NAN_Number of Reviews'] = pd.isna(df['Number of Reviews']).astype('float64')



# для удобства изменим название столбца

df.rename(columns={'Number of Reviews': 'Number_of_Reviews'}, inplace=True)
my.four_plot_with_log2('Number_of_Reviews', df[df['Sample'] == 1])
# выбросы есть, купол распределения с точками перегиба, необходимо посмотреть на гистограмму по крупнее

my.big_hist_log('Number_of_Reviews', df[df['Sample'] == 1])
# посмотрим на границы

my.borders_of_outliers('Number_of_Reviews', df[df['Sample'] == 1], log=True)
# выбросов не так много, удалим их, предварительно сохранив информацию о них

df['outliers_Number_of_Reviews'] = pd.DataFrame(df['Number_of_Reviews']>5252).astype('float64')

df.loc[df['Number_of_Reviews']>5252, 'Number_of_Reviews']=None
# в ревью нет пропусков, но 6471 строк со значением [[], []]. По сути это пустые строки сохраним их 

df['empty_Reviews'] = (df['Reviews']=='[[], []]').astype('float64')



# анализ тестовой базы выявил два пропуска, несмотря на то, что pandas.profiling на тренировочной базе пропусков не выявил, заполним их '[[], []]' и закинем в empty_Reviews

df['Reviews'] = df['Reviews'].fillna('[[], []]')

df['empty_Reviews'] = (df['Reviews']=='[[], []]').astype('float64')
# вытащим дату из ревью и создадим новые критерии

df['date_of_Review'] = df['Reviews'].str.findall('\d+/\d+/\d+')

df['len_date'] = df['date_of_Review'].apply(lambda x: len(x))



# проверим длину дат, на случай если там больше или меньше двух (2)

my.describe_without_plots('len_date', df[df['Sample'] == 1].len_date)
# есть значение 3 надо разобраться что там

print("кол-во значений Reviews с тремя датами :=" , len(df[df['len_date']==3]))

print("значения Reviews с тремя датами :=")

temp_list = df[df['len_date']==3].Reviews.to_list()

display(df[df['len_date']==3].Reviews.to_list())

print("даты после обработки регулярными выражениями:")

display([re.findall('\d+/\d+/\d+', x) for x in temp_list])
# видим что люди указывали даты в отзывах и эти даты попали в обработку

# из-за этого возникнут ошибки так как даты не верные и их формат отличается и формата выгрузки

# при этом таких строк всего четыре (4), можно было бы их не исправлять а выбросить потому что 17 

# год явно приведет к выбросу с которым надо будет разбираться. Выбрасывать жалко, тогда исправим,

# тем более, что это достачно просто



df['len_date'].date_of_Review = df[df['len_date']==3].date_of_Review.apply(lambda x: x.pop(0))
# также есть значение 1 надо разобраться что там

print("кол-во значений Reviews с одной датой :=" , len(df[df['len_date']==1]))

display(df[df['len_date']==1].Reviews[:4])
# оказалось, что есть отзывы с одним (1) отзывом и их достаточно много 5680 из (40000-6471) это 17%

# сохраним это на всякий случай, чтобы не потерять

df['one_Review'] = (df['len_date']==1).astype('float64')



# заполним перерыв между отзывами (по отзывам где len = 2) и насколько давно был сделан последний самый свежий отзыв

# создадим для этого функции:

def time_to_now(row):

    if row['date_of_Review'] == []:

        return None

    return datetime.datetime.now() - pd.to_datetime(row['date_of_Review']).max()



def time_between_Reviews(row):

    if row['date_of_Review'] == []:

        return None

    return pd.to_datetime(row['date_of_Review']).max() - pd.to_datetime(row['date_of_Review']).min()



df['day_to_now'] = df.apply(time_to_now, axis = 1).dt.days

df['day_between_Reviews'] = df[df['len_date']==2].apply(time_between_Reviews, axis = 1).dt.days
# посмотрим на критерий day_to_now - это насколько давно был сделан последний самый свежий отзыв в днях

my.four_plot_with_log2('day_to_now', df[df['Sample'] == 1])
# выбросов достаточно много посмотрим на границы выбросов

my.borders_of_outliers('day_to_now', df[df['Sample'] == 1], log=True)
# жалко терять 2356 значений 

# посмотрим на гистограмму крупно

my.big_hist_log('day_to_now', df[df['Sample'] == 1])
# никаких очевидных аномалий не видно 

# посмотрим основные статистики

my.describe_without_plots('day_to_now', df[df['Sample'] == 1].day_to_now)
# теперь посмотрим на разницу в датах отзывов в днях 

my.four_plot_with_log2('day_between_Reviews', df[df['Sample'] == 1])
my.big_hist_log('day_between_Reviews', df[df['Sample'] == 1])
my.borders_of_outliers('day_between_Reviews', df[df['Sample'] == 1], log=True)
# кол-во выбросов 495 (1.2%) - это статистически не значимо, но мы пока сохраняем информацию о выбросе, а потом проверим его важность в модели

df['out_day_between_Reviews'] = (df['day_between_Reviews']==0).astype('float64')



# и удаляем выбросы

df.loc[df['day_between_Reviews']==0, 'day_between_Reviews'] = None
my.describe_without_plots('day_between_Reviews', df[df['Sample'] == 1].day_between_Reviews)
df['code_ID_TA'] = df['ID_TA'].apply(lambda x: float(x[1:]))
df['code_after_g_URL_TA'] = df['URL_TA'].str.split('-').apply(lambda x: x[1][1:]).astype('float64')
df_City_dummies = pd.get_dummies(df['City'], dummy_na=False).astype('float64')

df = pd.concat([df,df_City_dummies], axis=1)
le = LabelEncoder()

le.fit(df['City'])

df['code_City'] = le.transform(df['City'])
list_Of_NotCapitalCity = ['Barcelona', 'Milan', 'Hamburg', 'Munich', 

                          'Lyon', 'Zurich', 'Oporto', 'Geneva', 'Krakow']

df['Capital_City'] = df['City'].apply(lambda x: 0.0 if x in list_Of_NotCapitalCity else 1.0)
dict_Сountries = {'London' : 'England', 'Paris' : 'France', 'Madrid' : 'Spain', 

                  'Barcelona' : 'Spain', 'Berlin' : 'Germany', 'Milan' : 'Italy', 

                  'Rome' : 'Italy', 'Prague' : 'Czech_c', 'Lisbon' : 'Portugal', 

                  'Vienna' : 'Austria', 'Amsterdam' : 'Holland', 

                  'Brussels' : 'Belgium', 'Hamburg' : 'Germany', 'Munich' : 'Germany', 

                  'Lyon' : 'France', 'Stockholm' : 'Sweden', 'Budapest' : 'Romania', 

                  'Warsaw' : 'Poland', 'Dublin' : 'Ireland', 'Copenhagen' : 'Denmark', 

                  'Athens' : 'Greece', 'Edinburgh' : 'Scotland', 'Zurich' : 'Switzerland', 

                  'Oporto' : 'Portugal', 'Geneva' : 'Switzerland', 'Krakow' : 'Poland', 

                  'Oslo' : 'Norway', 'Helsinki' : 'Finland', 'Bratislava' : 'Slovakia', 

                  'Luxembourg' : 'Luxembourg_c', 'Ljubljana' : 'Slovenia'}

df['Сountry'] = df.apply(lambda row: dict_Сountries[row['City']], axis = 1)



le = LabelEncoder()

le.fit(df['Сountry'])

df['code_Сountry'] = le.transform(df['Сountry'])
dict_Сity_population= {'London' : 8908, 'Paris' : 2206, 'Madrid' : 3223, 'Barcelona' : 1620, 

                        'Berlin' : 6010, 'Milan' : 1366, 'Rome' : 2872, 'Prague' : 1308, 

                        'Lisbon' : 506, 'Vienna' : 1888, 'Amsterdam' : 860, 'Brussels' : 179, 

                        'Hamburg' : 1841, 'Munich' : 1457, 'Lyon' : 506, 'Stockholm' : 961, 

                        'Budapest' : 1752, 'Warsaw' : 1764, 'Dublin' : 553, 

                        'Copenhagen' : 616, 'Athens' : 665, 'Edinburgh' : 513, 

                        'Zurich' : 415, 'Oporto' : 240, 'Geneva' : 201, 'Krakow' : 769, 

                        'Oslo' : 681, 'Helsinki' : 643, 'Bratislava' : 426, 

                        'Luxembourg' : 119, 'Ljubljana' : 284}

df['Сity_population'] = df.apply(lambda row: dict_Сity_population[row['City']], axis = 1)
my.four_plot_with_log2('Ranking', df[df['Sample'] == 1])
my.big_hist('Ranking', df[df['Sample'] == 1])
# У нас много ресторанов, которые не дотягивают и до 2500 места в своем городе, а что там по городам?

plt.rcParams['figure.figsize'] = (12,6)

df_train['City'].value_counts(ascending=True).plot(kind='barh')
# посмотрим на топ 10 городов

for x in (df_train['City'].value_counts())[0:10].index:

    df_train['Ranking'][df_train['City'] == x].hist(bins=100)

plt.show()
# Получается, что Ranking имеет нормальное распределение, 

# просто в больших городах больше ресторанов, из-за мы этого имеем смещение

# необходимо отнормировать критерий Ranking по городам City

mean_Ranking_on_City = df.groupby(['City'])['Ranking'].mean()

count_Restorant_in_City = df['City'].value_counts(ascending=False)

df['mean_Ranking_on_City'] = df['City'].apply(lambda x: mean_Ranking_on_City[x])

df['count_Restorant_in_City'] = df['City'].apply(lambda x: count_Restorant_in_City[x])

df['norm_Ranking_on_Rest_in_City'] = (df['Ranking'] - df['mean_Ranking_on_City']) / df['count_Restorant_in_City']
# посмотрим что получилось на топ 10 городов

for x in (df['City'].value_counts())[0:10].index:

    df['norm_Ranking_on_Rest_in_City'][df['City'] == x].hist(bins=100)

plt.show()
max_Ranking_on_City = df.groupby(['City'])['Ranking'].max()

df['max_Ranking_on_City'] = df['City'].apply(lambda x: max_Ranking_on_City[x])

df['norm_Ranking_on_maxRank_in_City'] = (df['Ranking'] - df['mean_Ranking_on_City']) / df['max_Ranking_on_City']
for x in (df['City'].value_counts())[0:10].index:

    df['norm_Ranking_on_maxRank_in_City'][df['City'] == x].hist(bins=100)

plt.show()
# критерий Ranking по населению в городах Population_Сity

mean_Ranking_on_City = df.groupby(['City'])['Ranking'].mean()

df['mean_Ranking_on_City'] = df['City'].apply(lambda x: mean_Ranking_on_City[x])

df['norm_Ranking_on_Popul_in_City'] = (df['Ranking'] - df['mean_Ranking_on_City']) / df['Сity_population']



for x in (df['City'].value_counts())[0:10].index:

    df['norm_Ranking_on_Popul_in_City'][df['City'] == x].hist(bins=100)

plt.show()
df['norm_Population_on_Rest'] = df['Сity_population']/df['count_Restorant_in_City']
display(df.head(2))
df.drop(['Restaurant_id', 'City', 'Cuisine Style', 'Price Range', 'Reviews', 'URL_TA', 'ID_TA', 'date_of_Review', 'len_date', 'Сountry', 'Сity_population', 'mean_Ranking_on_City', 'count_Restorant_in_City', 'max_Ranking_on_City', ], axis=1, inplace=True, errors='ignore')
# функция для стандартизации

def StandardScaler_column(d_col):

    scaler = StandardScaler()

    scaler.fit(df[[d_col]])

    return scaler.transform(df[[d_col]])

# стандартизируем все столбцы кроме целевой и Sample

for i  in list(df.columns):

    if i not in ['Rating','Sample']:

        df[i] = StandardScaler_column(i)

        if len(df[df[i].isna()]) < len(df):

            df[i] = df[i].fillna(0)
# проверяем заполнение

display(df.describe().head(1))
train_data = df.query('Sample == 1').drop(['Sample'], axis=1)

test_data = df.query('Sample == 0').drop(['Sample'], axis=1)



y = train_data.Rating.values            # наш таргет

X = train_data.drop(['Rating'], axis=1)
# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных

# выделим 20% данных на валидацию (параметр test_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
# проверяем

test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape
# Создаём модель (НАСТРОЙКИ НЕ ТРОГАЕМ)

model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)
# Обучаем модель на тестовом наборе данных

model.fit(X_train, y_train)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

y_pred = model.predict(X_test)
# функция стандартного математического округления

def classic_round(d_num):

    return int(d_num + (0.5 if d_num > 0 else -0.5))



# функция округления кратно 0.5

def my_round(d_pred):

    result = classic_round(d_pred*2)/2

    if result <=5:

        return result

    else:

        return 5

    

# создание функции для векторов np

my_vec_round = np.vectorize(my_round)
y_pred = my_vec_round(y_pred)
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

MAE = metrics.mean_absolute_error(y_test, y_pred)

print('MAE:', MAE)
# в RandomForestRegressor есть возможность вывести самые важные признаки для модели

plt.rcParams['figure.figsize'] = (12,10)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(15).plot(kind='barh')
df_temp = df.loc[df['Sample'] == 1, list(feat_importances.nlargest(15).index[0:15])]

plt.rcParams['figure.figsize'] = (12,6)

ax = sns.heatmap(df_temp.corr(), annot=True, fmt='.2g')

i, k = ax.get_ylim()

ax.set_ylim(i+0.5, k-0.5)
list_temp = list(feat_importances.nlargest(15).index[[9,10]])

display(df_temp[list_temp].corr())
# вспоминаем Резюме по критерию code_Restaurant_id. Удаляем так как была гипотеза о корреляции с Ranking

df.drop(['code_Restaurant_id'], axis=1, inplace=True, errors='ignore')
list_temp = list(feat_importances.nlargest(15).index[[0,1,6,10]])

df_temp[list_temp].corr()
# Метод главных компонент, PCA

C = np.array([

    [       1, 0.999832, 0.800703, 0.574781],

    [0.999832,        1, 0.796851, 0.570877],

    [0.800703, 0.796851,        1, 0.448070],

    [0.574781, 0.570877, 0.448070,        1]]) 

eig_num, eig_v = np.linalg.eig(C)

print(f"вектор главных компонент := {eig_v[:,0]}")
df['norm_Ranking_PCA'] = eig_v[:,0][0]*df['norm_Ranking_on_maxRank_in_City'] + eig_v[:,0][1]*df['norm_Ranking_on_Rest_in_City'] + eig_v[:,0][2]*df['norm_Ranking_on_Popul_in_City']+eig_v[:,0][3]*df['Ranking']
df['Ranking_on_square'] = df['Ranking']* df['Ranking']

df['doble_Ranking'] = df['Ranking']
# блок тестирования закомментирован так как список критериев для удаления с помощью него уже был сгенерирован

# # блок тестирования оптимального набора

# list_ofAllColumnsSortImportant = list(feat_importances.nlargest(len(train_data.columns)-1).index)

# min_MAE = round(MAE,3)

# print(f"min_MAE = {min_MAE}")

# remove_list = []

# log = []

# delta =0.002

# for i in range(0,len(list_ofAllColumnsSortImportant),1):

#     col = list_ofAllColumnsSortImportant[i]

#     print(f"{i}.{col}")

#     ###

#     train_data = df.query('Sample == 1').drop(['Sample']+drop_list2, axis=1)

#     test_data = df.query('Sample == 0').drop(['Sample']+drop_list2, axis=1)



#     y = train_data.Rating.values            # наш таргет

#     X = train_data.drop(['Rating']+[col], axis=1)



#     # Воспользуемся специальной функцие train_test_split для разбивки тестовых данных

#     # выделим 20% данных на валидацию (параметр test_size)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

#     print(test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape)



#     model.fit(X_train, y_train)



#     y_pred = model.predict(X_test)



#     y_pred = my_vec_round(y_pred)

#     temp_MAE = metrics.mean_absolute_error(y_test, y_pred)

#     ###

#     print(temp_MAE)

#     log.append([col, temp_MAE])

#     if round(temp_MAE,3) <= min_MAE-delta:

#         remove_list.append(col)

#         print(f"удаляем:= {col}")

#     else:

#         print(f"не удаляем:= {col}")

# print(f"i={i}")

# print(f"remove_list: {remove_list}")

# print(f"log_list: {log}")
log_list = ['out_day_between_Reviews', 0.163], ['Ranking', 0.1631875], ['Burmese', 0.163375], ['Georgian', 0.1635], ['Fusion', 0.1635625], ['Balti', 0.1635625], ['Tibetan', 0.1636875], ['norm_Ranking_PCA', 0.16375], ['Scottish', 0.16375], ['Eastern European', 0.16375], ['norm_Ranking_on_Rest_in_City', 0.1638125], ['count_Cuisine_Style', 0.1638125], ['Ethiopian', 0.1638125], ['Minority Chinese', 0.1638125], ['Madrid', 0.163875], ['Portuguese', 0.163875], ['Delicatessen', 0.163875], ['Halal', 0.163875], ['Azerbaijani', 0.163875], ['Caucasian', 0.163875], ['Geneva', 0.1639375], ['Scandinavian', 0.1639375], ['Uzbek', 0.1639375], ['Polynesian', 0.164], ['Price_Range', 0.1640625], ['Healthy', 0.1640625], ['Cuban', 0.1640625], ['Dublin', 0.164125], ['Peruvian', 0.164125], ['Salvadoran', 0.164125], ['Sushi', 0.1641875], ['Central European', 0.1641875], ['Venezuelan', 0.1641875], ['Capital_City', 0.16425], ['Gluten Free Options', 0.16425], ['Athens', 0.16425], ['Lebanese', 0.16425], ['Brew Pub', 0.16425], ['Taiwanese', 0.16425], ['Spanish', 0.1643125], ['African', 0.1643125], ['Lisbon', 0.1643125], ['Russian', 0.1643125], ['Central Asian', 0.1643125], ['Singaporean', 0.1643125], ['code_Сountry', 0.164375], ['Italian', 0.164375], ['Wine Bar', 0.164375], ['Barcelona', 0.164375], ['Diner', 0.164375], ['Arabic', 0.164375], ['Filipino', 0.164375], ['Xinjiang', 0.164375], ['Krakow', 0.1644375], ['Lyon', 0.1644375], ['London', 0.1644375], ['unique_Cuisine_Style', 0.1644375], ['European', 0.1645], ['Oporto', 0.1645], ['Contemporary', 0.1645], ['Vienna', 0.1645], ['Croatian', 0.1645], ['Romanian', 0.1645], ['Southwestern', 0.1645], ['Cambodian', 0.1645], ['Vegetarian Friendly', 0.1645625], ['Milan', 0.1645625], ['Egyptian', 0.1645625], ['Fujian', 0.1645625], ['Ranking_on_square', 0.164625], ['German', 0.164625], ['Thai', 0.164625], ['Brussels', 0.164625], ['Cajun & Creole', 0.164625], ['Colombian', 0.164625], ['Kosher', 0.164625], ['Indian', 0.1646875], ['Oslo', 0.1646875], ['Ecuadorean', 0.1646875], ['Latvian', 0.1646875], ['Fast Food', 0.16475], ['Grill', 0.16475], ['Prague', 0.16475], ['Czech', 0.16475], ['New Zealand', 0.16475], ['French', 0.1648125], ['Ljubljana', 0.1648125], ['Argentinean', 0.1648125], ['Stockholm', 0.164875], ['South American', 0.164875], ['Moroccan', 0.164875], ['Jamaican', 0.164875], ['Native American', 0.164875], ['Bar', 0.1649375], ['Asian', 0.1649375], ['Belgian', 0.1649375], ['Luxembourg', 0.1649375], ['Irish', 0.1649375], ['Rome', 0.165], ['Mediterranean', 0.165], ['Vietnamese', 0.165], ['Sri Lankan', 0.165], ['Afghani', 0.165], ['Seafood', 0.1650625], ['Budapest', 0.1650625], ['Hungarian', 0.1650625], ['Vegan Options', 0.165125], ['Chilean', 0.165125], ['Pub', 0.1651875], ['NAN_Cuisine Style', 0.1651875], ['Polish', 0.1651875], ['code_after_g_URL_TA', 0.16525], ['Street Food', 0.16525], ['Copenhagen', 0.16525], ['Ukrainian', 0.16525], ['Israeli', 0.16525], ['Slovenian', 0.16525], ['Albanian', 0.16525], ['NAN_Number of Reviews', 0.1653125], ['Brazilian', 0.1653125], ['Swedish', 0.1653125], ['Paris', 0.165375], ['Middle Eastern', 0.1654375], ['Munich', 0.1654375], ['Cafe', 0.1655], ['Other', 0.1655], ['Chinese', 0.1655], ['Berlin', 0.1655], ['Pizza', 0.1655625], ['International', 0.1655625], ['Pakistani', 0.1655625], ['Swiss', 0.1655625], ['Norwegian', 0.1655625], ['Bratislava', 0.165625], ['Steakhouse', 0.165625], ['Zurich', 0.165625], ['Mongolian', 0.165625], ['Canadian', 0.165625], ['Dutch', 0.1656875], ['British', 0.1656875], ['Danish', 0.1656875], ['Barbecue', 0.1656875], ['Austrian', 0.1656875], ['Gastropub', 0.16575], ['Nepali', 0.16575], ['Latin', 0.16575], ['Caribbean', 0.16575], ['Helsinki', 0.1658125], ['Armenian', 0.1658125], ['Central American', 0.165875], ['Welsh', 0.165875], ['Edinburgh', 0.1660625], ['norm_Ranking_on_maxRank_in_City', 0.166125], ['Tunisian', 0.166125], ['day_between_Reviews', 0.1661875], ['code_City', 0.166375], ['NaN_Price Range', 0.1666875], ['Greek', 0.1668125], ['day_to_now', 0.1669375], ['code_ID_TA', 0.1691875], ['Number_of_Reviews', 0.2305625]
drop_list = ['Australian', 'one_Review', 'outliers_Number_of_Reviews', 'norm_Ranking_on_Popul_in_City', 'Korean', 'Japanese', 'Turkish', 'Malaysian', 'Indonesian', 'Hawaiian', 'code_Cuisine Style', 'norm_Population_on_Rest', 'Amsterdam', 'Hamburg', 'doble_Ranking', 'Warsaw', 'Persian', 'Soups', 'Mexican', 'Bangladeshi', 'Yunnan', 'American', 'empty_Reviews']
train_data = df.query('Sample == 1').drop(['Sample']+drop_list, axis=1)

test_data = df.query('Sample == 0').drop(['Sample','Rating']+drop_list, axis=1)

y = train_data.Rating.values            # наш таргет

X = train_data.drop(['Rating'], axis=1)
sample_submission = pd.read_csv(path_to_file+'sample_submission.csv')

sample_submission.head()
sample_submission.shape, test_data.shape, X.shape, y.shape
model.fit(X, y)
predict_submission = model.predict(test_data)
predict_submission=my_vec_round(predict_submission)

predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head()