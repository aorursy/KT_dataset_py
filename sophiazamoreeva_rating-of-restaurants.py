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
RANDOM_SEED = 42

!pip freeze > requirements.txt

CURRENT_DATE = pd.to_datetime('25/06/2020')
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



rest = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
rest.head()
# Функция для подсчета пустых значений

def missing_values_table(df):

        # Количество пропущенных значений

        mis_val = df.isnull().sum()

        

        # Процент пропущенных значений

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        #Таблица с результатами

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Переименовываем колонки

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Сортируем значения по проценту

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Печать дополнительной информации

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

       

        return mis_val_table_ren_columns
missing_values_table(rest)
plt.subplots(figsize=(15, 15))

sns.heatmap(rest.isnull())
rest['code_Restaurant_id'] = rest['Restaurant_id'].apply(lambda x: float(x[3:]))
sns.heatmap(rest.corr(),annot=True)
rest.drop(['code_Restaurant_id'], axis=1, inplace=True)
# в переменной 9283 (23.2%) пропущенных значений 

# сохраним эту информацию

rest['NAN_Cuisine Style'] = pd.isna(rest['Cuisine Style']).astype('float64') 



# заполним пропуски значением 'Other'

rest['Cuisine Style'] = rest['Cuisine Style'].fillna("['Other']")
# проведем обработку значений переменной

rest['Cuisine Style'] = rest['Cuisine Style'].str.findall(r"'(\b.*?\b)'") 



temp_list = rest['Cuisine Style'].tolist()



def list_unrar(list_of_lists):

    result=[]

    for lst in list_of_lists:

        result.extend(lst)

    return result



temp_counter=Counter(list_unrar(temp_list))
# сформируем список достаточно уникальных кухонь и сформируем на его основе новый признак

list_of_unique_Cuisine = [x[0] for x in temp_counter.most_common()[-16:]]

rest['unique_Cuisine_Style'] = rest['Cuisine Style'].apply(lambda x: 1 if len(set(x) & set(list_of_unique_Cuisine))>0  else 0).astype('float64')
for cuisine in temp_counter:

    rest[cuisine] = rest['Cuisine Style'].apply(lambda x: 1 if cuisine in x else 0 ).astype('float64')



# генерируем новый признак кол-во кухонь в ресторане

rest['count_Cuisine_Style'] = rest['Cuisine Style'].apply(lambda x: len(x)).astype('float64')
# в переменной очень много пропусков 13886 (34.7%)

# сохраним информацию о пропусках чтобы не потерять

rest['NaN_Price Range'] = pd.isna(rest['Price Range']).astype('float64') 



# заполним значения в переменной по словарю

dic_value_Price = {'$':1,'$$ - $$$':2,'$$$$':3}

rest['Price_Range']=rest['Price Range'].map(lambda x: dic_value_Price.get(x,x))



# 18412 ресторанов это более 70% из заполненной информации имеют средний параметр цены

# поэтому заполняем пропуски двойкой (2)

rest['Price_Range'] = rest['Price_Range'].fillna(2)
# в переменной 2543 (6.4%) пропущенных значений 

# сохраним эту информацию

rest['NAN_Number of Reviews'] = pd.isna(rest['Number of Reviews']).astype('float64')



# для удобства изменим название столбца

rest.rename(columns={'Number of Reviews': 'Number_of_Reviews'}, inplace=True)
rest['Reviews'] = rest['Reviews'].fillna('[[], []]')

rest['empty_Reviews'] = (rest['Reviews']=='[[], []]').astype('float64')
rest['date_of_Review'] = rest['Reviews'].str.findall('\d+/\d+/\d+')

rest['len_date'] = rest['date_of_Review'].apply(lambda x: len(x))
print("кол-во значений Reviews с тремя датами :=" , len(rest[rest['len_date']==3]))

print("значения Reviews с тремя датами :=")

temp_list = rest[rest['len_date']==3].Reviews.to_list()

display(rest[rest['len_date']==3].Reviews.to_list())

print("даты после обработки регулярными выражениями:")

display([re.findall('\d+/\d+/\d+', x) for x in temp_list])
rest['len_date'].date_of_Review = rest[rest['len_date']==3].date_of_Review.apply(lambda x: x.pop(0))
print("кол-во значений Reviews с одной датой :=" , len(rest[rest['len_date']==1]))

display(rest[rest['len_date']==1].Reviews[:4])
rest['one_Review'] = (rest['len_date']==1).astype('float64')



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



rest['day_to_now'] = rest.apply(time_to_now, axis = 1).dt.days

rest['day_between_Reviews'] = rest[rest['len_date']==2].apply(time_between_Reviews, axis = 1).dt.days
rest['out_day_between_Reviews'] = (rest['day_between_Reviews']==0).astype('float64')



# и удаляем выбросы

rest.loc[rest['day_between_Reviews']==0, 'day_between_Reviews'] = None
rest['day_to_now'].isna().sum()
rest['day_between_Reviews'].isna().sum()
rest['code_ID_TA'] = rest['ID_TA'].apply(lambda x: float(x[1:]))
rest['code_after_g_URL_TA'] = rest['URL_TA'].str.split('-').apply(lambda x: x[1][1:]).astype('float64')
City_dummies = pd.get_dummies(rest['City'], dummy_na=False).astype('float64')

rest = pd.concat([rest,City_dummies], axis=1)
le = LabelEncoder()

le.fit(rest['City'])

rest['code_City'] = le.transform(rest['City'])
NotCapitalCity = ['Barcelona', 'Milan', 'Hamburg', 'Munich', 

                          'Lyon', 'Zurich', 'Oporto', 'Geneva', 'Krakow']

rest['Capital_City'] = rest['City'].apply(lambda x: 0.0 if x in NotCapitalCity else 1.0)
countries = {'London' : 'England', 'Paris' : 'France', 'Madrid' : 'Spain', 

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

rest['Сountry'] = rest.apply(lambda row: countries[row['City']], axis = 1)



le = LabelEncoder()

le.fit(rest['Сountry'])

rest['code_Сountry'] = le.transform(rest['Сountry'])
city_population= {'London' : 8908, 'Paris' : 2206, 'Madrid' : 3223, 'Barcelona' : 1620, 

                        'Berlin' : 6010, 'Milan' : 1366, 'Rome' : 2872, 'Prague' : 1308, 

                        'Lisbon' : 506, 'Vienna' : 1888, 'Amsterdam' : 860, 'Brussels' : 179, 

                        'Hamburg' : 1841, 'Munich' : 1457, 'Lyon' : 506, 'Stockholm' : 961, 

                        'Budapest' : 1752, 'Warsaw' : 1764, 'Dublin' : 553, 

                        'Copenhagen' : 616, 'Athens' : 665, 'Edinburgh' : 513, 

                        'Zurich' : 415, 'Oporto' : 240, 'Geneva' : 201, 'Krakow' : 769, 

                        'Oslo' : 681, 'Helsinki' : 643, 'Bratislava' : 426, 

                        'Luxembourg' : 119, 'Ljubljana' : 284}

rest['Сity_population'] = rest.apply(lambda row: city_population[row['City']], axis = 1)
rest.Ranking.hist()
# У нас много ресторанов, которые не дотягивают и до 2500 места в своем городе, а что там по городам?

plt.rcParams['figure.figsize'] = (12,6)

df_train['City'].value_counts(ascending=True).plot(kind='barh')
for x in (df_train['City'].value_counts())[0:10].index:

    df_train['Ranking'][df_train['City'] == x].hist(bins=100)

plt.show()
mean_Ranking= rest.groupby(['City'])['Ranking'].mean()

count_Restorant_in_City = rest['City'].value_counts(ascending=False)

rest['mean_Ranking'] = rest['City'].apply(lambda x: mean_Ranking[x])

rest['count_Restorant_in_City'] = rest['City'].apply(lambda x: count_Restorant_in_City[x])

rest['norm_Ranking'] = (rest['Ranking'] - rest['mean_Ranking']) / rest['count_Restorant_in_City']
for x in (rest['City'].value_counts())[0:10].index:

    rest['norm_Ranking'][rest['City'] == x].hist(bins=100)

plt.show()
rest['norm_Population'] = rest['Сity_population']/rest['count_Restorant_in_City']
rest.drop(['Restaurant_id','City', 'Cuisine Style', 'Price Range', 'Reviews', 'URL_TA', 'ID_TA', 'date_of_Review', 'len_date', 'Сountry', 'Сity_population', 'mean_Ranking', 'count_Restorant_in_City', ], axis=1, inplace=True, errors='ignore')
display(rest.head())
# функция для стандартизации

def StandardScaler_column(d_col):

    scaler = StandardScaler()

    scaler.fit(rest[[d_col]])

    return scaler.transform(rest[[d_col]])

# стандартизируем все столбцы кроме целевой и Sample

for i  in list(rest.columns):

    if i not in ['Rating','Sample']:

        rest[i] = StandardScaler_column(i)

        if len(rest[rest[i].isna()]) < len(rest):

            rest[i] = rest[i].fillna(0)
display(rest.describe().head(1))
train_data = rest.query('Sample == 1').drop(['Sample'], axis=1)

test_data = rest.query('Sample == 0').drop(['Sample'], axis=1)



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
plt.rcParams['figure.figsize'] = (10,10)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(25).plot(kind='barh')
train_data = rest.query('Sample == 1').drop(['Sample'], axis=1)

test_data = rest.query('Sample == 0').drop(['Sample','Rating'], axis=1)

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

sample_submission.head(10)
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head()