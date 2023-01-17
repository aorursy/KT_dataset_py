import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



import re

from sklearn.preprocessing import StandardScaler

from datetime import datetime

import warnings

from scipy.stats import norm

from collections import Counter



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

CURRENT_DATE = pd.to_datetime('15/06/2020')
# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

!pip freeze > requirements.txt
# Функции для определения выбросов в признака

def outliers_iqr(ys):

    quartile_1, quartile_3 = np.percentile(ys, [25, 75])

    iqr = quartile_3 - quartile_1

    lower_bound = quartile_1 - (iqr * 1.5)

    upper_bound = quartile_3 + (iqr * 1.5)

    return np.where((ys > upper_bound) | (ys < lower_bound))[0]



def outliers_z_score(ys, threshold=3):

    mean_y = np.mean(ys)

    std_y = np.std(ys)

    z_scores = [(y - mean_y) / std_y for y in ys]

    return np.where(np.abs(z_scores) > threshold)[0]
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')

pd.set_option('display.max_columns', 200)
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
sns.countplot('City', data=data)
data.City.value_counts()
data = pd.concat([data, pd.get_dummies(data.City)], axis=1)
# Посчитаем количество пропусков

data['Cuisine Style'].isna().sum()
# сохраним эту информацию

data['nan_cuisine_style'] = pd.isna(data['Cuisine Style']).astype('float64') 
# функция для обработки комментариев

def change_reviews(reviews):

    if not pd.isna(reviews):

        reviews = reviews.split("'")[-2::-2]

        new_reviews = []

        for i in reviews:

            if not pd.isna(i):

                new_reviews.append(i.lower())

        if len(new_reviews) == 0:

            return 'a'

        return new_reviews[::-1]

    else:

        return np.nan
#создадим новую колонку с изменёнными значениями

data['lower_case_reviews'] = data['Reviews'].apply(change_reviews)
# найдём все уникальные названия типов кухнь

uniq_cuisine = set()

for i in data['Cuisine Style']:

    if not pd.isna(i):

        cuisines = i.split("'")

        for style in cuisines:

            if style not in "[], '\/":

                uniq_cuisine.add(style.lower())

print(uniq_cuisine)

print(len(uniq_cuisine))
# пропущенные значениями заполним словом 'other'

data['Cuisine Style'].fillna("['other']", inplace=True)

#выделяем данные с пропущенными значениями в cuisine style

unknowen_cuisine_style = data[data['Cuisine Style'] == "['other']"]

#смотрим на ревью в каждой строке и сравниваем есть ли совпадения по тем уникальным кухням которые нам уже удалось найти

for index, row in unknowen_cuisine_style.iterrows():

    try:

        for review in row.lower_case_reviews:

            found_styles = []

            for cuisine_style in uniq_cuisine:

                if cuisine_style.lower() in review:

                    found_styles.append(cuisine_style)

            if len(found_styles) > 0:

                data.at[index, 'Cuisine Style'] = "['{}']".format(found_styles[0].title()) 

            if len(found_styles) > 1:

                data.at[index, 'Cuisine Style'] = "['{}','{}']".format(found_styles[0].title(), found_styles[1].title())

    except:

        print('NaN value found')
unknowen_cuisine_style = data[data['Cuisine Style'] == "['other']"]

print(len(unknowen_cuisine_style))
# проведем обработку значений переменной

data['Cuisine Style'] = data['Cuisine Style'].str.findall(r"'(\b.*?\b)'") 



temp_list = data['Cuisine Style'].tolist()



def list_unrar(list_of_lists):

    result=[]

    for lst in list_of_lists:

        result.extend(lst)

    return result



temp_counter=Counter(list_unrar(temp_list))

temp_counter
# сформируем список уникальных кухонь и сформируем на его основе новый признак

list_of_unique_Cuisine = [x[0] for x in temp_counter.most_common()[-20:]]

data['unique_Cuisine_Style'] = data['Cuisine Style'].apply(lambda x: 1 if len(set(x) & set(list_of_unique_Cuisine))>0  else 0).astype('float64')

for cuisine in temp_counter:

    data[cuisine] = data['Cuisine Style'].apply(lambda x: 1 if cuisine in x else 0 ).astype('float64')



# генерируем новый признак кол-во кухонь в ресторане

data['count_cuisine_style'] = data['Cuisine Style'].apply(lambda x: len(x)).astype('float64')
data['count_cuisine_style'].hist(bins=30)
#Посмотрим какими уникальными данными обладает этот признак

data['Price Range'].unique()
# nan - 'nan'(str), $ - 1, $$ - $$$ - 2, $$$$ - 3.
def change_price_range(price_range):

    if price_range == '$':

        return 1

    elif price_range == '$$ - $$$':

        return 2

    elif price_range == '$$$$':

        return 3

    elif pd.isna(price_range):

        return 'nan'
data['Price Range'] = data['Price Range'].apply(change_price_range)
data[data['Price Range'] != 'nan']['Price Range'].value_counts()
price_range_nan_index = data[data['Price Range'] == 'nan'].index

for index in price_range_nan_index:

    data.at[index, 'Price Range'] = 2
data = pd.concat([data, pd.get_dummies(data['Price Range'])], axis=1)
data.info(verbose=True, null_counts=True)
data['Number of Reviews'].value_counts().hist(bins=100)
data['Number of Reviews'].isna().sum()
sns.boxplot(data['Number of Reviews'])
# IQR

o = outliers_iqr(data['Number of Reviews'])

o
# Z-score

with warnings.catch_warnings():

    warnings.simplefilter('ignore')

    o = outliers_z_score(data['Number of Reviews'], threshold=10)

len(o)
data.loc[o]['Number of Reviews'].min()
data['Number of Reviews'].fillna(round(data['Number of Reviews'].mean()), inplace=True)
# в ревью нет пропусков, но 6089 строк со значением [[], []]. По сути это пустые строки сохраним их 

data['empty_Reviews'] = (data['Reviews']=='[[], []]').astype('float64')



# анализ тестовой базы выявил два пропуска, несмотря на то, что на тренировочной базе пропусков нет, заполним их '[[], []]' и закинем в empty_Reviews

data['Reviews'] = data['Reviews'].fillna('[[], []]')

data['empty_Reviews'] = (data['Reviews']=='[[], []]').astype('float64')
# вытащим дату из ревью и создадим новые критерии

data['date_of_review'] = data['Reviews'].str.findall('\d+/\d+/\d+')

data['len_date'] = data['date_of_review'].apply(lambda x: len(x))



# проверим длину дат, на случай если там больше или меньше двух

data[data.len_date == 3]
# есть значения с 3 датади, надо разобраться что там

print("кол-во значений Reviews с тремя датами :=" , len(data[data['len_date']==3]))

print("значения Reviews с тремя датами :=")

temp_list = data[data['len_date']==3].Reviews.to_list()

display(data[data['len_date']==3].Reviews.to_list())

display([re.findall('\d+/\d+/\d+', x) for x in temp_list])
data['len_date'].date_of_review = data[data['len_date']==3].date_of_review.apply(lambda x: x.pop(0))

data.len_date.loc[data[data.len_date == 3].index] = 2
print("кол-во значений Reviews с одной датой :=" , len(data[data['len_date']==1]))

display(data[data['len_date']==1].Reviews[:4])
data['one_Review'] = (data['len_date']==1).astype('float64')



# заполним перерыв между отзывами (по отзывам где len = 2) и насколько давно был сделан последний самый свежий отзыв

# создадим для этого функции:

def time_to_now(row):

    if row['date_of_review'] == []:

        return None

    return pd.datetime.now() - pd.to_datetime(row['date_of_review']).max()



def time_between_Reviews(row):

    if row['date_of_review'] == []:

        return None

    return pd.to_datetime(row['date_of_review']).max() - pd.to_datetime(row['date_of_review']).min()



data['day_to_now'] = data.apply(time_to_now, axis = 1).dt.days

data['day_between_reviews'] = data[data['len_date']==2].apply(time_between_Reviews, axis = 1).dt.days
data['day_to_now'].hist(bins=50)
sns.boxplot(data['day_to_now'])
# теперь посмотрим на разницу в датах отзывов в днях 

data['day_between_reviews'].hist(bins=100)
sns.boxplot(data['day_between_reviews'])
o = outliers_iqr(data['day_between_reviews'])

len(o)
o = outliers_z_score(data['day_between_reviews'])

len(o)
plt.rcParams['figure.figsize'] = (10,7)

df_train['Ranking'].hist(bins=100)
df_train['City'].value_counts(ascending=True).plot(kind='barh')
df_train['Ranking'][df_train['City'] =='London'].hist(bins=100)
# посмотрим на топ 10 городов

for x in (df_train['City'].value_counts())[0:10].index:

    df_train['Ranking'][df_train['City'] == x].hist(bins=100)

plt.show()
mean_Ranking_on_City = data.groupby(['City'])['Ranking'].mean()

count_Restorant_in_City = data['City'].value_counts(ascending=False)

data['mean_Ranking_on_City'] = data['City'].apply(lambda x: mean_Ranking_on_City[x])

data['count_Restorant_in_City'] = data['City'].apply(lambda x: count_Restorant_in_City[x])

data['norm_Ranking_on_Rest_in_City'] = (data['Ranking'] - data['mean_Ranking_on_City']) / data['count_Restorant_in_City']
# посмотрим что получилось для топ 10 городов

for x in (data['City'].value_counts())[0:10].index:

    data['norm_Ranking_on_Rest_in_City'][data['City'] == x].hist(bins=100)

plt.show()
max_Ranking_on_City = data.groupby(['City'])['Ranking'].max()

data['max_Ranking_on_City'] = data['City'].apply(lambda x: max_Ranking_on_City[x])

data['norm_Ranking_on_maxRank_in_City'] = (data['Ranking'] - data['mean_Ranking_on_City']) / data['max_Ranking_on_City']
for x in (data['City'].value_counts())[0:10].index:

    data['norm_Ranking_on_maxRank_in_City'][data['City'] == x].hist(bins=100)

plt.show()
df_train['Rating'].value_counts(ascending=True).plot(kind='barh')
df_train['Ranking'][df_train['Rating'] == 5].hist(bins=100)
df_train['Ranking'][df_train['Rating'] < 4].hist(bins=100)
plt.rcParams['figure.figsize'] = (15,10)

sns.heatmap(data.drop(['sample'], axis=1).corr(),)
data.drop(['Restaurant_id', 'City', 'Cuisine Style', 

           'Price Range', 'Reviews', 'URL_TA', 'ID_TA', 

           'date_of_Review', 'len_date', 'mean_Ranking_on_City', 

           'count_Restorant_in_City', 'max_Ranking_on_City', 

           'lower_case_reviews', 'date_of_review'], 

          axis=1, inplace=True, errors='ignore')
# функция для стандартизации

def StandardScaler_column(d_col):

    scaler = StandardScaler()

    scaler.fit(data[[d_col]])

    return scaler.transform(data[[d_col]])
# стандартизируем все столбцы кроме целевой и Sample

for i  in list(data.columns):

    if i not in ['Rating','sample']:

        data[i] = StandardScaler_column(i)

        if len(data[data[i].isna()]) < len(data):

            data[i] = data[i].fillna(0)
# проверяем заполнение

display(data.describe().head(1))
# Теперь выделим тестовую часть

train_data = data.query('sample == 1').drop(['sample'], axis=1)

test_data = data.query('sample == 0').drop(['sample'], axis=1)



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
def classic_round(d_num):

    return int(d_num + (0.5 if d_num > 0 else -0.5))



def my_round(d_pred):

    result = classic_round(d_pred*2)/2

    if result <=5:

        return result

    else:

        return 5



my_vec_round = np.vectorize(my_round)
y_pred = my_vec_round(y_pred)
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
predict_submission
sample_submission['Rating'] = my_vec_round(predict_submission)

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)