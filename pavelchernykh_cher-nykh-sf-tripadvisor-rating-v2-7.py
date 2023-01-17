# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import datetime

from datetime import datetime, timedelta



import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



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

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
df_train.info()
df_train.head(5)
df_test.info()
df_test.head(5)
sample_submission
sample_submission.info()
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 

# в тесте у нас нет значения Rating, мы его должны предсказать, 

# поэтому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
data.info()
data.sample(5)
data.Reviews[1]
# Обработка для Cuisine Style

def str_to_list(string):

    if pd.isnull(string):

        return []

    else:

        string = string.replace('"', '')

        string = string.replace('[', '')

        string = string.replace(']', '')

        string = string.replace("'", '')

        return string.split(', ')





# Обработка для Reviews

def reviews_to_list(string):

    if string == '[[], []]':

        return []

    else:

        string = string.replace(']]', '')

        string = string.replace("'", '')

        string = string.split('], [')[1]

        string = string.split(', ')

        return string
# Для примера я возьму столбец Number of Reviews

data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')
data['Number_of_Reviews_isNAN']
data['Number_of_Reviews_isNAN'].value_counts()
# Далее заполняем пропуски 0, вы можете попробовать заполнением средним или средним по городу и тд...

data['Number of Reviews'].fillna(0, inplace=True)
data['Number_of_Reviews_isNAN'].value_counts()
data.nunique(dropna=False)
data.head(5)
data.sample(5)
# т.к. это база Trip_Adviser, то начнем с их уникального идентификатора ресторана в базе данных



data['ID_TA'].value_counts()
# посмотрим внутри на примере одного дубля



data_ID_TA = data[data['ID_TA'] == 'd2669414']

data_ID_TA



# все одинаково кроме 'Restaurant-id'
# чтобы удостовериться в повторяемости дублей посмотрим следующий признак 'URL_TA'



data['URL_TA'].value_counts()
# дубли есть и их то же количество

# посмотрим, что внутри.

data_ID_TA = data[data['URL_TA'] == '/Restaurant_Review-g187514-d2477531-Reviews-Haya_19-Madrid.html']

data_ID_TA

# полное совпадение с предыдущей историей
# с чистой совестью могу удалить дубли (проверка показала, что также исчезли дубли и в 'URL_TA')



data.drop_duplicates(subset=['ID_TA'], inplace=True)
data['URL_TA'].value_counts()
data.info()
# посмотрим на дубли в 'Restaurant_id'



data['Restaurant_id'].value_counts()
# посмотрим на содержимое



data_ID = data[data['Restaurant_id'] == 'id_871']

data_ID.sample(7)
data['City'].value_counts(dropna = False)
# Добавил в датасет население городов City_pop - внешний источник данных: Википедия. данные на 2018 год



dict_tmp = {

    'London': 8787892,

    'Paris': 2140526,

    'Madrid': 3223334,

    'Barcelona': 1620343,

    'Berlin': 3601131,

    'Milan': 1366180,

    'Rome': 2872800,

    'Prague': 1280508,

    'Lisbon': 505526,

    'Vienna': 1840573,

    'Amsterdam': 859732,

    'Brussels': 144784,

    'Hamburg': 1830584,

    'Munich': 1456039,

    'Lyon': 515695,

    'Stockholm': 961609,

    'Budapest': 1749734,

    'Warsaw': 1758143,

    'Dublin': 553165,

    'Copenhagen': 615993,

    'Athens': 655780,

    'Edinburgh': 476100,

    'Zurich': 402275,

    'Oporto': 221800,

    'Geneva': 196150,

    'Krakow': 766739,

    'Oslo': 673469,

    'Helsinki': 643272,

    'Bratislava': 413192,

    'Luxembourg': 576249,

    'Ljubljana': 277554

}



data['City_pop'] = data['City'].apply(lambda x: dict_tmp[x])
# добавил бинарный признак столица или нет.



dict_tmp = {

    'London': 1,

    'Paris': 1,

    'Madrid': 1,

    'Barcelona': 0,

    'Berlin': 1,

    'Milan': 0,

    'Rome': 1,

    'Prague': 1,

    'Lisbon': 1,

    'Vienna': 1,

    'Amsterdam': 1,

    'Brussels': 1,

    'Hamburg': 0,

    'Munich': 0,

    'Lyon': 0,

    'Stockholm': 1,

    'Budapest': 1,

    'Warsaw': 1,

    'Dublin': 1,

    'Copenhagen': 1,

    'Athens': 1,

    'Edinburgh': 1,

    'Zurich': 1,

    'Oporto': 0,

    'Geneva': 1,

    'Krakow': 0,

    'Oslo': 1,

    'Helsinki': 1,

    'Bratislava': 1,

    'Luxembourg': 1,

    'Ljubljana': 1

}



data['Capital'] = data['City'].apply(lambda x: dict_tmp[x])
# Добавим столбец Country - название страны.



dict_tmp = {

    'London': 'UK',

    'Paris': 'FR',

    'Madrid': 'ESP',

    'Barcelona': 'ESP',

    'Berlin': 'GER',

    'Milan': 'IT',

    'Rome': 'IT',

    'Prague': 'CZ',

    'Lisbon': 'PO',

    'Vienna': 'AUS',

    'Amsterdam': 'NDL',

    'Brussels': 'BLG',

    'Hamburg': 'GER',

    'Munich': 'GER',

    'Lyon': 'FR',

    'Stockholm': 'SWE',

    'Budapest': 'HUN',

    'Warsaw': 'PL',

    'Dublin': 'IRL',

    'Copenhagen': 'DM',

    'Athens': 'GRC',

    'Edinburgh': 'SCT',

    'Zurich': 'SWZ',

    'Oporto': 'PO',

    'Geneva': 'SWZ',

    'Krakow': 'PL',

    'Oslo': 'NOR',

    'Helsinki': 'FIN',

    'Bratislava': 'SVK',

    'Luxembourg': 'LUX',

    'Ljubljana': 'SVN'

}



data['Country'] = data['City'].apply(lambda x: dict_tmp[x])
data['Country'].nunique()
# посчитаем сколько ресторанов в каждой стране. Сделаем столбец с таким количеством



data_tmp = data['Country'].value_counts()

data['Rest_Count_Country'] = data['Country'].apply(lambda x: data_tmp[x])
# Создадим признак R_counts - кол-во ресторанов в каждом городе.



data_tmp = data['City'].value_counts()

data['R_counts'] = data['City'].apply(lambda x: data_tmp[x])
# Добавим новый признак R_City_concentration - коэффициент количества ресторанов в городах страны.

data['Rest_City_concentration'] = data['R_counts'] / data['Rest_Count_Country']
# с сайта World Tourism Organization (UNWTO) скачал отчет https://www.e-unwto.org/doi/book/10.18111/9789284419876

# взял значение, сколько тратит в среднем один турист при посещении той или иной страны в долл.США.



dict_tmp = {

    'UK': 1337.63,

    'FR': 659.38,

    'ESP': 803.33,

    'GER': 1052.25,

    'IT': 768.46,

    'CZ': 492.58,

    'PO': 771.21,

    'AUS': 684.90,

    'NDL': 887.92,

    'BLG': 1552.20,

    'SWE': 1944.50,

    'HUN': 371.26,

    'PL': 628.59,

    'IRL': 513.47,

    'DM': 653.65,

    'GRC': 589.50,

    'SCT': 1337.63,

    'SWZ': 1562.87,

    'NOR': 873.15,

    'FIN': 979.20,

    'SVK': 412.37,

    'LUX': 3867.17,

    'SVN': 799.47

}



data['spend_visitor'] = data['Country'].apply(lambda x: dict_tmp[x])
# удалю вспомогательный столбец

data.drop(['Rest_Count_Country'], axis=1, inplace=True)
# распределение количества отзывов по городам.



data_tmp = data.groupby(['City'])['Number of Reviews'].sum().sort_values(ascending=False)

data_tmp
# количество отзывов в каждом городе

data['Rev_num_city'] = data['City'].apply(lambda x: data_tmp[x])
# Добавим параметр Rel_num_rev - ранг ресторана по количеству отзывов о ресторанах в городе.

data['Rel_num_rev'] = data['Ranking'] / data['Rev_num_city']
# удаляем вспомогательный признак

data.drop(['Rev_num_city'], axis=1, inplace=True)
data['Price Range'].value_counts()
# Ваша обработка 'Price Range'

data['Price Range'] = data['Price Range'].replace(['$$ - $$$', '$', '$$$$'], [2, 1, 3])
median_p = int(data['Price Range'].median())

data['Price Range'] = data['Price Range'].fillna(median_p)
# Преобразуем кухони в списки



data['Cuisine_Style_list'] = data['Cuisine Style'].apply(str_to_list)

#df.drop(['Cuisine Style'], axis=1, inplace=True)
# Посмотрим какие кухни бывают



cuisine_set = set()

for cuis_list in data['Cuisine_Style_list']:

    for cuis in cuis_list:

        cuisine_set.add(cuis)



print('Всего кухонь: ', len(cuisine_set))
# Считаем сколько раз кухня встречается в ресторанах

cuisine_count = dict.fromkeys(cuisine_set, 0)

for cuis in cuisine_set:

    for cuis_list in data['Cuisine_Style_list']:

        if cuis in cuis_list:

            cuisine_count[cuis] += 1



cuisine_count = pd.Series(cuisine_count)

cuisine_count.sort_values(ascending=False)
# самая популярная кухня Vegetarian Friendly.



cuis_nan = 'Vegetarian Friendly'

data['Cuisine Style'].fillna(cuis_nan, inplace=True)

data['Cuisine_Style_list'] = data['Cuisine Style'].apply(str_to_list)



# далее в разделе dummy переменные буду использовать эти данные для формирования новых признаков.
# Заполним пропуски, если есть

data['Reviews'] = data['Reviews'].apply(

    lambda x: '[[], []]' if pd.isnull(x) else x)
data['Reviews_date_temp'] = data['Reviews'].apply(reviews_to_list)


data['Reviews_date_first'] = data['Reviews_date_temp'].apply(

    lambda x: None if len(x) == 0

    else pd.to_datetime(x[0], format='%m/%d/%Y') if len(x) == 1

    else pd.to_datetime(x[0], format='%m/%d/%Y') if pd.to_datetime(x[0], format='%m/%d/%Y') < pd.to_datetime(x[1], format='%m/%d/%Y')

    else pd.to_datetime(x[1], format='%m/%d/%Y'))

data['Reviews_date_last'] = data['Reviews_date_temp'].apply(

    lambda x: None if len(x) == 0

    else pd.to_datetime(x[0], format='%m/%d/%Y') if len(x) == 1

    else pd.to_datetime(x[0], format='%m/%d/%Y') if pd.to_datetime(x[0], format='%m/%d/%Y') > pd.to_datetime(x[1], format='%m/%d/%Y')

    else pd.to_datetime(x[1], format='%m/%d/%Y'))
min_date = data['Reviews_date_first'].mean()

max_date = data['Reviews_date_last'].mean()
# Заполним пропуски средней датой



data['Reviews_date_first'] = data['Reviews_date_first'].apply(

    lambda x: min_date if pd.isnull(x) else min_date if x == None else x)

data['Reviews_date_last'] = data['Reviews_date_last'].apply(

    lambda x: max_date if pd.isnull(x) else min_date if x == None else x)
# Создадим параметр date_delta - сколько дней прошло между первым и последним отзывом.



data['delta_days'] = (data['Reviews_date_last'] - data['Reviews_date_first']).dt.days
# Создадим параметр Rev_year - год последнего отзыва. Возможно, с годами уровень лояльности пользователей меняется.



data['Rev_last_year'] = data['Reviews_date_last'].dt.year
data.sample(5)
# Удалим лишние столбцы, на основе которых работали с датами.



data.drop(['Reviews_date_temp', 'Reviews_date_first','Reviews_date_last'], axis=1, inplace=True)
plt.rcParams['figure.figsize'] = (10,7)

df_train['Ranking'].hist(bins=100)
df_train['City'].value_counts(ascending=True).plot(kind='barh')
df_train['Ranking'][df_train['City'] =='London'].hist(bins=100)
# посмотрим на топ 10 городов

for x in (df_train['City'].value_counts())[0:10].index:

    df_train['Ranking'][df_train['City'] == x].hist(bins=100)

plt.show()
# параметр Relative_rank - относительного ранга = ранг / кол-во ресторанов в городе.

data['Ranking_rel'] = round((data['Ranking'] / data['R_counts']), 2)
# удаляем вспомогательный столбец



data.drop(['R_counts'], axis=1, inplace=True)
df_train['Rating'].value_counts(ascending=True).plot(kind='barh')
df_train['Ranking'][df_train['Rating'] == 5].hist(bins=100)
df_train['Ranking'][df_train['Rating'] < 4].hist(bins=100)
data_t = data[data['sample'] == 1]
data_t.info()
data_t['Rating'].hist()

data_t['Rating'].describe()
median_r = data_t['Rating'].median()

IQR_r = data_t['Rating'].quantile(0.75) - data_t['Rating'].quantile(0.25)

perc25 = data_t['Rating'].quantile(0.25)

perc75 = data_t['Rating'].quantile(0.75)

print('25-й перцентиль: {},'.format(perc25), '75-й перцентиль: {},'.format(perc75), "IQR: {}, ".format(IQR_r),

      "Границы выбросов: [{f}, {l}].".format(f=perc25 - 1.5*IQR_r, l=perc75 + 1.5*IQR_r))

data_t['Rating'].loc[data_t['Rating'].between(perc25 - 1.5*IQR_r, perc75 + 1.5*IQR_r)].hist(bins=10, range=(0, 10),

                                                                                        label='IQR')

plt.legend()
data = data.loc[data['Rating'] != 1]

data.info()
plt.rcParams['figure.figsize'] = (15,10)

sns.heatmap(data.drop(['sample'], axis=1).corr(),)
# создаем dummy по отзывам



data['R_terrible'] = data['Reviews'].apply(

    lambda x: 1 if 'terrible' in x.lower() else 0)

data['R_horrible'] = data['Reviews'].apply(

    lambda x: 1 if 'horrible' in x.lower() else 0)

data['R_not_good'] = data['Reviews'].apply(

    lambda x: 1 if 'not good' in x.lower() else 0)

data['R_disappointing'] = data['Reviews'].apply(

    lambda x: 1 if 'disappointing' in x.lower() else 0)

data['R_worst'] = data['Reviews'].apply(

    lambda x: 1 if 'worst' in x.lower() else 0)

data['R_bad'] = data['Reviews'].apply(

    lambda x: 1 if 'bad' in x.lower() else 0)

data['R_better'] = data['Reviews'].apply(

    lambda x: 1 if 'better' in x.lower() else 0)

data['R_excellent'] = data['Reviews'].apply(

    lambda x: 1 if 'excellent' in x.lower() else 0)

data['R_best'] = data['Reviews'].apply(

    lambda x: 1 if 'best' in x.lower() else 0)

data['R_amazing'] = data['Reviews'].apply(

    lambda x: 1 if 'amazing' in x.lower() else 0)

data['R_great'] = data['Reviews'].apply(

    lambda x: 1 if 'great' in x.lower() else 0)

data['R_wonderful'] = data['Reviews'].apply(

    lambda x: 1 if 'wonderful' in x.lower() else 0)

data['R_super'] = data['Reviews'].apply(

    lambda x: 1 if 'super' in x.lower() else 0)

data['R_good_food'] = data['Reviews'].apply(

    lambda x: 1 if 'good_food' in x.lower() else 0)

data['R_delicious'] = data['Reviews'].apply(

    lambda x: 1 if 'delicious' in x.lower() else 0)

data['R_but'] = data['Reviews'].apply(

    lambda x: 1 if 'but' in x.lower() else 0)

data['R_not'] = data['Reviews'].apply(

    lambda x: 1 if 'not' in x.lower() else 0)
#Переформатируем информацию о кухнях в dummy-переменные.



for cuis in cuisine_set:

    data[cuis] = 0

    data[cuis] = data['Cuisine Style'].apply(lambda x: 1 if cuis in x else 0)
# теперь уберем вспомогательный столбец

data.drop(['Cuisine_Style_list'], axis=1, inplace=True)
# из кухонь возможно можно было вытащить еще, но надо больше времени. 

# там есть интересная взаимосвязь со странами и городами. 

# пробовал количество типов кухонь на ресторан, но MAE не очень сильно изменился. 

# Удаляю столбец



data.drop(['Cuisine Style'], axis=1, inplace=True)
# для One-Hot Encoding в pandas есть готовая функция - get_dummies. 

# Особенно радует параметр dummy_na

data = pd.get_dummies(data, columns=['City'], dummy_na=True)
#Предположим, что индентификатор присваивается последовательно в момент регистрации ресторана на сайте TripAdvisor, 

#может влиять на рейтинг.



data['ID_TA_num'] = data['ID_TA'].apply(lambda x: int(x[1:]))
# у Restaurant_id значения частично совпадают с Rating можно создать кластеры если убрать индекс



data['R_id_num'] = data['Restaurant_id'].apply(lambda x: round(int(x[3:])/10))
# Удалим теперь уже не нужные столбцы, на основе которых работали



data.drop(['Reviews', 'URL_TA', 'Country'], axis=1, inplace=True)
data.sample(3)
def preproc_data(df_input):

    '''includes several functions to pre-process the predictor data.'''

    

    df_output = df_input.copy()

    

    # ################### 1. Предобработка ############################################################## 

    # убираем не нужные для модели признаки

    df_output.drop(['Restaurant_id','ID_TA',], axis = 1, inplace=True)

    

    

    # ################### 2. NAN ############################################################## 

    # Далее заполняем пропуски, вы можете попробовать заполнением средним или средним по городу и тд...

    #df_output['Number of Reviews'].fillna(0, inplace=True)

    # тут ваш код по обработке NAN

    # ....

    

    

    # ################### 3. Encoding ############################################################## 

    # для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

    # df_output = pd.get_dummies(df_output, columns=[ 'City',], dummy_na=True)

    # тут ваш код не Encoding фитчей

    # ....

    

    

    # ################### 4. Feature Engineering ####################################################

    # тут ваш код не генерацию новых фитчей

    # ....

    

    

    # ################### 5. Clean #################################################### 

    # убираем признаки которые еще не успели обработать, 

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
y_pred = np.round(y_pred*2)/2
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# в RandomForestRegressor есть возможность вывести самые важные признаки для модели

plt.rcParams['figure.figsize'] = (10,10)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(20).plot(kind='barh')
#test_data = data.query('sample == 0').drop(data['sample'], axis=1)

test_data.sample(10)
test_data = test_data.drop(['Rating'], axis=1)
sample_submission
predict_submission = model.predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)