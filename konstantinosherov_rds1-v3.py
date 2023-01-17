# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import datetime



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
df_train.head(10)
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
# # Для примера я возьму столбец Number of Reviews

# data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')
# data['Number_of_Reviews_isNAN']
# # Далее заполняем пропуски 0, вы можете попробовать заполнением средним или средним по городу и тд...

# data['Number of Reviews'].fillna(0, inplace=True)
# data.nunique(dropna=False)
data.head(5)
data.sample(5)
data['Price Range'].value_counts()
# # Ваша обработка 'Price Range'

# #Готовим столбец Price Range для создания dummies

# def price_range_rating (price_range_string):

#     if "$$$$" in str(price_range_string):

#         return(int(3))

#     elif "$$ - $$$" in str(price_range_string):

#         return(int(2))

#     elif "$" in str(price_range_string):

#         return(int(1))

#     elif str(0) in str(price_range_string):

#         return(int(0))

# data['Price Range'] = data['Price Range'].apply(price_range_rating)

# #Делаем dummies на основе изменного Price Range

# data = pd.get_dummies(data, columns=['Price Range', ], dummy_na=True)
# # Добавляем новый признак "Столица - Не столица"

# capitals = {'City': [

#     'London', 'Paris', 'Madrid', 'Barcelona', 'Berlin', 'Milan', 'Rome', 'Prague',

#     'Lisbon', 'Vienna', 'Amsterdam', 'Brussels', 'Hamburg', 'Munich', 'Lyon', 'Stockholm',

#     'Budapest', 'Warsaw', 'Dublin', 'Copenhagen', 'Athens', 'Edinburgh', 'Zurich', 'Oporto',

#     'Geneva', 'Krakow', 'Oslo', 'Helsinki', 'Bratislava', 'Luxembourg', 'Ljubljana'],

#     'Country': [

#     'United Kingdom', 'France', 'Spain', 'Spain', 'Germainy', 'Italy', 'Italy', 'Czech Republic',

#     'Portugal', 'Austria', 'Netherlands', 'Belgium', 'Germainy', 'Germainy', 'France', 'Sweden',

#     'Hungary', 'Poland', 'Ireland', 'Denmark', 'Greece', 'Scotland', 'Switzerland', 'Portugal',

#     'Switzerland', 'Poland', 'Norway', 'Finland', 'Slovakia', 'Luxembourg', 'Slovenia'],

#     'Is capital': [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# }

# capitals_list = []

# for i,j in zip(capitals['City'], capitals['Is capital']):

#     if j == 1:

#         capitals_list.append(i)

# def capital_sign(city_string):

#     if city_string in capitals_list:

#         return(1)

#     else:

#         return(0)

# data['Is_capital'] = data['City'].apply(capital_sign)

# #Делаем даммиз на основании нового признака

# data = pd.get_dummies(data, columns=['Is_capital', ], dummy_na=True)
# # для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

# data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
# #Создаем новый признак из разницы между датами отзывов



# # Вычленяем из строки даты и записываем в отдельный столбец

# data['Number of Reviews'].fillna(0, inplace=True)

# #Заполняем наны в столбце Reviews

# data['Reviews'].fillna('[[], []]', inplace=True)

# def data_extract(reviews_str):

#     pattern = re.compile('\d\d\W\d\d\W\d\d\d\d')

#     tmp_data = pattern.findall(reviews_str)

#     return(tmp_data)

# data['Data_Reviews'] = data['Reviews'].apply(data_extract)

# #Резделим даты отзывов на два столбца. Пригодится и для следующего задания

# def data_reviews_separate_1(data_reviews_string):

#     tmp = 0

#     if len(data_reviews_string) == 0:

#         tmp = 0

#     else:

#         tmp = data_reviews_string[0]

#     return(tmp)



# def data_reviews_separate_2(data_reviews_string):

#     tmp = 0

#     if len(data_reviews_string) == 0:

#         tmp = 0

#     elif len(data_reviews_string) == 1:

#         tmp = 0

#     elif len(data_reviews_string) == 2:

#         tmp = data_reviews_string[1]

#     return(tmp)



# data['Data_Reviews_1'] = data['Data_Reviews'].apply(data_reviews_separate_1)

# data['Data_Reviews_2'] = data['Data_Reviews'].apply(data_reviews_separate_2)



# #Преобразовываем столбцы Data_Reviews_ в формат datetime

# data['Data_Reviews_1'] = data['Data_Reviews_1'].apply(lambda x: pd.to_datetime(x) if x != 0 else False)

# data['Data_Reviews_2'] = data['Data_Reviews_2'].apply(lambda x: pd.to_datetime(x) if x != 0 else False) 



# # Вычисляем разницу в днях между отзывами, чистим значения от None и преобразовываем в int

# def data_delta(Data_Reviews_item):

#     if (len(Data_Reviews_item) == 0) or (len(Data_Reviews_item) == 1):

#         tmp = 0

#     else:

#         tmp = pd.to_datetime(

#             Data_Reviews_item[0]) - pd.to_datetime(Data_Reviews_item[1])



#     if tmp != 0:

#         return(str(tmp))





# def data_delta_to_int(data_delta_item):

#     tmp = str(data_delta_item)

#     if '-' in tmp:

#         tmp1 = tmp[1:-15]

#     elif 'None' in tmp:

#         tmp1 = '0'

#     else:

#         tmp1 = tmp[:-14]

#     return(int(tmp1))





# data['Data_Delta'] = data['Data_Reviews'].apply(data_delta)

# data['Data_Delta'] = data['Data_Delta'].apply(data_delta_to_int)





# #Убираем пропуски из столбцов Data_Reviews_х

# def error_date_to_zero(data_string):

#     if data_string == False:

#         return(int(0))

#     else:

#         return(data_string)

# data['Data_Reviews_1'] = data['Data_Reviews_1'].apply(error_date_to_zero)

# data['Data_Reviews_2'] = data['Data_Reviews_2'].apply(error_date_to_zero)
# #Создаем даммиз из столбца с набором кухонь в ресторане



# #Собираем множество из значений столбца data['Cuisine']

# pd.options.mode.chained_assignment = None  # default='warn'

# cuisine_set = set()

# def clearing_string(cuisine_str):

#     tmp = str(cuisine_str).replace('\'',"")

#     tmp = tmp.replace(' ',"")

#     tmp = tmp[1:-1]

#     tmp_lst = tmp.split(',')

#     for i in range(0,len(tmp_lst)):

#         cuisine_set.add(tmp_lst[i])

#     return(cuisine_set)

# data['Cuisine Style'].apply(clearing_string)

# cuisine_set.remove('a')



# #Создаем даммиз из столбца с набором кухонь в ресторане



# #Собираем множество из значений столбца data['Cuisine']

# pd.options.mode.chained_assignment = None  # default='warn'

# cuisine_set = set()

# def clearing_string(cuisine_str):

#     tmp = str(cuisine_str).replace('\'',"")

#     tmp = tmp.replace(' ',"")

#     tmp = tmp[1:-1]

#     tmp_lst = tmp.split(',')

#     for i in range(0,len(tmp_lst)):

#         cuisine_set.add(tmp_lst[i])

#     return(cuisine_set)

# data['Cuisine Style'].apply(clearing_string)

# cuisine_set.remove('a')





# #Для начала преобразуем элементы столбца Cuisine из строк в списки

# def cuisine_convert_to_list (cuisine_str):

#     tmp = str(cuisine_str).replace('\'',"")

#     tmp = tmp.replace(' ',"")

#     tmp = tmp[1:-1]

#     tmp_lst = tmp.split(',')

#     return(tmp_lst)

# #Преобразовываем элементы слобца 'Cuisine' из строк в списки

# data['Cuisine Style'] = data['Cuisine Style'].apply(cuisine_convert_to_list)



# #Создадим словарь из ключей - названий кухнь с нулевыми начениями

# cuisine_list = list(cuisine_set)

# cuisine_df = pd.DataFrame(cuisine_list)

# cuisine_df = cuisine_df[0].sort_values()

# cuisine_df = cuisine_df.reset_index(drop = True)

# #Добавляем новые столбцы по именам кухонь для будущих признаков

# for i in range (0, len(cuisine_list)):

#     data[cuisine_df[i]] = int(0)



# #Проверяем есть ли имя столбца, одноименного с именем кухни, в строке "Cuisine". Если да, в соотв. ячеку столбца с именем кухни ставим 1

# for i in range (0, len(cuisine_df)):

#     for j in range (0,len(data)):

#         if cuisine_df[i] in data['Cuisine Style'][j]:

#             data[cuisine_df[i]][j] = int(1)
# заполняю нулевые значения Number of Reviews нулями, а Rating, Ranking средним арифметическим (посчитал в своем ноутбуке отдельно)

data['Number of Reviews'] = data['Number of Reviews'].fillna(0)

data['Rating'] = data['Rating'].fillna(3.9930375)

data['Ranking'] = data['Ranking'].fillna(3676.028525)
# создаю дополнительный столбец PriceCategory для анализа ценовой категории

PriceCategory = []

for i in data['Price Range']:

    if i == '$':

        PriceCategory.append(1)

    elif i == '$$ - $$$':

        PriceCategory.append(2)

    elif i == '$$$$':

        PriceCategory.append(3)

    else:

        PriceCategory.append(0)

        

data['PriceCategory'] = PriceCategory
# создаю столбец с количеством представленных в ресторане кухонь

cuisine = data['Cuisine Style']

cuisine2 = cuisine.fillna('No data')

Cuisune_Quantity = []

n = -1

for i in cuisine2:

    n += 1

    if i == 'No data':

        Cuisune_Quantity.append(0)

    else:

        cuisine2[n] = cuisine2[n][1:-1]

        cuisine2[n] = cuisine2[n].split(', ')

        Cuisune_Quantity.append(int(len(cuisine2[n])))



data['Cuisune_Quantity'] = Cuisune_Quantity
# создаю столбец с количеством отзывов о ресторане

dr=[]

Num_Reviews = []

n = 0

for i in data['Reviews']:

    dr.append(re.findall('\d\d/\d\d/\d\d\d\d',i))

    Num_Reviews.append(len(dr[n]))

    n += 1

data['Num_Reviews'] = Num_Reviews
# # убираю нечисловые столбцы

# data = data.drop(['City','Cuisine Style','Price Range','Reviews','URL_TA','ID_TA'], axis = 1)
data
data.info()
plt.rcParams['figure.figsize'] = (10,7)

df_train['Ranking'].hist(bins=100)
df_train['City'].value_counts(ascending=True).plot(kind='barh')
df_train['Ranking'][df_train['City'] =='London'].hist(bins=100)
# посмотрим на топ 10 городов

for x in (df_train['City'].value_counts())[0:10].index:

    df_train['Ranking'][df_train['City'] == x].hist(bins=100)

plt.show()
df_train['Rating'].value_counts(ascending=True).plot(kind='barh')
df_train['Ranking'][df_train['Rating'] == 5].hist(bins=100)
df_train['Ranking'][df_train['Rating'] < 4].hist(bins=100)
plt.rcParams['figure.figsize'] = (15,10)

sns.heatmap(data.drop(['sample'], axis=1).corr(),)
# на всякий случай, заново подгружаем данные

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем

data.info()
def preproc_data(df_input):

    '''includes several functions to pre-process the predictor data.'''

    

    df_output = df_input.copy()

    

    # ################### 1. Предобработка ############################################################## 

    # убираем не нужные для модели признаки

    df_output.drop(['Restaurant_id','ID_TA',], axis = 1, inplace=True)

    

    

    # ################### 2. NAN ############################################################## 

    # Далее заполняем пропуски, вы можете попробовать заполнением средним или средним по городу и тд...

    df_output['Number of Reviews'].fillna(0, inplace=True)

    # тут ваш код по обработке NAN

    # ....

    

    

    # ################### 3. Encoding ############################################################## 

    # для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

    #df_output = pd.get_dummies(df_output, columns=[ 'City',], dummy_na=True)

    

    # ################### 4. Feature Engineering ####################################################

    # тут ваш код не генерацию новых фитчей

    #Готовим столбец Price Range для создания dummies

    def price_range_rating (price_range_string):

        if "$$$$" in str(price_range_string):

            return(int(3))

        elif "$$ - $$$" in str(price_range_string):

            return(int(2))

        elif "$" in str(price_range_string):

            return(int(1))

        elif str(0) in str(price_range_string):

            return(int(0))

    df_output['Price Range'] = df_output['Price Range'].apply(price_range_rating)

    #Делаем dummies на основе изменного Price Range

    df_output = pd.get_dummies(df_output, columns=['Price Range', ], dummy_na=True)

    

    # Добавляем новый признак "Столица - Не столица"

    capitals = {'City': [

    'London', 'Paris', 'Madrid', 'Barcelona', 'Berlin', 'Milan', 'Rome', 'Prague',

    'Lisbon', 'Vienna', 'Amsterdam', 'Brussels', 'Hamburg', 'Munich', 'Lyon', 'Stockholm',

    'Budapest', 'Warsaw', 'Dublin', 'Copenhagen', 'Athens', 'Edinburgh', 'Zurich', 'Oporto',

    'Geneva', 'Krakow', 'Oslo', 'Helsinki', 'Bratislava', 'Luxembourg', 'Ljubljana'],

    'Country': [

    'United Kingdom', 'France', 'Spain', 'Spain', 'Germainy', 'Italy', 'Italy', 'Czech Republic',

    'Portugal', 'Austria', 'Netherlands', 'Belgium', 'Germainy', 'Germainy', 'France', 'Sweden',

    'Hungary', 'Poland', 'Ireland', 'Denmark', 'Greece', 'Scotland', 'Switzerland', 'Portugal',

    'Switzerland', 'Poland', 'Norway', 'Finland', 'Slovakia', 'Luxembourg', 'Slovenia'],

    'Is capital': [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    }

    capitals_list = []

    for i,j in zip(capitals['City'], capitals['Is capital']):

        if j == 1:

            capitals_list.append(i)

    def capital_sign(city_string):

        if city_string in capitals_list:

            return(1)

        else:

            return(0)

    df_output['Is_capital'] = df_output['City'].apply(capital_sign)

    #Делаем даммиз на основании нового признака

    df_output = pd.get_dummies(df_output, columns=['Is_capital', ], dummy_na=True)

    



    # для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

    df_output = pd.get_dummies(df_output, columns=[ 'City',], dummy_na=True)

    

    #Создаем новый признак из разницы между датами отзывов



    # Вычленяем из строки даты и записываем в отдельный столбец

    df_output['Number of Reviews'].fillna(0, inplace=True)

    #Заполняем наны в столбце Reviews

    df_output['Reviews'].fillna('[[], []]', inplace=True)

    def data_extract(reviews_str):

        pattern = re.compile('\d\d\W\d\d\W\d\d\d\d')

        tmp_data = pattern.findall(reviews_str)

        return(tmp_data)

    df_output['Data_Reviews'] = df_output['Reviews'].apply(data_extract)

    #Резделим даты отзывов на два столбца. Пригодится и для следующего задания

    def data_reviews_separate_1(data_reviews_string):

        tmp = 0

        if len(data_reviews_string) == 0:

            tmp = 0

        else:

            tmp = data_reviews_string[0]

        return(tmp)



    def data_reviews_separate_2(data_reviews_string):

        tmp = 0

        if len(data_reviews_string) == 0:

            tmp = 0

        elif len(data_reviews_string) == 1:

            tmp = 0

        elif len(data_reviews_string) == 2:

            tmp = data_reviews_string[1]

        return(tmp)



    df_output['Data_Reviews_1'] = df_output['Data_Reviews'].apply(data_reviews_separate_1)

    df_output['Data_Reviews_2'] = df_output['Data_Reviews'].apply(data_reviews_separate_2)



    #Преобразовываем столбцы Data_Reviews_ в формат datetime

    df_output['Data_Reviews_1'] = df_output['Data_Reviews_1'].apply(lambda x: pd.to_datetime(x) if x != 0 else False)

    df_output['Data_Reviews_2'] = df_output['Data_Reviews_2'].apply(lambda x: pd.to_datetime(x) if x != 0 else False) 



    # Вычисляем разницу в днях между отзывами, чистим значения от None и преобразовываем в int

    def data_delta(Data_Reviews_item):

        if (len(Data_Reviews_item) == 0) or (len(Data_Reviews_item) == 1):

            tmp = 0

        else:

            tmp = pd.to_datetime(

                Data_Reviews_item[0]) - pd.to_datetime(Data_Reviews_item[1])



        if tmp != 0:

            return(str(tmp))





    def data_delta_to_int(data_delta_item):

        tmp = str(data_delta_item)

        if '-' in tmp:

            tmp1 = tmp[1:-15]

        elif 'None' in tmp:

            tmp1 = '0'

        else:

            tmp1 = tmp[:-14]

        return(int(tmp1))





    df_output['Data_Delta'] = df_output['Data_Reviews'].apply(data_delta)

    df_output['Data_Delta'] = df_output['Data_Delta'].apply(data_delta_to_int)





    #Убираем пропуски из столбцов Data_Reviews_х

    def error_date_to_zero(data_string):

        if data_string == False:

            return(int(0))

        else:

            return(data_string)

    df_output['Data_Reviews_1'] = df_output['Data_Reviews_1'].apply(error_date_to_zero)

    df_output['Data_Reviews_2'] = df_output['Data_Reviews_2'].apply(error_date_to_zero)

    

    #Создаем даммиз из столбца с набором кухонь в ресторане



    #Собираем множество из значений столбца data['Cuisine']

    pd.options.mode.chained_assignment = None  # default='warn'

    cuisine_set = set()

    def clearing_string(cuisine_str):

        tmp = str(cuisine_str).replace('\'',"")

        tmp = tmp.replace(' ',"")

        tmp = tmp[1:-1]

        tmp_lst = tmp.split(',')

        for i in range(0,len(tmp_lst)):

            cuisine_set.add(tmp_lst[i])

        return(cuisine_set)

    df_output['Cuisine Style'].apply(clearing_string)

    cuisine_set.remove('a')



    #Создаем даммиз из столбца с набором кухонь в ресторане



    #Собираем множество из значений столбца data['Cuisine']

    pd.options.mode.chained_assignment = None  # default='warn'

    cuisine_set = set()

    def clearing_string(cuisine_str):

        tmp = str(cuisine_str).replace('\'',"")

        tmp = tmp.replace(' ',"")

        tmp = tmp[1:-1]

        tmp_lst = tmp.split(',')

        for i in range(0,len(tmp_lst)):

            cuisine_set.add(tmp_lst[i])

        return(cuisine_set)

    df_output['Cuisine Style'].apply(clearing_string)

    cuisine_set.remove('a')





    #Для начала преобразуем элементы столбца Cuisine из строк в списки

    def cuisine_convert_to_list (cuisine_str):

        tmp = str(cuisine_str).replace('\'',"")

        tmp = tmp.replace(' ',"")

        tmp = tmp[1:-1]

        tmp_lst = tmp.split(',')

        return(tmp_lst)

    #Преобразовываем элементы слобца 'Cuisine' из строк в списки

    df_output['Cuisine Style'] = df_output['Cuisine Style'].apply(cuisine_convert_to_list)



    #Создадим словарь из ключей - названий кухнь с нулевыми начениями

    cuisine_list = list(cuisine_set)

    cuisine_df = pd.DataFrame(cuisine_list)

    cuisine_df = cuisine_df[0].sort_values()

    cuisine_df = cuisine_df.reset_index(drop = True)

    #Добавляем новые столбцы по именам кухонь для будущих признаков

    for i in range (0, len(cuisine_list)):

        df_output[cuisine_df[i]] = int(0)



    #Проверяем есть ли имя столбца, одноименного с именем кухни, в строке "Cuisine". Если да, в соотв. ячеку столбца с именем кухни ставим 1

    for i in range (0, len(cuisine_df)):

        for j in range (0,len(df_output)):

            if cuisine_df[i] in df_output['Cuisine Style'][j]:

                df_output[cuisine_df[i]][j] = int(1)



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
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)