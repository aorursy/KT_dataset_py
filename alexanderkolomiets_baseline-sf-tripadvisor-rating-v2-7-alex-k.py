# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



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
# Для примера я возьму столбец Number of Reviews

data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')
data['Number_of_Reviews_isNAN']
# Далее заполняем пропуски 0, вы можете попробовать заполнением средним или средним по городу и тд...

data['Number of Reviews'].fillna(0, inplace=True)
data.nunique(dropna=False)
# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
data.head(5)
data.sample(4)
data['Price Range'].value_counts()
# Ваша обработка 'Price Range'

price_range_dict = {'$' : 1,

                    '$$ - $$$' : 2,

                    '$$$$' : 3}

data['Price Range'] = data['Price Range'].replace(to_replace = price_range_dict)
data['Price Range'].value_counts()
# тут ваш код на обработку других признаков

#Добавим отдельный столбцы на дату отзыва

data['last_rev_date_gap'] = data.apply(lambda _: '', axis=1)

#Избавимся от NaN, чтобы функция выделения даты сработала

data['Reviews'].fillna("No_data", inplace=True)

import re

#выделяем последнюю дату отзыва

def last_rev_date_gap(cell):

    today = pd.to_datetime(np.datetime64('today'))

    pattern_txt = re.compile('\d+\/\d+\/\d+')

    dates_str = pattern_txt.findall(cell)

    dates = set()

    if len(dates_str) > 0:

        for i in dates_str:

            date = pd.to_datetime(i)

            dates.add(date)

        return (today - max(dates)).days

    

data['last_rev_date_gap'] = data['Reviews'].apply(last_rev_date_gap)

data['last_rev_date_gap'].fillna(data['last_rev_date_gap'].mean())
data['last_rev_date_gap']
# Создадим признак 'Cuisine_Style_isNAN'

data['Cuisine_Style_isNAN'] = pd.isna(data['Cuisine Style']).astype('uint8') #

# Далее пропуски в 'Cuisine Style' заполним "empty"

data['Cuisine Style'].fillna("No_data", inplace=True)
# почистим строку в 'Cuisine Style' от мусора и разобьем на кухни, получив список :

def string_clean_n_split (string):

    string = string.replace("'",'')

    string = string.replace("[",'')

    string = string.replace("]",'')

    string_list = string.split(", ")

    return string_list

data['Cuisine Style'] = data['Cuisine Style'].apply(string_clean_n_split)
#Проверим, что получилось:

data['Cuisine Style'].sample(5)
#Соберем множество из всех наименований кухонь:

cuisine_set = set()

for i in data['Cuisine Style']:

    for j in i:

        cuisine_set.add(j)
#Создадим столбцы для каждого наименования кухни:

for i in cuisine_set:

    data[i] = 0

    

#И посмотрим, что получилось:

data.sample(2)
#Напишем функцию, которая сверяет наименование кухни в 'Cuisine Style' с полным списком

#Если совпадение есть, ставим 1 в соответствующий столбец:

def dummy_cuisine_set(cell_data):

    if item in cell_data:

        return 1

    else:

        return 0

    

for item in list(cuisine_set):

    data[item] = data['Cuisine Style'].apply(dummy_cuisine_set)
#Выборочно проверим для строки z -> при запуске кода каждому наименованию кухни должна соответствовать 1:

z = np.random.randint(0, high = data.shape[0])

print (data['Cuisine Style'][z][0], data[data['Cuisine Style'][z][0]][z])
#Заполним 'Number of Reviews' - 0 , где нет 'Reviews'; мин, где есть хоть что-то:

data[data['Number of Reviews']>=0]['Number of Reviews'] = data['Reviews'].apply(lambda x: 2 )
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

    #Хотелось бы удалить дубликаты (предварительно нашел, что Reviews по многим позициям аналогичны)

    #Но не смог увязать итоговый код так, чтобы модель дошла до решения без ошибок

    #При удалении дубликатов размерность снижается иприсваивание в конце перестае работать 

    #df_output = df_output.drop_duplicates('Reviews')

    

    

    # ################### 2. NAN ############################################################## 

    # Далее заполняем пропуски, вы можете попробовать заполнением средним или средним по городу и тд...

    df_output['Number of Reviews'].fillna((df_output['Number of Reviews'].min()), inplace=True)

    # тут ваш код по обработке NAN

    # пропуски в 'Cuisine Style' заполним "No_data"

    df_output['Cuisine Style'].fillna("No_data", inplace=True)

        

    

    # ################### 3. Encoding ############################################################## 

    # для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

    df_output = pd.get_dummies(df_output, columns=[ 'City',], dummy_na=True)

    # тут ваш код не Encoding фитчей

    # обработка 'Price Range'

    price_range_dict = {'$' : 1,

                        '$$ - $$$' : 2,

                        '$$$$' : 3}

    df_output['Price Range'] = df_output['Price Range'].replace(to_replace = price_range_dict)

    # пропуски в 'Price range' заполним минимумум 'Price range':

    df_output['Price Range'].fillna(1, inplace = True)

      

    

    

    # ################### 4. Feature Engineering ####################################################

    # тут ваш код не генерацию новых фитчей

    # почистим строку в 'Cuisine Style' от мусора и разобьем на кухни, получив список :

    df_output['Cuisine Style'] = df_output['Cuisine Style'].apply(string_clean_n_split)

    #Соберем множество из всех наименований кухонь:

    cuisine_set = set()

    for i in df_output['Cuisine Style']:

        for j in i:

            cuisine_set.add(j)

    #Создадим столбцы для каждого наименования кухни:

    for i in cuisine_set:

        df_output[i] = 0

    #Напишем функцию, которая сверяет наименование кухни в 'Cuisine Style' с полным списком

    #Если совпадение есть, ставим 1 в соответствующий столбец, если нет - оставляем 0:

    def dummy_cuisine_set(cell_data):

        if item in cell_data:

            return 1

        else:

            return 0

    

    for item in list(cuisine_set):

        df_output[item] = df_output['Cuisine Style'].apply(dummy_cuisine_set)

        

    #Добавим отдельный столбцы на дату отзыва

    df_output['last_rev_date_gap'] = df_output.apply(lambda _: '', axis=1)

    #Избавимся от NaN, чтобы функция выделения даты сработала

    df_output['Reviews'].fillna("No_data", inplace=True)

    import re

    #выделяем последнюю дату отзыва

    def last_rev_date_gap(cell):

        today = pd.to_datetime(np.datetime64('today'))

        pattern_txt = re.compile('\d+\/\d+\/\d+')

        dates_str = pattern_txt.findall(cell)

        dates = set()

        if len(dates_str) > 0:

            for i in dates_str:

                date = pd.to_datetime(i)

                dates.add(date)

            return (today - max(dates)).days

    

    df_output['last_rev_date_gap'] = df_output['Reviews'].apply(last_rev_date_gap)

    df_output['last_rev_date_gap'].fillna(df_output['last_rev_date_gap'].mean(), inplace = True)

    

    

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
len(list(predict_submission))
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)