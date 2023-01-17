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
data.info()
data.nunique(dropna=False)
# Для примера я возьму столбец Number of Reviews

data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')
data['Number_of_Reviews_isNAN']
# Далее заполняем пропуски 0, вы можете попробовать заполнением средним или средним по городу и тд...

data['Number of Reviews'].fillna(0, inplace=True)
data['Reviews_isNAN'] = pd.isna(data['Reviews']).astype('uint8')
data['Reviews'].fillna(0, inplace=True)
data['Cuisine Style'].fillna('[unknown]', inplace=True)

data.sample(5)
data['Price Range'].fillna(0, inplace=True)

data.sample(5)
data.nunique(dropna=False)
data['City'].unique()
capital = {'Paris':1, 'Stockholm':1, 'London':1, 'Berlin':1, 'Munich':0, 'Oporto':0,

       'Milan':0, 'Bratislava':1, 'Vienna':2, 'Rome':3, 'Barcelona':0, 'Madrid':1,

       'Dublin':1, 'Brussels':1, 'Zurich':0, 'Warsaw':1, 'Budapest':1, 'Copenhagen':1,

       'Amsterdam':1, 'Lyon':0, 'Hamburg':0, 'Lisbon':0, 'Prague':1, 'Oslo':1,

       'Helsinki':1, 'Edinburgh':0, 'Geneva':0, 'Ljubljana':1, 'Athens':1,

       'Luxembourg':1, 'Krakow':0}



data['Capital'] = data.City.apply(lambda x: capital[x])
city_size = {'Paris':3, 'Stockholm':2, 'London':3, 'Berlin':3, 'Munich':2, 'Oporto':1,

       'Milan':2, 'Bratislava':1, 'Vienna':1, 'Rome':1, 'Barcelona':2, 'Madrid':3,

       'Dublin':1, 'Brussels':1, 'Zurich':1, 'Warsaw':2, 'Budapest':2, 'Copenhagen':2,

       'Amsterdam':2, 'Lyon':1, 'Hamburg':2, 'Lisbon':1, 'Prague':2, 'Oslo':2,

       'Helsinki':2, 'Edinburgh':1, 'Geneva':1, 'Ljubljana':1, 'Athens':3,

       'Luxembourg':2, 'Krakow':2}



data['City_size'] = data.City.apply(lambda x: city_size[x])

data.head(5)
data.sample(5)
data['City_name'] = data['City']
# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

data = pd.get_dummies(data, columns=[ 'City_name',], dummy_na=True)
data.sample(5)
rest_num_id = data.Restaurant_id.value_counts().to_dict()

data['rest_num_id'] = data.Restaurant_id.apply(lambda x: rest_num_id[x])
data.info()
data['Price Range'].value_counts()
print(data['Price Range'].unique())

def price_identification(value):

    if value == 0:

        return 0

    elif value == '$':

        return 1

    elif value == '$$ - $$$':

        return 2

    elif value == '$$$$':

        return 3

data['Price Range'] = data['Price Range'].apply(lambda x: price_identification(x))
data.sample(5)
data.Reviews[4]
data["Reviews"] = data["Reviews"].str.replace("nan", "' '")

data["Reviews"].fillna("[[], []]", inplace=True)
data.Reviews[6]
rx = r'(\d\d/\d\d/\d{4})'



data['reviews_Date'] = data["Reviews"].str.findall(rx).apply(','.join)
new_df = data['reviews_Date'].str.split(',',expand=True)

new_df.columns = ['reviews_Date1','reviews_Date2']

new_df['reviews_Date1'] = pd.to_datetime(new_df['reviews_Date1'], utc=True)

new_df['reviews_Date2'] = pd.to_datetime(new_df['reviews_Date2'], errors='ignore', utc=True)

new_df['reviews_Date2'] = new_df['reviews_Date2'].dt.date

new_df['reviews_Date1'] = new_df['reviews_Date1'].dt.date

new_df['reviews_Date1 - reviews_Date2'] = (new_df['reviews_Date1'] - new_df['reviews_Date2'])
data = pd.concat((data, new_df), axis = 1)

data.drop(['reviews_Date'],axis=1, inplace=True)

data.sample(4)
data['reviews_Date1'].fillna(0, inplace=True)

data['reviews_Date2'].fillna(0, inplace=True)

data['reviews_Date1 - reviews_Date2'].fillna(0, inplace=True)
data.info()
data['Cuisine Style'].describe()
data['Cuisine Style'] = data['Cuisine Style'].str.replace('[','')

data['Cuisine Style'] = data['Cuisine Style'].str.replace(']','')

data['Cuisine Style'] = data['Cuisine Style'].str.replace("'",'')

data['Cuisine Style'] = data['Cuisine Style'].str.replace(",",'|')

len(data['Cuisine Style'].values[0].split('|'))
def num_1 (x):

    y = 0

    return y

data['couisin_num'] = data.apply(num_1, axis =1)

data['couisin_num'] = data.couisin_num.apply(lambda x: len(data['Cuisine Style'][x].split('|')))

data['couisin_num'][0]
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
df_train.info()
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
def price_identification(value):

        if value == 0:

            return 0

        elif value == '$':

            return 1

        elif value == '$$ - $$$':

            return 2

        elif value == '$$$$':

            return 3

def num_1 (x):

        y = 0

        return y
def preproc_data(df_input):

    '''includes several functions to pre-process the predictor data.'''

    

    df_output = df_input.copy()

    

    # ################### 1. Предобработка ############################################################## 

    # убираем не нужные для модели признаки

    df_output.drop(['ID_TA',], axis = 1, inplace=True)

    

    

    # ################### 2. NAN ############################################################## 

    # Далее заполняем пропуски, вы можете попробовать заполнением средним или средним по городу и тд...

    df_output['Number of Reviews'].fillna(0, inplace=True)

    df_output['Reviews'].fillna(0, inplace=True)

    

    df_output['Price Range'].fillna(0, inplace=True)

    # тут ваш код по обработке NAN

    # ....

    

    

    # ################### 3. Encoding ############################################################## 

    # для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

    df_output['City_name'] = df_output['City']

    df_output = pd.get_dummies(df_output, columns=[ 'City_name',], dummy_na=True)

    

    

    

    # ################### 4. Feature Engineering ####################################################

    # тут ваш код не генерацию новых фитчей

    capital = {'Paris':1, 'Stockholm':1, 'London':1, 'Berlin':1, 'Munich':0, 'Oporto':0,

       'Milan':0, 'Bratislava':1, 'Vienna':2, 'Rome':3, 'Barcelona':0, 'Madrid':1,

       'Dublin':1, 'Brussels':1, 'Zurich':0, 'Warsaw':1, 'Budapest':1, 'Copenhagen':1,

       'Amsterdam':1, 'Lyon':0, 'Hamburg':0, 'Lisbon':0, 'Prague':1, 'Oslo':1,

       'Helsinki':1, 'Edinburgh':0, 'Geneva':0, 'Ljubljana':1, 'Athens':1,

       'Luxembourg':1, 'Krakow':0}



    df_output['Capital'] = df_output.City.apply(lambda x: capital[x])

    

    city_size = {'Paris':3, 'Stockholm':2, 'London':3, 'Berlin':3, 'Munich':2, 'Oporto':1,

       'Milan':2, 'Bratislava':1, 'Vienna':1, 'Rome':1, 'Barcelona':2, 'Madrid':3,

       'Dublin':1, 'Brussels':1, 'Zurich':1, 'Warsaw':2, 'Budapest':2, 'Copenhagen':2,

       'Amsterdam':2, 'Lyon':1, 'Hamburg':2, 'Lisbon':1, 'Prague':2, 'Oslo':2,

       'Helsinki':2, 'Edinburgh':1, 'Geneva':1, 'Ljubljana':1, 'Athens':3,

       'Luxembourg':2, 'Krakow':2}



    df_output['City_size'] = data.City.apply(lambda x: city_size[x])

    

    rest_num_id = df_output.Restaurant_id.value_counts().to_dict()

    df_output['rest_num_id'] = df_output.Restaurant_id.apply(lambda x: rest_num_id[x])

    df_output['Price Range'] = df_output['Price Range'].apply(lambda x: price_identification(x))

    

        

    df_output['Cuisine Style'] = df_output['Cuisine Style'].str.replace('[','')

    df_output['Cuisine Style'] = df_output['Cuisine Style'].str.replace(']','')

    df_output['Cuisine Style'] = df_output['Cuisine Style'].str.replace("'",'')

    df_output['Cuisine Style'] = df_output['Cuisine Style'].str.replace(",",'|')

   

       

    df_output['couisin_num'] = df_output.apply(num_1, axis =1)

    df_output['couisin_num'] = df_output.couisin_num.apply(lambda x: len(df_output['Cuisine Style'][x].split('|')))

    df_output['couisin_num']

    

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