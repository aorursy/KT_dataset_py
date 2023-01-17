# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



# Загружаем специальный инструмент для формирования dummies из данных, представленных в виде списка:

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()



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

df_train = pd.read_csv(DATA_DIR+'main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'sample_submission.csv')
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
data.columns
data.columns = ['restaurant_id','city','cuisine_style','ranking','price_range','number_of_reviews','reviews','url_ta','id_ta','sample','rating']
data.isna().sum()
display(data.number_of_reviews.describe())

display(data.number_of_reviews.value_counts())

display(data.number_of_reviews.isna().sum())
data.number_of_reviews.fillna(0, inplace=True)

display(data.number_of_reviews.describe())

display(data.number_of_reviews.value_counts())
display(data.price_range.describe())

display(data.price_range.value_counts())

display(data.price_range.isna().sum())
data.price_range.fillna('0$', inplace=True) 

display(data.price_range.describe())

display(data.price_range.value_counts())

display(data.price_range.isna().sum())
display(data.cuisine_style.describe())

display(data.cuisine_style.value_counts())

display(data.cuisine_style.isna().sum())
data.cuisine_style.fillna('Other', inplace=True) 

display(data.cuisine_style.describe())

display(data.cuisine_style.value_counts())

display(data.cuisine_style.isna().sum())
display(data.reviews.describe())

display(data.reviews.value_counts())

display(data.reviews.isna().sum())
data.reviews.fillna(data.reviews.mode()[0],inplace=True) 

display(data.reviews.describe())

display(data.reviews.value_counts())

display(data.reviews.isna().sum())
data.info()
display(data.nunique(dropna=False))

data
def filter_cuisine_style(x):

    x = x.replace('[','')

    x = x.replace(']','')

    x = x.strip()

    x = [style.strip() for style in x.split(',')]

    x = [style for style in x if len(style) > 0]

    return x



data['cuisine_style'] = data['cuisine_style'].apply(filter_cuisine_style)

data_cuisine_90 = data.explode(column='cuisine_style')

data = data.join(pd.DataFrame(mlb.fit_transform(data.pop('cuisine_style')), index=data.index, columns=mlb.classes_))

data
data_cuisine_90 = data_cuisine_90[data_cuisine_90['cuisine_style'] != 'Other']

all_cuisine = data_cuisine_90['cuisine_style'].value_counts()

top_cuisine = all_cuisine.head(40).index

all_cuisine = data_cuisine_90['cuisine_style'].value_counts().index

cuisine_to_throw_away = list(set(all_cuisine) - set(top_cuisine))

cuisine_to_throw_away
data.drop(cuisine_to_throw_away, axis = 1, inplace=True)

data.drop('Other', axis = 1, inplace=True)

data
dummies = pd.get_dummies(data.city).rename(columns=lambda x: 'city_' + str(x))

data = pd.concat([data, dummies], axis=1)

data
city_dict = {'London':8982,'Paris':2148,'Madrid':6642,'Barcelona':5575, 'Berlin':3769, 'Milan':1352,

            'Rome':2873, 'Prague':1309, 'Lisbon': 0.504718, 'Vienna':1897, 'Amsterdam': 0.821752,'Brussels': 0.174383, 'Budapest':1.752, 'Dublin': 1.388,

           'Copenhagen':0.602481,'Athens':0.664046,'Edinburgh':0.482005, 'Zurich':0.402762, 'Oporto':0.214349,'Geneva':0.499408,'Hamburg':1.899, 'Stockholm':0.975904, 'Munich':1.473, 'Warsaw':1.798,

             'Krakow':0.769498,'Lyon':0.513275,'Oslo':0.681067,'Helsinki':0.631695, 'Bratislava':0.422428,'Luxembourg':0.613894,'Ljubljana':0.279631}

display(city_dict)



data['city_count'] = data.city

data['city_count'] = data['city_count'].replace(to_replace=city_dict)

data
price_dict = {'0$':0,'$':1,'$$ - $$$':2,'$$$$':3,}

data.price_range = data.price_range.replace(to_replace=price_dict)

display(data.price_range.value_counts())

data
display(data.number_of_reviews.describe())



fig = plt.figure()

axes = fig.add_axes([0, 0, 1, 1])

axes.hist(data.number_of_reviews)
fig = plt.figure()

axes = fig.add_axes([0, 0, 1, 1])

np.sqrt(data.number_of_reviews[data.number_of_reviews > 0]).hist(bins=100)
data['sqrt_number_of_reviews'] = round(np.sqrt(data.number_of_reviews[data.number_of_reviews >= 0]))

data
def new_reviews(x):

    if x == '[[], []]':

        return []

    else:

        x = x.replace(']]', '')

        x = x.replace("'", '')

        x = x.split('], [')[1]

        x = x.split(', ')

        return x

data['dates_of_reviews'] = data['reviews'].apply(new_reviews)

data[['date_1', 'date_2']] = pd.DataFrame(data['dates_of_reviews'].tolist())



data['date_1'] = pd.to_datetime(data['date_1']).dt.date #Дата первого отзыва.

data['date_2'] = pd.to_datetime(data['date_2']).dt.date #Дата второго отзыва.

data['timedelta'] = data['date_1'] - data['date_2']



display(data.timedelta.describe())

display(data.timedelta.value_counts())

display(data.timedelta.isna().sum())

data
data.timedelta.fillna(data.timedelta.mode()[0],inplace=True) 

display(data.timedelta.describe())

display(data.timedelta.value_counts())

display(data.timedelta.isna().sum())
fig = plt.figure()

axes = fig.add_axes([0, 0, 1, 1])

axes.hist(data.ranking/data.city_count)
data['ranking_vs_city_count'] = data['ranking']/data['city_count']

data
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
object_columns = [s for s in data.columns if data[s].dtypes == 'object']

data.drop(object_columns, axis = 1, inplace=True)

data.drop(columns = ['timedelta'], axis = 1, inplace=True)

data
# на всякий случай, заново подгружаем данные

df_train = pd.read_csv(DATA_DIR+'main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем



data.info()
def filter_cuisine_style(x):

    x = x.replace('[','')

    x = x.replace(']','')

    x = x.strip()

    x = [style.strip() for style in x.split(',')]

    x = [style for style in x if len(style) > 0]

    return x



def preproc_data(df_input):

    '''includes several functions to pre-process the predictor data.'''

    

    df_output = df_input.copy()

    

    # ################### 1. Предобработка ############################################################## 

    # переименовываем названия столбцов в исходном датасете в единый формат

    df_output.columns = ['restaurant_id','city','cuisine_style','ranking','price_range','number_of_reviews','reviews','url_ta','id_ta','sample','rating']

    # убираем не нужные для модели признаки

    df_output.drop(['restaurant_id','id_ta',], axis = 1, inplace=True)

    # задаем используемые функции и добавляем словари

    city_dict = {'London':8982,'Paris':2148,'Madrid':6642,'Barcelona':5575, 'Berlin':3769, 'Milan':1352, 'Rome':2873, 'Prague':1309, 

                 'Lisbon': 0.504718, 'Vienna':1897, 'Amsterdam': 0.821752,'Brussels': 0.174383, 'Budapest':1.752, 'Dublin': 1.388,

                 'Copenhagen':0.602481,'Athens':0.664046,'Edinburgh':0.482005, 'Zurich':0.402762, 'Oporto':0.214349,'Geneva':0.499408,

                 'Hamburg':1.899, 'Stockholm':0.975904, 'Munich':1.473, 'Warsaw':1.798, 'Krakow':0.769498,'Lyon':0.513275,'Oslo':0.681067,

                 'Helsinki':0.631695, 'Bratislava':0.422428,'Luxembourg':0.613894,'Ljubljana':0.279631}



    price_dict = {'0$':0,'$':1,'$$ - $$$':2,'$$$$':3,}





    

    # ################### 2. NAN ############################################################## 

    # Далее заполняем пропуски и формируем промежуточные выборки для следующего шага

    df_output.number_of_reviews.fillna(0, inplace=True)

    

    df_output.price_range.fillna('0$', inplace=True)

    

    df_output.reviews.fillna(df_output.reviews.mode()[0], inplace=True)

    

    df_output.cuisine_style.fillna('Other', inplace=True)

    df_output['cuisine_style'] = df_output['cuisine_style'].apply(filter_cuisine_style)

    data_cuisine_90 = df_output.explode(column='cuisine_style')

    data_cuisine_90 = data_cuisine_90[data_cuisine_90['cuisine_style'] != 'Other']

    all_cuisine = data_cuisine_90['cuisine_style'].value_counts()

    top_cuisine = all_cuisine.head(40).index

    all_cuisine = data_cuisine_90['cuisine_style'].value_counts().index

    cuisine_to_throw_away = list(set(all_cuisine) - set(top_cuisine))



    

    # ################### 3. Encoding ############################################################## 

    # для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

    df_output = df_output.join(pd.DataFrame(mlb.fit_transform(df_output.pop('cuisine_style')), index=df_output.index, columns=mlb.classes_))

    

    dummies_city = pd.get_dummies(df_output.city).rename(columns=lambda x: str(x))

    df_output = pd.concat([df_output, dummies_city], axis=1)

    

    # тут ваш код не Encoding фитчей

    df_output.price_range = df_output.price_range.replace(to_replace=price_dict)



   



    # ################### 4. Feature Engineering ####################################################

    # тут ваш код не генерацию новых фитчей

    df_output['city_count'] = df_output.city

    df_output['city_count'] = df_output['city_count'].replace(to_replace=city_dict)  

    

    df_output['sqrt_number_of_reviews'] = round(np.sqrt(df_output.number_of_reviews[df_output.number_of_reviews >= 0]))

    df_output['ranking_vs_city_count'] = df_output['ranking']/df_output['city_count']

    

    # ################### 5. Clean #################################################### 

    # убираем признаки которые еще не успели обработать, 

    # модель на признаках с dtypes "object" обучаться не будет, просто выберем их и удалим

    object_columns = [s for s in df_output.columns if df_output[s].dtypes == 'object']

    df_output.drop(object_columns, axis = 1, inplace=True)

    df_output.drop(cuisine_to_throw_away, axis = 1, inplace=True)

    df_output.drop('Other', axis = 1, inplace=True)

    

    return df_output
df_preproc = preproc_data(data)

df_preproc.sample(10)
df_preproc.info()
# Теперь выделим тестовую часть

train_data = df_preproc.query('sample == 1').drop(['sample'], axis=1)

test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1)



y = train_data.rating.values            # наш таргет

X = train_data.drop(['rating'], axis=1)
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
test_data = test_data.drop(['rating'], axis=1)
sample_submission
predict_submission = model.predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)