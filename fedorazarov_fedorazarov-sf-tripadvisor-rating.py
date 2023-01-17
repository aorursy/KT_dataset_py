# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re



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
# Для примера я возьму столбец Number of Reviews

data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')
data['Number_of_Reviews_isNAN']
# Далее заполняем пропуски 0, вы можете попробовать заполнением средним или средним по городу и тд...

data['Number of Reviews'].fillna(0, inplace=True)
data.nunique(dropna=False)
#Создадем новый датафрейм с количеством ресторанов в каждом городе (по данным имеющегося датафрейма) 

df2 = data['City'].value_counts().rename_axis('City').to_frame(name='Rest Count')

#Добавим количество ресторанов в исходный датафрейм

data = data.merge(df2, on='City', how='left')

#Новый ранг - отношение ранга к количеству ресторанов в городе

data['New Rank'] = data['Ranking'] / data['Rest Count']
data.head()
# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
data.head(5)
data.sample(5)
data['Price Range'].value_counts()
#Преобразуем Price Range

#Заполняем пустые значения колонки Price Range самой распространенной ценовой категорией

data['Price Range'] = data['Price Range'].fillna('$$ - $$$')

#Заменяем значения на числа

data['Price Range'] = data['Price Range'].replace('$', '1')

data['Price Range'] = data['Price Range'].replace('$$ - $$$', '2')

data['Price Range'] = data['Price Range'].replace('$$$$', '3')
#Кухни

import re

#Преобразуем Cuisine Style в список кухонь

data['Cuisine Style'].fillna("[]", inplace=True)

data['Cuisine Style'] = data['Cuisine Style'].apply(lambda x: re.findall('\w+\s*\w+\s*\w+', str(x)))

#Добавляем признак с количеством кухонь, меняем 0 на 1

data['Number of Cuisines'] = data['Cuisine Style'].apply(lambda x: len(x))

data['Number of Cuisines'] = data['Number of Cuisines'].replace(0,1)
#Даты комментариев

#Добавляем 3 новых столбца: время между последним комментарием в датасете и последнем в ресторане,

#аналогично с предпоследним

#время между комментариями

import datetime as dt

pattern = re.compile('\d{2}/\d{2}/\d{4}')

reviews=[]

for i in data['Reviews']:

    reviews.append(re.findall(pattern, str(i)))

rev=pd.DataFrame(reviews).dropna()

rev.columns=['date1', 'date2']

rev['date1'] = pd.to_datetime(rev['date1']) 

rev['date2'] = pd.to_datetime(rev['date2']) 

rev['dd']=rev['date1']-rev['date2']

data['date1'] = rev['date1'].max() - rev['date1']

data['date1'] = data['date1'].apply(lambda x: x.days)

data['date1'] = data['date1'].fillna(0)

data['date1'] = data['date1'].apply(lambda x: int(x))

data['date2'] = rev['date1'].max() - rev['date2']

data['date2'] = data['date2'].apply(lambda x: x.days)

data['date2'] = data['date2'].fillna(0)

data['date2'] = data['date2'].apply(lambda x: int(x))

data['dd']=data['date2']-data['date1']
#Создаем dummy variables для видов кухонь

#Функция для отображения кухонь в записи

def find_item(cell):

    if item in cell:

        return 1

    return 0

#Создаем набор кухонь

cuisines = set()

for cusinelist in data['Cuisine Style']:

    for cuisine in cusinelist:

        cuisines.add(cuisine)

#создаем столбцы с кухнями и заполняем

for item in cuisines:

    data[item] = data['Cuisine Style'].apply(find_item)
data.head(5)
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
def find_item(cell):

    if item in cell:

        return 1

    return 0



def preproc_data(df_input):

    '''includes several functions to pre-process the predictor data.'''

    

    df_output = df_input.copy()

    

    # ################### 1. Предобработка ############################################################## 

    # убираем не нужные для модели признаки

    df_output.drop(['Restaurant_id','ID_TA',], axis = 1, inplace=True)

    

    

    # ################### 2. NAN ############################################################## 

    # Далее заполняем пропуски, вы можете попробовать заполнением средним или средним по городу и тд...

    df_output['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')

    df_output['Number of Reviews'].fillna(1, inplace=True)

    df_output['Price Range_isNAN'] = pd.isna(data['Price Range']).astype('uint8')

    df_output['Cuisine Style_isNAN'] = pd.isna(data['Cuisine Style']).astype('uint8')

    # тут ваш код по обработке NAN

    # ....

    

    

    # ################### 3. Encoding ############################################################## 

    

    #Признак New Rank - отношение ранга к количеству ресторанов в городе

    #Создадем новый с количеством ресторанов в каждом городе (по данным имеющегося датафрейма) 

    df2 = df_output['City'].value_counts().rename_axis('City').to_frame(name='Rest Count')

    #Добавим количество ресторанов в исходный датафрейм

    df_output = df_output.merge(df2, on='City', how='left')

    #Новый ранг - отношение ранга к количеству ресторанов в городе

    df_output['New Rank'] = df_output['Ranking'] / df_output['Rest Count']

    

    # для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

    df_output = pd.get_dummies(df_output, columns=[ 'City',], dummy_na=True)

    # тут ваш код не Encoding фитчей

    #Кухни

    

    #Преобразуем Cuisine Style в список кухонь

    df_output['Cuisine Style'].fillna("[]", inplace=True)

    df_output['Cuisine Style'] = df_output['Cuisine Style'].apply(lambda x: re.findall('\w+\s*\w+\s*\w+', str(x)))

    #Добавляем признак с количеством кухонь, меняем 0 на 1

    df_output['Number of Cuisines'] = df_output['Cuisine Style'].apply(lambda x: len(x))

    df_output['Number of Cuisines'] = df_output['Number of Cuisines'].replace(0,1)

    

    #Создаем dummy variables для видов кухонь

    #Функция для отображения кухонь в записи



    #Создаем набор кухонь

    cuisines = set()

    for cusinelist in df_output['Cuisine Style']:

        for cuisine in cusinelist:

            cuisines.add(cuisine)

    #создаем столбцы с кухнями и заполняем

    for item in cuisines:

        df_output[item] = df_output['Cuisine Style'].apply(find_item)

    

    # ################### 4. Feature Engineering ####################################################

    # тут ваш код не генерацию новых фитчей

    #Преобразуем Price Range

    #Заполняем пустые значения колонки Price Range самой распространенной ценовой категорией

    df_output['Price Range'] = df_output['Price Range'].fillna('$$ - $$$')

    #Заменяем значения на числа

    df_output['Price Range'] = df_output['Price Range'].replace('$', '1')

    df_output['Price Range'] = df_output['Price Range'].replace('$$ - $$$', '2')

    df_output['Price Range'] = df_output['Price Range'].replace('$$$$', '3')



    #Даты комментариев

    #Добавляем 3 новых столбца: время между последним комментарием в датасете и последнем в ресторане,

    #аналогично с предпоследним

    #время между комментариями

    import datetime as dt

    pattern = re.compile('\d{2}/\d{2}/\d{4}')

    reviews=[]

    for i in data['Reviews']:

        reviews.append(re.findall(pattern, str(i)))

    rev=pd.DataFrame(reviews).dropna()

    rev.columns=['date1', 'date2']

    rev['date1'] = pd.to_datetime(rev['date1']) 

    rev['date2'] = pd.to_datetime(rev['date2']) 

    rev['dd']=rev['date1']-rev['date2']

    df_output['date1'] = rev['date1'].max() - rev['date1']

    df_output['date1'] = df_output['date1'].apply(lambda x: x.days)

    df_output['date1'] = df_output['date1'].fillna(0)

    df_output['date1'] = df_output['date1'].apply(lambda x: int(x))

    df_output['date2'] = rev['date1'].max() - rev['date2']

    df_output['date2'] = df_output['date2'].apply(lambda x: x.days)

    df_output['date2'] = df_output['date2'].fillna(0)

    df_output['date2'] = df_output['date2'].apply(lambda x: int(x))

    df_output['dd']=df_output['date2']-df_output['date1']

    

    # ################### 5. Clean #################################################### 

    # убираем признаки которые еще не успели обработать, 

    # df_output.drop(['Ranking','Rest Count',], axis = 1, inplace=True)    

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