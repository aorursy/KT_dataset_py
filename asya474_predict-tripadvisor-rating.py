import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline

import datetime

from datetime import datetime, timedelta

import re

# Загружаем специальный удобный инструмент для разделения датасета:

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
RANDOM_SEED = 42
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
# Для корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
data.info()
data.sample(5)
data.Reviews[1]
data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')

data['Number of Reviews'].fillna(value=0, inplace=True)

data['Price Range'].fillna(value=0, inplace=True)

#data['Cuisine Style'] = data['Cuisine Style'].fillna(value='Other', inplace=True)

data.head()
data.info()
data['Restaurant_id'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')

data['Restaurant_id']=pd.to_numeric(data['Restaurant_id'], )
price_dict={0:0, '$':1, '$$ - $$$':2, '$$$$':3}

data['Price Range'] = data['Price Range'].replace(to_replace=price_dict)
cities_dummies=pd.get_dummies(data['City'], drop_first=True)

data = pd.concat([pd.DataFrame(data), cities_dummies], axis=1)

data.drop(columns='City', inplace=True)
data['Cuisine Style'] = data.apply(lambda x: x['Cuisine Style'].replace('[', '').replace(']', '').replace(

    "'", '').replace(' ', '') if type(x['Cuisine Style']) != float else x['Cuisine Style'], axis=1)

data['cuisines_count'] = data['Cuisine Style'].str.split(',').str.len().fillna(1)

cuisines = data['Cuisine Style'].str.get_dummies(

    ',').sum().sort_values(ascending=False)

top_cuisines = [x for x in cuisines.index if cuisines[x] < 1000]

data = data.join(data['Cuisine Style'].str.get_dummies(

    ',').drop(top_cuisines, axis=1), how='left')

data.drop(columns='Cuisine Style', inplace=True)

data.head()

data.info()
data['Reviews']=data['Reviews'].astype(str)

data['Reviews'] = data.apply(lambda x: x['Reviews'].replace('[[], []]', 'No reviews'), axis=1)

data['Reviews'] = data['Reviews'].apply(lambda x: x.replace(

    '[[', ''))

data['Reviews'] = data['Reviews'].apply(lambda x: x.replace(

    ']]', ''))

data['Reviews'] = data['Reviews'].apply(lambda x: x.replace(

    '[', ''))

data['Reviews'] = data['Reviews'].apply(lambda x: x.replace(

    ']', ''))

data['Reviews'] = data['Reviews'].apply(lambda x: str(x) if type(x) == list else x)

time_reviews = []

for item in data['Reviews']:

    time_reviews.append(re.findall(r'(\d\d/\d\d/\d\d\d\d)', item))

reviews = pd.DataFrame(time_reviews)

data['first_date_reviews'] = pd.to_datetime(reviews[0])

data['second_date_reviews']= pd.to_datetime(reviews[1])

data['difference_between_reviews_date'] = data['first_date_reviews']-data['second_date_reviews']

data['difference_between_reviews_date'].max()

data['difference_between_reviews_date'].fillna(value=pd.Timedelta(seconds=0), inplace=True)

data['difference_between_reviews_date']=(data['difference_between_reviews_date'] / np.timedelta64(1, 'D')).astype(int) 

data.drop(columns=['first_date_reviews', 'second_date_reviews'], axis=1, inplace=True)
data['ID_TA']=data['ID_TA'].apply(lambda x: x.replace('d', ''))

data['ID_TA']=data['ID_TA'].astype(int)

df_selected = data[['Restaurant_id', 'Ranking', 'Rating', 'Price Range',

                 'Number of Reviews', 'ID_TA', 'difference_between_reviews_date']]
data.drop(columns=['Reviews'], axis=1, inplace=True)

data.drop(columns=['URL_TA'], axis=1, inplace=True)

data.info()
data.sample()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit_transform(data)
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
import seaborn as sns

corr = df_selected.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
corr = df_selected.corr()

corr.style.background_gradient(cmap='coolwarm')
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

    df_output.drop(['Restaurant_id'], axis = 1, inplace=True)

    df_output['Number_of_Reviews_isNAN'] = pd.isna(df_output['Number of Reviews']).astype('uint8')

    df_output['Number of Reviews'].fillna(value=0, inplace=True)

    df_output['Price Range'].fillna(value=0, inplace=True)

    df_output['Cuisine Style'].fillna(value='Other', inplace=True)

    df_output['Reviews'].fillna(value='0000-00-00 00:00:00', inplace=True)

    price_dict={0:0, '$':1, '$$ - $$$':2, '$$$$':3}

    df_output['Price Range'] = df_output['Price Range'].replace(to_replace=price_dict)

    cities_dummies=pd.get_dummies(df_output['City'], drop_first=True)

    df_output = pd.concat([pd.DataFrame(df_output), cities_dummies], axis=1)

    df_output.drop(columns='City', inplace=True)

    df_output['Cuisine Style'] = df_output.apply(lambda x: x['Cuisine Style'].replace('[', '').replace(']', '').replace(

        "'", '').replace(' ', '') if type(x['Cuisine Style']) != float else x['Cuisine Style'], axis=1)

    df_output['cuisines_count'] = df_output['Cuisine Style'].str.split(',').str.len().fillna(1)

    cuisines = df_output['Cuisine Style'].str.get_dummies(

        ',').sum().sort_values(ascending=False)

    top_cuisines = [x for x in cuisines.index if cuisines[x] < 1000]

    df_output = df_output.join(df_output['Cuisine Style'].str.get_dummies(

        ',').drop(top_cuisines, axis=1), how='left')

    df_output.drop(columns='Cuisine Style', inplace=True)

    df_output['Reviews']=df_output['Reviews'].astype(str)

    df_output['Reviews'] = df_output.apply(lambda x: x['Reviews'].replace('[[], []]', 'No reviews'), axis=1)

    df_output['Reviews'] = df_output['Reviews'].apply(lambda x: x.replace(

        '[[', ''))

    df_output['Reviews'] = df_output['Reviews'].apply(lambda x: x.replace(

        ']]', ''))

    df_output['Reviews'] = df_output['Reviews'].apply(lambda x: x.replace(

        '[', ''))

    df_output['Reviews'] = df_output['Reviews'].apply(lambda x: x.replace(

        ']', ''))

    df_output['Reviews'] = df_output['Reviews'].apply(lambda x: str(x) if type(x) == list else x)

    time_reviews = []

    for item in df_output['Reviews']:

        time_reviews.append(re.findall(r'(\d\d/\d\d/\d\d\d\d)', item))

    reviews = pd.DataFrame(time_reviews)

    df_output['first_date_reviews'] = pd.to_datetime(reviews[0])

    df_output['second_date_reviews']= pd.to_datetime(reviews[1])

    df_output['difference_between_reviews_date'] = df_output['first_date_reviews']-df_output['second_date_reviews']

    df_output['difference_between_reviews_date'].max()

    df_output['difference_between_reviews_date'].fillna(value=pd.Timedelta(seconds=0), inplace=True)

    df_output['difference_between_reviews_date']=(df_output['difference_between_reviews_date'] / np.timedelta64(1, 'D')).astype(int) 

    df_output.drop(columns=['first_date_reviews', 'second_date_reviews'], axis=1, inplace=True)

    df_output['ID_TA']=df_output['ID_TA'].apply(lambda x: x.replace('d', ''))

    df_output['ID_TA']=df_output['ID_TA'].astype(int)

    df_output.drop(columns=['Reviews'], axis=1, inplace=True)

    df_output.drop(columns=['URL_TA'], axis=1, inplace=True)

    object_columns = [s for s in df_output.columns if df_output[s].dtypes == 'object']

    df_output.drop(object_columns, axis = 1, inplace=True)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()

    scaler.fit_transform(df_output)

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