import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline

import datetime
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

import re 

# Загружаем специальный удобный инструмент для разделения датасета:
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
from sklearn import metrics # инструменты для оценки точности модели

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# фиксируем RANDOM_SEED, чтобы эксперименты были воспроизводимы:
RANDOM_SEED = 42
# фиксируем версию пакетов, чтобы эксперименты были воспроизводимы:
!pip freeze > requirements.txt
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'
df_train = pd.read_csv(DATA_DIR+'/main_task.csv')
df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')
sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
# ВАЖНО! для корректной обработки признаков объединяем трейн и тест в один датасет
df_train['sample'] = 1 # помечаем где у нас трейн
df_test['sample'] = 0 # помечаем где у нас тест
df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями

data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
data.head()
data.info()
fig, ax = plt.subplots(figsize=(15, 5))
sns_heatmap = sns.heatmap(
    data.isnull(), yticklabels=False, cbar=False)
plt.figure(figsize = (16, 6))

plt.subplot(1,2,1)
plt.hist(df_train['Rating'])
plt.title('train data')

plt.subplot(1,2,2)
plt.hist(df_test['Rating'])
plt.title('test data')
data['cuisine_style_isNAN'] = pd.isna(data['Cuisine Style']).astype('uint8')
data['price_range_isNAN'] = pd.isna(data['Price Range']).astype('uint8')
data['number_of_reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')
data['ranked_cities'] = data['City'].rank()
data['City'].nunique()
data['City'].value_counts()
data['City'].value_counts(ascending=True).plot(kind='barh')
data['rest_counts'] = data['City'].apply(lambda x: data['City'].value_counts()[x])
data['reviews_in_city'] = data['City'].apply(lambda x: data.groupby(['City'])\
                                             ['Number of Reviews'].sum().sort_values(ascending=False)[x])
population = {'London': 8.982,'Paris': 2.148, 'Madrid': 6.642, 'Berlin': 3.769,  'Rome': 2.873 , 'Prague': 1.309,
              'Lisbon': 0.504, 'Vienna': 1.897, 'Amsterdam': 0.821, 'Brussels': 0.174, 'Stockholm': 0.975, 
              'Budapest': 1.752, 'Warsaw': 1.708, 'Dublin': 1.388, 'Copenhagen': 0.602, 'Athens': 0.664, 
              'Edinburgh': 0.482, 'Oslo': 0.681, 'Helsinki': 0.631, 'Bratislava': 0.424, 'Luxembourg': 0.116, 
              'Ljubljana': 0.279, 'Munich': 1.472, 'Oporto': 0.214, 'Milan': 1.352, 'Barcelona': 5.575,
              'Zurich': 0.402, 'Lyon': 0.513, 'Hamburg': 1.899, 'Geneva': 0.499, 'Krakow': 0.769}

data['population'] = data['City'].map(population)
data['rest_density'] = data['rest_counts'] / data['population']
data['Cuisine Style'] = data['Cuisine Style'].fillna(0)

count = []
for i in data['Cuisine Style'].values:
    if i == 0:
        count.append(1)
    else:
        count.append(len(i.split(',')))

data['count_cuisine'] = count
for x in (data['City'].value_counts())[0:10].index:
    data['Ranking'][data['City'] == x].hist(bins=100)
plt.show()
data['relative_rank'] = data['Ranking'] / data['rest_counts']
data['relative_rank_reviews'] = data['Ranking'] / data['reviews_in_city']
data['Price Range'].value_counts()
restaurant_level = []

for i in data['Price Range']:
    if i == '$':
        restaurant_level.append(1)
    elif i == '$$ - $$$':
        restaurant_level.append(2)
    elif i == '$$$$':
        restaurant_level.append(3)
    else:
        restaurant_level.append(2)

data['restaurant_level'] = restaurant_level
data['Number of Reviews'] = data['Number of Reviews'].fillna(0)
data['Reviews'] = data['Reviews'].fillna(0)
reviews_date = []
for i in data['Reviews']:
    if i == '[[], []]' or i == 0:
        reviews_date.append('')
    else:
        i = str(i).replace(']]', '')
        i = i.replace("'", '')
        i = i.split('], [')[1]
        i = i.split(', ')
        reviews_date.append(i)

data['reviews_date'] = reviews_date      
        
data['reviews_date_1'] = data['reviews_date'].apply(lambda x: x[1] if len(x) == 2 else None)
data['reviews_date_2'] = data['reviews_date'].apply(lambda x: x[0] if len(x) > 0 else None)

data['reviews_date_1'] = pd.to_datetime(data['reviews_date_1'])
data['reviews_date_2'] = pd.to_datetime(data['reviews_date_2'])

data['reviews_date_1'] = data['reviews_date_1'].apply(lambda x: data['reviews_date_1'].min()\
                                                      if pd.isnull(x) else x)
data['reviews_date_2'] = data['reviews_date_2'].apply(lambda x: data['reviews_date_1'].min()\
                                                      if pd.isnull(x) else x)

data['between_dates'] = (data['reviews_date_2'] - data['reviews_date_1']).dt.days
data['days_to_today'] = (datetime.now() - data['reviews_date_2']).dt.days
data['id_ta'] = data['ID_TA'].apply(lambda x: int(x[1:]))
data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
data = data.drop(['Restaurant_id', 'Cuisine Style', 'Price Range', 'Reviews', 'URL_TA', 'ID_TA', 'reviews_date_1', 
           'reviews_date_2', 'reviews_date'
                 ], axis = 1)
plt.rcParams['figure.figsize'] = (15, 10)
sns.heatmap(data.corr())
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
# Создаём модель 
model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)
data.info()
# Обучаем модель на тестовом наборе данных
model.fit(X_train, y_train)

# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
# Предсказанные значения записываем в переменную y_pred
y_pred = model.predict(X_test)

# Округлим предсказанные значения до степени округления целевой переменной
y_pred = np.round(y_pred*2)/2
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются
# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# в RandomForestRegressor есть возможность вывести самые важные признаки для модели
plt.rcParams['figure.figsize'] = (10,10)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
test_data = test_data.drop(['Rating'], axis=1)
sample_submission
predict_submission = model.predict(test_data)
predict_submission = np.round(predict_submission*2)/2
predict_submission
sample_submission['Rating'] = predict_submission
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head(10)
