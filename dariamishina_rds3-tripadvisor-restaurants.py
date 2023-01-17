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
#нашла датасет со столицами стран на kaggle, буду использовать для создания нового признака - является ли город столицей
capital = pd.read_csv('/kaggle/input/world-capitals-gps/concap.csv')
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет
df_train['sample'] = 1 # помечаем где у нас трейн
df_test['sample'] = 0 # помечаем где у нас тест
df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями

data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
#посмотрим на типы данных и количество пропусков
data.info()
fig, ax = plt.subplots(figsize=(20,12))
sns_heatmap = sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
data.sample(5)
data.Reviews[1]
# В датасете 3 признака с пропусками, по всем 3м вынесем отсутствие информации в отдельный признак
data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')
data['Cuisine_Style_isNAN'] = pd.isna(data['Cuisine Style']).astype('uint8')
data['Price_Range_isNAN'] = pd.isna(data['Price Range']).astype('uint8')
data['Number of Reviews'].fillna(0, inplace=True)
data['Price Range'].value_counts(dropna = False)
data['Price Range'] = data['Price Range'].fillna(data['Price Range'].mode()[0])
data['Cuisine Style'] = data['Cuisine Style'].fillna("['Other']")
data.nunique(dropna=False)
data.drop(['Restaurant_id'], axis=1, inplace=True)
#сделаем из колонки в найденном на kaggle датафрейме (сериз) - список, так удобнее искать
caplist = capital.CapitalName.to_list()
caplist
data['is_capital'] = data.City.apply(lambda x: 1 if x in caplist else 0)
data['City'].value_counts()
rest_in_city = data['City'].value_counts()
data['Rest_in_city'] = data['City'].apply(lambda x: rest_in_city[x])
population = {
    'London': 8173900,
    'Paris': 2240621,
    'Madrid': 3155360,
    'Barcelona': 1593075,
    'Berlin': 3326002,
    'Milan': 1331586,
    'Rome': 2870493,
    'Prague': 1272690,
    'Lisbon': 547733,
    'Vienna': 1765649,
    'Amsterdam': 825080,
    'Brussels': 144784,
    'Hamburg': 1718187,
    'Munich': 1364920,
    'Lyon': 496343,
    'Stockholm': 1981263,
    'Budapest': 1744665,
    'Warsaw': 1720398,
    'Dublin': 506211 ,
    'Copenhagen': 1246611,
    'Athens': 3168846,
    'Edinburgh': 476100,
    'Zurich': 402275,
    'Oporto': 221800,
    'Geneva': 196150,
    'Krakow': 756183,
    'Oslo': 673469,
    'Helsinki': 574579,
    'Bratislava': 413192,
    'Luxembourg': 576249,
    'Ljubljana': 277554
}
data['Population'] = data['City'].map(population)
data['Rest_per_man'] = data['Population'] / data['Rest_in_city']
country = {
    'London': 'GB',
    'Paris': 'FR',
    'Madrid': 'ES',
    'Barcelona': 'ES',
    'Berlin': 'DE',
    'Milan': 'IT',
    'Rome': 'IT',
    'Prague': 'CZ',
    'Lisbon': 'PT',
    'Vienna': 'AT',
    'Amsterdam': 'NL',
    'Brussels': 'BE',
    'Hamburg': 'DE',
    'Munich': 'DE',
    'Lyon': 'FR',
    'Stockholm': 'SE',
    'Budapest': 'HU',
    'Warsaw': 'PL',
    'Dublin': 'IE',
    'Copenhagen': 'DK',
    'Athens': 'GR',
    'Edinburgh': 'GB',
    'Zurich': 'CH',
    'Oporto': 'PT',
    'Geneva': 'CH',
    'Krakow': 'PL',
    'Oslo': 'NO',
    'Helsinki': 'FI',
    'Bratislava': 'SK',
    'Luxembourg': 'LU',
    'Ljubljana': 'SI'
}

data['Country'] = data['City'].apply(lambda x: country[x])
data['Country'].nunique()
data['Country'].value_counts()
rest_in_country = data['Country'].value_counts()
data['Rest_in_country'] = data['Country'].apply(lambda x: rest_in_country[x])
# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na
#Add a column to indicate NaNs, if False NaNs are ignored.
#без этого параметра МАЕ лучше
data = pd.get_dummies(data, columns=[ 'City',])
#без параметра dummy_na МАЕ лучше
data = pd.get_dummies(data, columns=[ 'Country',])
data['Price Range'].value_counts()
#data['Price Category'] = data['Price Range'].apply(lambda x: x!= '$$ - $$$') 
#data['Price Category'] = data['Price Category'].astype(int)
#dummy_na=True убрала, так как уже все пропуски ранее заполнила
#data = pd.get_dummies(data, columns=[ 'Price Range',])
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
data['Price Range_Cat'] = labelencoder.fit_transform(data['Price Range'])
#заполняем пропуски
data['Reviews'] = data['Reviews'].fillna("['no_Reviews']")
#создаем новую колонку и туда кладем только данные, ктр содержат дату
data['date_of_Review'] = data['Reviews'].str.findall('\d+/\d+/\d+')
#создаем новую колонку, ктр содержит разницу между значениями колонки date_of_Review и превращаем ее в дни
data['day_between_Reviews'] = data.apply(lambda x: pd.to_datetime(x['date_of_Review']).max() - pd.to_datetime(x['date_of_Review']).min(), axis = 1).dt.days
#проверяем, что получилось
data.info()
data['day_between_Reviews'].fillna(0, inplace=True)
#сравниваем сроки между отзывами с годом
#data['Old']=data['day_between_Reviews'].apply(lambda x: float(x)>float(365)) 
#конвертируем буллево значение в 0/1
#data['Old'] = data['Old'].astype(int)
data['day_from_last_review'] = data.apply(lambda x: pd.datetime.now() - pd.to_datetime(x['date_of_Review']).max(), axis = 1).dt.days
data.info()
data['day_from_last_review'].fillna(0, inplace=True)
#копируем датафрейм
data_copy = data.copy()
#создаем новую колонку в копии датафрейма - в каждой строке новой колонки список из рассплитованных  значений
data_copy['Cuisine'] = data['Cuisine Style'].str.findall(r"'(\b.*?\b)'")
# 'раздвигаем' исходный датасет, чтобы внутри признака было только одно значение вида кухни, а не список
data_copy = data_copy.explode('Cuisine')
data_copy['Cuisine'].value_counts()
data['Number_of_cuisines'] = data['Cuisine Style'].apply(lambda x: len(x.split(',')))
aver_cuis = data['Cuisine Style'].apply(lambda x: len(x.split(','))).sum()/len(data)
#сравниваем кол-во кухонь в ресторане со средним значением
#data['More/less_aver_cuis']=data['Number_of_cuisines'].apply(lambda x: x > aver_cuis) 
#конвертируем буллево значение в 0/1
#data['More/less_aver_cuis'] = data['More/less_aver_cuis'].astype(int)
plt.rcParams['figure.figsize'] = (10,7)
df_train['Ranking'].hist(bins=100)
df_train['City'].value_counts(ascending=True).plot(kind='barh')
df_train['Ranking'][df_train['City'] =='London'].hist(bins=100)
# посмотрим на топ 10 городов
for x in (df_train['City'].value_counts())[0:10].index:
    df_train['Ranking'][df_train['City'] == x].hist(bins=100)
plt.show()
data['Comp_Ranking'] = data['Ranking'] / data['Rest_in_city']
data['Country'].value_counts(ascending=True).plot(kind='barh')
data['Ranking'][data['Country'] =='GB'].hist(bins=100)
# посмотрим на топ 10 стран
for x in (data['Country'].value_counts())[0:10].index:
    data['Ranking'][data['Country'] == x].hist(bins=100)
plt.show()
df_train['Rating'].value_counts(ascending=True).plot(kind='barh')
df_train['Ranking'][df_train['Rating'] == 5].hist(bins=100)
df_train['Ranking'][df_train['Rating'] < 4].hist(bins=100)
data.drop(['Cuisine Style', 'Price Range', 'Reviews', 'URL_TA', 'ID_TA', 'date_of_Review'], axis=1, inplace=True)
plt.rcParams['figure.figsize'] = (25,20)
sns.heatmap(data.drop(['sample'], axis=1).corr(),)
data.info()
data.sample(10)
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
y_pred = np.round(y_pred*2)/2
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
predict_submission = np.round(predict_submission*2)/2
predict_submission
sample_submission['Rating'] = predict_submission
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head(10)