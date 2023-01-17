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
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



df = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
df.info()
#заполнение пропусков в колонке "кол-во отзывов"

mean_num_of_rev = round(df['Number of Reviews'].mean(),1)

df['Number of Reviews']= df['Number of Reviews'].fillna(mean_num_of_rev)
df.info()
# Добавляем столбец признак города столицы



city_list = ['London', 'Paris', 'Stockholm', 'Madrid', 'Berlin', 'Rome', 'Prague', 'Lisbon', 'Vienna', 'Amsterdam', 'Budapest', 'Warsaw', 'Dublin', 'Copenhagen', 

             'Athens', 'Edinburgh', 'Oslo', 'Helsinki', 'Bratislava', 'Ljubljana', 'Brussels', 'Luxembourg']



df['City_capital'] = df['City'].apply(lambda x: 1 if x in city_list else 0 )





# Добавим новый признаки с кол-во людей в городах, кол-ве ресторанов и кол-ве ресторанов на душу населения



cities_population = {

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



res_count = {

    'Paris': 17593,

    'Stockholm': 3131,

    'London': 22366,

    'Berlin': 8110, 

    'Munich': 3367,

    'Oporto': 2060, 

    'Milan': 7940,

    'Bratislava': 1331,

    'Vienna': 4387, 

    'Rome': 12086,

    'Barcelona': 10086,

    'Madrid': 11562,

    'Dublin': 2706,

    'Brussels': 3703,

    'Zurich': 1901,

    'Warsaw': 3210,

    'Budapest': 3445, 

    'Copenhagen': 2637,

    'Amsterdam': 4189,

    'Lyon': 2833,

    'Hamburg': 3501, 

    'Lisbon': 4985,

    'Prague': 5850,

    'Oslo': 1441, 

    'Helsinki': 1661,

    'Edinburgh': 2248,

    'Geneva': 1753,

    'Ljubljana': 647,

    'Athens': 2814,

    'Luxembourg': 759,

    'Krakow': 1832       

}



df['Population'] = df['City'].map(cities_population)







df['Restaurants Count'] = df['City'].map(res_count)





df['restaraunts_per_people'] = df['Restaurants Count']/ df['Population']
df.info()
# Добавляем столбец количество представленных стилей кухонь в ресторане



df['Cuisine_count'] =df['Cuisine Style'].str[2:-2].str.split("', '").fillna('1').str.len()



# Добавляем столбец показатель разницы дней отзывов ресторане



df['Date_1'] = df['Reviews'].str.findall(r'\d+\W\d+\W\d+').str.get(0)

df['Date_1'] =pd.to_datetime(df['Date_1'], errors='coerce')

df['Date_2'] = df['Reviews'].str.findall(r'\d+\W\d+\W\d+').str.get(1)

df['Date_2'] =pd.to_datetime(df['Date_2'], errors='coerce')

df['Data_difference'] = (df['Date_1'] - df['Date_2'])

df['Data_difference'] = df['Data_difference'].astype('timedelta64[D]').fillna(0)
df.info()
# Преобразуем признак Price Range в числовой



df['Price Range']=df['Price Range'].fillna('$$ - $$$')



price = {

    '$': 1,

    '$$ - $$$': 2,

    '$$$$': 3

}



df['Price Range'] = df['Price Range'].map(price)

df.info()
#### Создаем думис с городами





df = pd.get_dummies(df, columns=[ 'City',], dummy_na=True)
# Удалим все столбцы формата object

df.drop(['Restaurant_id', 'Reviews', 'URL_TA', 'ID_TA', 'Date_1', 'Date_2', 'Cuisine Style'], axis='columns', inplace = True)

#df = df.dropna()

df.info(verbose = True, null_counts = True)
df.info()
# Теперь выделим тестовую часть

train_data = df.query('sample == 1').drop(['sample'], axis=1)

test_data = df.query('sample == 0').drop(['sample'], axis=1)



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
test_data.tail(10)
test_data = test_data.drop(['Rating'], axis=1)

len(test_data)
test_data.tail(10)

test_data.info()
sample_submission
predict_submission = model.predict(test_data)
predict_submission

len(predict_submission)

len (sample_submission)

sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)