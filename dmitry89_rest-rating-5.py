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
# фиксируем RANDOM_SEED, чтобы эксперименты были воспроизводимы!
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
# Далее заполняем пропуски 0 
data['Number of Reviews'].fillna(0, inplace=True)
data['Number of Reviews'].sample(10)
data.nunique(dropna=False)
data['Price Range'].value_counts()
#обработка 'Price Range'
price_dict = {'$': 10, '$$ - $$$': 100, '$$$$': 1000}
data['Price'] = data['Price Range'].map(price_dict)

#Пропуски заполняем медианным значением
data['Price'] = data['Price'].fillna(data['Price'].median())

#Убираем столбец 'Price Range'
data.drop(['Price Range',], axis = 1, inplace=True)

## для One-Hot Encoding в pandas есть готовая функция - get_dummies. 
## Особенно радует параметр dummy_na. Кодируем уровень цен в ресторанах 'Price'
data = pd.get_dummies(data, columns=[ 'Price',], dummy_na=True)
data.sample(5)
# Избавимся от NaN в признаке Cuisine Style местной кухней Local
data['Cuisine Style'] = data['Cuisine Style'].fillna('Local')

# Избавимся от NaN в признаке Number of Reviews. Заполним значения NaN -- 0
data['Number of Reviews'] = data['Number of Reviews'].fillna(0)

# Преобразуем данные признака Cuisine Style из str в list
# Функция, преобразующая строковые данные и строки Cuisine Style в данные типа список
def str_to_list(x):
    y = []
    a = ['Local']
    if x != 'Local':
        x = str(x[2:-2]).split('\', \'')
        for i in x:
            y.append(i)
        return y
    else:
        return a

data['Cuisine Style'] = data['Cuisine Style'].apply(str_to_list)

# Считаем кол-во кухонь в каждом ресторане
def count_of_cuisine(data):
    if type(data) == list:
        return len(data)
    
data['Count cuisines'] = data['Cuisine Style'].apply(count_of_cuisine)

data.sample(5)
# проверяем города на принадлежность столице и потом кодируем по этому признаку
capitals = {'London': 1, 'Paris': 1, 'Madrid': 1, 'Barcelona': 0, 'Berlin': 1, 'Milan': 0, 'Rome': 1, 'Prague': 1, 'Lisbon': 1, 
    'Vienna': 1, 'Amsterdam': 1, 'Brussels': 1, 'Hamburg': 0, 'Munich': 0, 'Lyon': 0, 'Stockholm': 1, 'Budapest': 1, 'Warsaw': 1,
    'Dublin': 1, 'Copenhagen': 1, 'Athens': 1, 'Edinburgh': 1, 'Zurich': 0, 'Oporto': 0, 'Geneva': 0, 'Krakow': 0, 'Oslo': 1,
    'Helsinki': 1, 'Bratislava': 1, 'Luxembourg': 1, 'Ljubljana': 1}

def dict_capital(city):
    return capitals[city]
data['Capital'] = data.City.apply(dict_capital)

data = pd.get_dummies(data, columns=[ 'Capital',], dummy_na=True)

    
# добавляем столбец население городов

population = {'Paris': 2190327, 'Stockholm': 961609, 'London': 8908081, 'Berlin': 3644826, 'Munich': 1456039, 'Oporto': 237591,
                  'Milan': 1378689,'Bratislava': 432864, 'Vienna': 1821582, 'Rome': 4355725, 'Barcelona': 1620343, 'Madrid': 3223334,
                  'Dublin': 1173179,'Brussels': 179277, 'Zurich': 428737, 'Warsaw': 1758143, 'Budapest': 1752286, 'Copenhagen': 615993,
                  'Amsterdam': 857713,'Lyon': 506615, 'Hamburg': 1841179,'Lisbon': 505526, 'Prague': 1301132, 'Oslo': 673469,
                  'Helsinki': 643272,'Edinburgh': 488100,'Geneva': 200548, 'Ljubljana': 284355,'Athens': 664046, 'Luxembourg': 115227,
                  'Krakow': 769498}

def dict_population(city):
    return population[city]
 

data['Population'] = data.City.apply(dict_population)
data.sample(5)
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
df_output = data.copy()
def preproc_data(df_input):
    '''includes several functions to pre-process the predictor data.'''
    
    df_output = df_input.copy()
    
    # ################### 1. Предобработка ############################################################## 
    # убираем ненужные для модели признаки
    df_output.drop(['Restaurant_id','ID_TA', 'URL_TA'], axis = 1, inplace=True)
    
    
    # ################### 2. NAN ############################################################## 
    # Далее заполняем пропуски 0
    df_output['Number of Reviews'].fillna(0, inplace=True)
    
    
    # ################### 3. Encoding ############################################################## 
    #обработка 'Price Range'
    price_dict = {'$': 10, '$$ - $$$': 100, '$$$$': 1000}
    df_output['Price'] = df_output['Price Range'].map(price_dict)

    #Пропуски заполняем медианным значением
    df_output['Price'] = df_output['Price'].fillna(df_output['Price'].median())

    #Убираем столбец 'Price Range'
    df_output.drop(['Price Range',], axis = 1, inplace=True)

    ## для One-Hot Encoding в pandas есть готовая функция - get_dummies. 
    ## Особенно радует параметр dummy_na. Кодируем уровень цен в ресторанах 'Price'
    df_output = pd.get_dummies(df_output, columns=[ 'Price',], dummy_na=True)

    # ################### 4. Feature Engineering ####################################################
    
    # Избавимся от NaN в признаке Cuisine Style местной кухней Local
    df_output['Cuisine Style'] = df_output['Cuisine Style'].fillna('Local')

    # Преобразуем данные признака Cuisine Style из str в list
    # Функция, преобразующая строковые данные и строки Cuisine Style в данные типа список
    def str_to_list(x):
        y = []
        a = ['Local']
        if x != 'Local':
            x = str(x[2:-2]).split('\', \'')
            for i in x:
                y.append(i)
            return y
        else:
            return a

    df_output['Cuisine Style'] = df_output['Cuisine Style'].apply(str_to_list)

    # Считаем кол-во кухонь в каждом ресторане
    def count_of_cuisine(data):
        if type(data) == list:
            return len(data)
    
    df_output['Count cuisines'] = df_output['Cuisine Style'].apply(count_of_cuisine)

    # проверяем города на принадлежность столице и потом кодируем по этому признаку
    capitals = {'London': 1, 'Paris': 1, 'Madrid': 1, 'Barcelona': 0, 'Berlin': 1, 'Milan': 0, 'Rome': 1, 'Prague': 1, 'Lisbon': 1, 
    'Vienna': 1, 'Amsterdam': 1, 'Brussels': 1, 'Hamburg': 0, 'Munich': 0, 'Lyon': 0, 'Stockholm': 1, 'Budapest': 1, 'Warsaw': 1,
    'Dublin': 1, 'Copenhagen': 1, 'Athens': 1, 'Edinburgh': 1, 'Zurich': 0, 'Oporto': 0, 'Geneva': 0, 'Krakow': 0, 'Oslo': 1,
    'Helsinki': 1, 'Bratislava': 1, 'Luxembourg': 1, 'Ljubljana': 1}

    def dict_capital(city):
        return capitals[city]
    df_output['Capital'] = df_output.City.apply(dict_capital)

    df_output = pd.get_dummies(df_output, columns=[ 'Capital',], dummy_na=True)

    
    # добавляем столбец население городов

    population = {'Paris': 2190327, 'Stockholm': 961609, 'London': 8908081, 'Berlin': 3644826, 'Munich': 1456039, 'Oporto': 237591,
                  'Milan': 1378689,'Bratislava': 432864, 'Vienna': 1821582, 'Rome': 4355725, 'Barcelona': 1620343, 'Madrid': 3223334,
                  'Dublin': 1173179,'Brussels': 179277, 'Zurich': 428737, 'Warsaw': 1758143, 'Budapest': 1752286, 'Copenhagen': 615993,
                  'Amsterdam': 857713,'Lyon': 506615, 'Hamburg': 1841179,'Lisbon': 505526, 'Prague': 1301132, 'Oslo': 673469,
                  'Helsinki': 643272,'Edinburgh': 488100,'Geneva': 200548, 'Ljubljana': 284355,'Athens': 664046, 'Luxembourg': 115227,
                  'Krakow': 769498}

    def dict_population(city):
        return population[city]
 

    df_output['Population'] = df_output.City.apply(dict_population)
    
    # ################### 5. Clean #################################################### 
    # убираем признаки которые еще не успели обработать, 
    # модель на признаках с dtypes "object" обучаться не будет, просто выберем их и удалим
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
sample_submission.to_csv('submission5.csv', index=False)
sample_submission.head(10)