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



import re



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
# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
data.head(5)
data.sample(5)
data['Price Range'].value_counts()
# Ваша обработка 'Price Range'
# тут ваш код на обработку других признаков

# .....
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

    df_output = pd.get_dummies(df_output, columns=[ 'City',], dummy_na=True)

    # тут ваш код не Encoding фитчей

    # ....

    

    

    # ################### 4. Feature Engineering ####################################################

    # тут ваш код не генерацию новых фитчей

    # ....

    

    

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
# Functions:



# Размер сети ресторанов:

def network_large(idx, counts):

    return counts.loc[idx]



# Проверка наличия определенной кухни в ячейке 'Cuisine Style' отдельного ресторана

def find_cuisine(cell):

    if cuisine in cell.replace(' ',''): # т.к. при создании списка кухонь пробелы были убраны

        return 1

    return 0



# Перевод ценовой категории в число

def change_price_range(line):

    if line == '$': return 1

    if line == '$$ - $$$': return 2

    if line == '$$$$': return 3



# Количество кухонь

def fill_cuisine_style(line):

    if str(line)!='nan': return len(line.split(','))



# Год последнего отзыва

def year_last_review(cell):

    if cell=='': return np.nan

    a = re.findall(r'[0-9]+/[0-9]+/[0-9]+', cell)

    if a==list(): return np.nan

    return pd.to_datetime(a[0]).year



# Дата последнего отзыва

def latest_date(line):

    # -1 используется из-за массива размером 3 в id=28330

    return pd.to_datetime(line.split(' [')[-1].replace(']]','').replace("'",'').split(', ')).max()
# Create dataframe to change:

df = data.copy()
# MAE: 0.209925

0.21240125-0.209925
restaurants = df['City'].value_counts()

restaurants
# Попробуем скорректировать значение с учетом кол-ва ресторанов в городах.
df['Corrected_Ranking'] = (df['Ranking'])/(df['City'].apply(lambda x: restaurants.loc[x]))*1000

#*1000 для обозначения более четких границ. Улучшило результат, но на незначительное значение.
df['Corrected_Ranking'].sample(5)
# -ухудшает результат
df['City_Restaurants'] = df['City'].apply(lambda x: restaurants.loc[x])
df['City_Restaurants'].sample(5)
# Любые попытки использования ухудшают результат по MAE
df['Restaurant_id'].isna().sum()
counts = data['Restaurant_id'].value_counts()
# Распределение объема сетей ресторанов:

counts.value_counts()
counts.value_counts().plot(kind='barh')
counts.value_counts().plot(kind='area')
df['Network_Large'] = df['Restaurant_id'].apply(lambda x: network_large(x, counts))
df['Network_Large'].isna().sum()
df.sample(5)
# Попытка разделить сети на крупные и мелкие:

#df['Network_Large'] = df['Network_Large'].apply(lambda x: 1 if x>=6 else 0) - сильно ухудшает результат
# Попытка разделить рестораны на одиночные и сетевые:

# df['Network_Large'] = df['Network_Large'].apply(lambda x: 1 if x>=2 else 0) - результат ухудшился
# Проверим корреляцию, убрав часть данных с неизвестным рейтингом:

df[['Network_Large', 'Rating']][10000:].corr()
#0.207421875

print('Без количества кухонь:',0.209925-0.207421875) #Без кол-ва кухонь

print('С количетсвом кухонь:', 0.209925-0.2079575)
# Список кухонь:

cuisines = set()

for cuisines_list in df['Cuisine Style'].fillna('Other'):

    for cuisine in cuisines_list.replace('[','').replace(']','').replace(' ','').replace("'",'').split(','):

        cuisines.add(cuisine)
df['Cuisine Style'].fillna('Other').apply(find_cuisine)
hst = pd.DataFrame()

for cuisine in cuisines:

    hst[cuisine] = df['Cuisine Style'].fillna('Other').apply(find_cuisine).value_counts()
hst.iloc[1].plot(kind='barh', figsize=(8,25))
#0.20712937499999998 - результат улучшен крайне незначительно. При замене NaN значений общей категорией результат ухудшался.

print(0.207421875-0.20712937499999998)
df['Price Range'] = df['Price Range'].apply(change_price_range)
df.sample(5)
df['Price Range'].hist()
df['Price Range'].isna().sum()
# датафрейм для корректировки NaN значений методом ML

cor_df = df[['Corrected_Ranking', 'Number of Reviews', 'City', 'Cuisine Style', 'Price Range']]



#City

cor_df = pd.get_dummies(cor_df, columns=['City'], dummy_na=True)



#Cuisine Style

for cuisine in cuisines:

    cor_df[cuisine] = df['Cuisine Style'].fillna('Other').apply(find_cuisine)

cor_df.drop('Cuisine Style', axis=1, inplace=True)



# Сохраним датафрейм для обработки после построения модели:

saved_df = cor_df.copy()



# Убирает только NaN значения Price Range, т.к. именно для их нахождения строится модель

cor_df.dropna(inplace=True)



cor_df.sample(5)
y = cor_df['Price Range'].values            # наш таргет

X = cor_df.drop(['Price Range'], axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)



X.shape, X_train.shape, X_test.shape
model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)



# Обучаем модель на тестовом наборе данных

model.fit(X_train, y_train)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

y_pred = model.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# После получения метрики проведем тренировку на большем количестве данных:

model.fit(X, y)
# Сформируем данные для предсказания по обученной модели:

X = saved_df[saved_df['Price Range'].isna()].drop('Price Range', axis=1)
X['Number of Reviews'].fillna(0, inplace=True)
y = model.predict(X)
y
y.round()
# Сохраним NaN значения колонки Price Range

renewed_price_range = pd.DataFrame(y.round()).set_index(saved_df[saved_df['Price Range'].isna()]['Price Range'].index)[0]
renewed_price_range
saved_df['Price Range'].fillna(renewed_price_range).isna().sum()
#df['Corrected_Number_of_Reviews'] = (df['Number of Reviews']/(df['City'].apply(lambda x: restaurants.loc[x])))*1000

# - сильно ухудшило результат, судя по всему кол-во отзывов не зависит от кол-ва ресторанов
#df['Corrected_Number_of_Reviews']
# Перед отправкой в модель заполним NaN значения:

df['Number of Reviews'].fillna(0, inplace=True)
#0.20491437499999998

0.20712937499999998-0.20491437499999998
df['Last_Review'] = df['Reviews'].fillna('').apply(lambda x:year_last_review(x))
df['Last_Review'].sample(5)
preproc = df[['Corrected_Ranking', 'Number of Reviews', 'Rating', 'City', 'sample', 'Cuisine Style', 'Price Range', 'Last_Review', 'URL_TA']]



#City

preproc = pd.get_dummies(preproc, columns=['City'], dummy_na=True)



#Network Large:

# Числовые данные результат ухудшают. Попробуем через get_dummies:

#preproc = pd.get_dummies(preproc, columns=['Network_Large'], dummy_na=True) # - результат ухудшился



#Cuisine Style

for cuisine in cuisines:

    preproc[cuisine] = df['Cuisine Style'].fillna('Other').apply(find_cuisine)

preproc.drop('Cuisine Style', axis=1, inplace=True)

    

#Number of cuisines:  - делает результат хуже  

#preproc['Cuisine Style'] = df['Cuisine Style'].apply(lambda x: fill_cuisine_style(x)) 

#preproc['Cuisine Style'].fillna(1, inplace=True)





# Price Range:

preproc['Price Range'].fillna(renewed_price_range, inplace=True)

preproc = pd.get_dummies(preproc, columns=['Price Range'])

# замена через get_dummies дала лучший результат.



#URL_TA: - длина адреса, имеет зависимость с длиной названия, результат немного улучшился.

#preproc['URL_TA'] = preproc['URL_TA'].apply(lambda x: len(x)) - ухудшает результат



#Last Review

preproc['Last_Review'].fillna(0, inplace=True)

#preproc = pd.get_dummies(preproc, columns=['Last_Review'], dummy_na=True) -результат чуть хуже чем с использованием числа





preproc
train_data = preproc.query('sample == 1').drop(['sample'], axis=1)

test_data = preproc.query('sample == 0').drop(['sample'], axis=1)



y = train_data.Rating.values            # наш таргет

X = train_data.drop(['Rating'], axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)



test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape
model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)



# Обучаем модель на тестовом наборе данных

model.fit(X_train, y_train)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

y_pred = model.predict(X_test)
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
test_data.sample(10)
test_data = test_data.drop(['Rating'], axis=1)
sample_submission
predict_submission = model.predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)