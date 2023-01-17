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
df = data # переименуем дата фрейм как в скиллфактори
df.head(5)
df.columns # рассмотрим названия столбцов
df.info()
df['Restaurant_id'].value_counts()
df['Restaurant_id'].describe()
df['Restaurant_id_new'] = df['Restaurant_id'].apply(lambda x: x[3:]) # уберем первые буквы, для использования в модели 

df['Restaurant_id_new'] = df['Restaurant_id_new'].astype(float)

df['Restaurant_id_new']
df['Cuisine Style'].value_counts()
df['Cuisine Style'].describe()
df['Cuisine Style'] = df['Cuisine Style'].fillna('No_answer') # заменяем пустные значения на "без ответа" 
def count_cuisine(x): # фунцкия считающая количество видов кухонь в ресторане

    x = x.replace('[','')

    x = x.replace(']', '')

    x = x.strip()

    x = [style.strip() for style in x.split(',')]

    x = [style for style in x if len(style) > 0]

    return len(x)
df['number_cuisine'] = df['Cuisine Style'].apply(count_cuisine)

df['number_cuisine'].head(10)
df['number_cuisine'].hist()
df['number_cuisine'].describe()
df['number_cuisine'] = df['number_cuisine'].replace('No_answer', '2')

df['number_cuisine'].describe()
median = df['number_cuisine'].median()

IQR = df['number_cuisine'].quantile(0.75) - df['number_cuisine'].quantile(0.25)

perc25 = df['number_cuisine'].quantile(0.25)

perc75 = df['number_cuisine'].quantile(0.75)

print('25-й перцентиль: {},'.format(perc25), '75-й перцентиль: {},'.format(perc75)

      , "IQR: {}, ".format(IQR),"Границы выбросов: [{f}, {l}].".format(f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR))
(df['number_cuisine']>8.5).sum()
df['Price Range'].value_counts()
df['Price Range'] = df['Price Range'].fillna('$$ - $$$') # заменяем пустные значения
def price(x): # функция заменяющая обозначения градацией цен

        x = x.replace('$$ - $$$', '2')

        x = x.replace('$$$$', '3')

        x = x.replace('$', '1')

        return x
df['price_new'] = df['Price Range'].apply(price)

df['price_new'].value_counts()
df['price_new'] = df['price_new'].astype(int)

df['price_new'].hist()
df['Reviews'].value_counts()
def date_Reviews(x): # вытащим даты отзывов в отдельные столбцы и посчитаем разницу

    if x == '[[], []]':

        return []

    else:

        x = x.replace(']]', '')

        x = x.replace("'", '')

        x = x.split('], [')[1]

        x = x.split(', ')

        return x
df['Reviews'] = df['Reviews'].apply(lambda x: str([[], []]) if type(x) == float else x) 
df['Dates of Reviews'] = df['Reviews'].apply(date_Reviews) # создадим 2 столбца: первый и последний отзыв
df[['Date_last', 'Date_first']] = pd.DataFrame(df['Dates of Reviews'].tolist()) 
df['Date_last'].value_counts()
df['Date_first'].value_counts()
df['Date_last'] = df['Date_last'].fillna('01/07/2018') # заменим пустные значения самыми часто встречаемыми

df['Date_first'] = df['Date_first'].fillna('01/03/2018')
df['Date_last'] = pd.to_datetime(df['Date_last'])

df['Date_first'] = pd.to_datetime(df['Date_first'])
df['Timedelta'] = df.apply(lambda x: x['Date_last'] - x['Date_first'], axis = 1)

df['Timedelta'].head(10)
def zero(x):

    x = x.replace('-', '')

    return x
df['Timedelta'] = df['Timedelta'].astype(str)
df['Timedelta_new'] = df['Timedelta'].apply(zero)
df['Timedelta_new'].head(10)
df['Timedelta_new'] = df['Timedelta_new'].apply(lambda x: x.replace('00:00:00.000000000', ''))  

df['Timedelta_new'] = df['Timedelta_new'].apply(lambda x: x.replace(' days', '')) 

df['Timedelta_new'] = df['Timedelta_new'].apply(lambda x: x.replace(' +', '')) 

df['Timedelta_new'] = df['Timedelta_new'].apply(lambda x: x.replace(' ', ''))

df['Timedelta_new'].head(10)
df['Timedelta_new'] = df['Timedelta_new'].astype(float) 

df['Timedelta_new'].hist()

df['Timedelta_new'].describe()
df['Reviews'] = df['Reviews'].apply(lambda x: x.lower()) # делаем регистр одинаковым для удобства поиска ключевых слов
words_good = ['good','great','best','wonderful','nice','excellent'] # 3-положительный, 2-нейтральный, 1-отрицательный отзыв

words_bad = ['disappointing','overpriced','bad','horrible','grumpiest','awful']

def find_word(cell):

    if cell is not None:

        for word in words_good:

            if word in cell:

                return 3

        for word in words_bad:

            if word in cell:

                return 1

    return 2 # если пустные, то ставим 2. считаем что отзыв нейтральный
df['reviews_sense'] = df['Reviews'].apply(find_word)

df['reviews_sense'].value_counts()
df['City'].value_counts()
city_all = set() # создадим справочник городов

for x in df['City']:

        city_all.add(x) 

print(city_all)
def find_city(cell): # функция проставления наличие или отсутствия элемента в ячейке

    if x in cell:

        return 1

    return 0
for x in city_all:

    df[x] = df['City'].apply(find_city) # создали столбцы с названиями городов, в которых находятся рестораны
df.head(2)
Сity_pop_dict = {'London' : 8908, 'Paris' : 2206, 'Madrid' : 3223, 'Barcelona' : 1620, 

                        'Berlin' : 6010, 'Milan' : 1366, 'Rome' : 2872, 'Prague' : 1308, 

                        'Lisbon' : 506, 'Vienna' : 1888, 'Amsterdam' : 860, 'Brussels' : 179, 

                        'Hamburg' : 1841, 'Munich' : 1457, 'Lyon' : 506, 'Stockholm' : 961, 

                        'Budapest' : 1752, 'Warsaw' : 1764, 'Dublin' : 553, 

                        'Copenhagen' : 616, 'Athens' : 665, 'Edinburgh' : 513, 

                        'Zurich' : 415, 'Oporto' : 240, 'Geneva' : 201, 'Krakow' : 769, 

                        'Oslo' : 681, 'Helsinki' : 643, 'Bratislava' : 426, 

                        'Luxembourg' : 119, 'Ljubljana' : 284}
df['Сity_pop'] = df.apply(lambda row: Сity_pop_dict[row['City']], axis = 1)
df['Сity_pop'].hist()
df['Number of Reviews'].head(10)
df['Number of Reviews'].value_counts()
df['Number of Reviews'].hist()

df['Number of Reviews'].describe()
Reviews_in_city_median = df.groupby(['City'])['Number of Reviews'].median().sort_values(ascending=False)

Reviews_in_city_median
#df['Number of Reviews'] = df['Number of Reviews'].fillna(df['City'].apply(lambda x: Reviews_in_city_median[x]))
df['Number of Reviews'] = df['Number of Reviews'].fillna(0)
df['Number of Reviews'].head(10)
df['Number of Reviews'] = df['Number of Reviews'].astype(int)

df['Number of Reviews'].describe()
median = df['Number of Reviews'].median()

IQR = df['Number of Reviews'].quantile(0.75) - df['Number of Reviews'].quantile(0.25)

perc25 = df['Number of Reviews'].quantile(0.25)

perc75 = df['Number of Reviews'].quantile(0.75)

print('25-й перцентиль: {},'.format(perc25), '75-й перцентиль: {},'.format(perc75)

      , "IQR: {}, ".format(IQR),"Границы выбросов: [{f}, {l}].".format(f = perc25 - 1.5*IQR, l = perc75 + 1.5*IQR))
(df['Number of Reviews'] > 300).sum()
Reviews_in_city_count = df.groupby(['City'])['Number of Reviews'].sum().sort_values(ascending=False)

Reviews_in_city_count
df['Reviews_in_city'] = df['City'].apply(lambda x: Reviews_in_city_count[x])
df['Reviews_in_city'].head()
df['Relative_rank_Reviews'] = df['Ranking'] / df['Reviews_in_city']
df['Relative_rank_Reviews']
df['ID_TA'].head()
df['ID_TA'] = df['ID_TA'].apply(lambda x: int(x[1:]))
df['ID_TA'].head()
df['Ranking'].hist()

df['Ranking'].describe()
for x in (df['City'].value_counts())[0:10].index:

    df['Ranking'][df['City'] == x].hist(bins=100)

plt.show()
# чем больше город, тем больше Ranking

# отнормируем критерий Ranking по городам City

mean_Ranking_in_City = df.groupby(['City'])['Ranking'].mean() # средний ранг в городе

count_Restorant_in_City = df['City'].value_counts(ascending=False) # число ресторанов в городе

df['mean_Ranking_in_City'] = df['City'].apply(lambda x: mean_Ranking_in_City[x])

df['count_Restorant_in_City'] = df['City'].apply(lambda x: count_Restorant_in_City[x])

df['norm_Ranking_in_Rest_in_City'] = (df['Ranking'] - df['mean_Ranking_in_City']) / df['count_Restorant_in_City']
df['URL_TA'].head()
plt.rcParams['figure.figsize'] = (15, 10)

sns.heatmap(df[['mean_Ranking_in_City','count_Restorant_in_City','Restaurant_id', 'Ranking', 'Rating', 'Number of Reviews', 'ID_TA',

       'Restaurant_id_new', 'number_cuisine', 'price_new', 'Timedelta_new',

       'reviews_sense', 'Сity_pop', 'Reviews_in_city',

       'Relative_rank_Reviews', 'norm_Ranking_in_Rest_in_City']].corr(), annot = True, fmt='.1g')
df = df.drop(['mean_Ranking_in_City','count_Restorant_in_City','Restaurant_id','Timedelta','City','Cuisine Style','Price Range','Reviews','URL_TA','Dates of Reviews','Date_last','Date_first'], axis = 1)
df.info()
# на всякий случай, заново подгружаем данные

#df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

#df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')

#df_train['sample'] = 1 # помечаем где у нас трейн

#df_test['sample'] = 0 # помечаем где у нас тест

#df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями









#data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем

#data.info()
#def preproc_data(df_input):#

#    df_output = df_input.copy()

#    df_output.drop(['Restaurant_id'], axis = 1, inplace=True)

#    object_columns = [s for s in df_output.columns if df_output[s].dtypes == 'object']

#    df_output.drop(object_columns, axis = 1, inplace=True)

#    

#    return df_output
#def preproc_data(df_input):

#    '''includes several functions to pre-process the predictor data.'''

#    

#    df_output = df_input.copy()

#    

#    # ################### 1. Предобработка ############################################################## 

#    # убираем не нужные для модели признаки

#    df_output.drop(['Restaurant_id','ID_TA',], axis = 1, inplace=True)

#    

#    

#    # ################### 2. NAN ############################################################## 

#    # Далее заполняем пропуски, вы можете попробовать заполнением средним или средним по городу и тд...

#    df_output['Number of Reviews'].fillna(0, inplace=True)

#    # тут ваш код по обработке NAN

#    # ....

#    

#    

#    # ################### 3. Encoding ############################################################## 

#    # для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

#    df_output = pd.get_dummies(df_output, columns=[ 'City',], dummy_na=True)

#    # тут ваш код не Encoding фитчей

#    # ....

#    

#    

#    # ################### 4. Feature Engineering ####################################################

#    # тут ваш код не генерацию новых фитчей

#    # ....

#    

#    

#    # ################### 5. Clean #################################################### 

#    # убираем признаки которые еще не успели обработать, 

#    # модель на признаках с dtypes "object" обучаться не будет, просто выберим их и удалим

#    object_columns = [s for s in df_output.columns if df_output[s].dtypes == 'object']

#    df_output.drop(object_columns, axis = 1, inplace=True)

#    

#    return df_output
#df_preproc = preproc_data(df)

#df_preproc.sample(10)
#df_preproc.info()#
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
test_data.sample(10)
test_data = test_data.drop(['Rating'], axis=1)
sample_submission
predict_submission = model.predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)