# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline

from itertools import combinations
from scipy.stats import ttest_ind
import statsmodels.api as sm
import scipy.stats as sst
from collections import Counter
import re

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Загружаем специальный удобный инструмент для разделения датасета:
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
from sklearn import metrics # инструменты для оценки точности модели

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

filenames_list = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        filenames_list.append(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import os
print(os.listdir("/kaggle/working"))
cities_pop_filename = '/kaggle/input/world-cities/worldcities.csv'
cities_pop_filename
# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!
RANDOM_SEED = 42
# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:
!pip freeze > requirements.txt
# Завернем модель в функцию для того, чтобы было удобнее вызывать

def model_func(df_preproc):
    # выделим тестовую часть
    train_data = df_preproc.query('sample == 1').drop(['sample'], axis=1)
    test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1)

    y = train_data.Rating.values            # наш таргет
    X = train_data.drop(['Rating'], axis=1)
    
    RANDOM_SEED = 42
    
    # Воспользуемся специальной функцие train_test_split для разбивки тестовых данных
    # выделим 20% данных на валидацию (параметр test_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    # Создаём модель (НАСТРОЙКИ НЕ ТРОГАЕМ)
    model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)
    
    # Обучаем модель на тестовом наборе данных
    model.fit(X_train, y_train)

    # Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
    # Предсказанные значения записываем в переменную y_pred
    y_pred = model.predict(X_test)
    
    result = metrics.mean_absolute_error(y_test, y_pred)
    
    # в RandomForestRegressor есть возможность вывести самые важные признаки для модели
    plt.rcParams['figure.figsize'] = (10,10)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(15).plot(kind='barh')
    
    plt.show
    
    return result
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'
df_train = pd.read_csv(DATA_DIR+'/main_task.csv')
df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')
sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')

# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет
df_train['sample'] = 1 # помечаем где у нас трейн
df_test['sample'] = 0 # помечаем где у нас тест
df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями

data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
data.info()
sample_submission
# обработаем столбец Number of Reviews
number_rew_nan = pd.isna(data['Number of Reviews']).astype('uint8')

number_rew_nan.name = 'number_rew_nan'
number_rew_nan.value_counts()
# Далее заполняем пропуски нулем

#mean = data['Number of Reviews'].mean()
number_rew = data['Number of Reviews'].fillna(0)

number_rew.name = 'number_rew'
number_rew.sample(5)
# проверим модель на имеющихся числовых признаках без обработки

df_preproc = pd.concat([data.loc[:,['Rating', 'sample','Ranking']], number_rew, number_rew_nan], axis = 1)
model_func(df_preproc)
data.isna().sum()
data['Number of Reviews'].fillna(0, inplace = True)
# посмотрим на топ 10 городов
for x in (data['City'].value_counts())[0:10].index:
    data['Ranking'][data['City'] == x].hist(bins=100)
plt.show()
data[data.City == 'London'].describe()
data[(data.City == 'London') & (data.Rating == 5)].loc[:,['Number of Reviews', 'Ranking']].\
sort_values('Ranking', ascending = False).iloc[:10,]
plt.figure(figsize = (5,5))
sns.jointplot(data = data[(data.City == 'London') & (data.Rating > 0)], x = 'Ranking', y = 'Rating', kind = 'kde')
# создадим признак, который на основании Ranking вычисляет значение Rating. Зависимость линейная
# f(1) = 5 и f(n) = 1 при f(x) = k*x + b

max_rank_by_city = data.groupby(['City']).max().Ranking

def true_rating(row):
    return round(5 - 4*(1 - row['Ranking'])/(1 - max_rank_by_city[row['City']]),1)

data['rating_by_rank'] = data.apply(true_rating, axis = 1)
data['rating_by_rank'].sort_values()
fig, ax = plt.subplots(1,1, figsize = (10,5))
ax = sns.heatmap(data.loc[data.City == 'London',['Rating', 'rating_by_rank']].corr(),annot = True, cmap = 'coolwarm')
data['rating_by_rank'].hist()
data[(data.City == 'London') & (data.Rating == 5)].loc[:,['Number of Reviews', 'Ranking', 'rating_by_rank']].\
sort_values('Ranking', ascending = False).iloc[:20,]
df_preproc = pd.concat([data.loc[:,['Number of Reviews','rating_by_rank']], data.loc[:,['Rating', 'sample']]], axis = 1)
model_func(df_preproc)
rating_by_rank = data['rating_by_rank']
data[data.City == 'London'].loc[:,['Number of Reviews', 'Ranking', 'rating_by_rank','Rating']].\
sort_values('rating_by_rank', ascending = True).iloc[:20,]
# создадим еще один признак на основе Ranking, проверим, какой работает лучше
# просто поделим Ranking на макс. значение в городе

def norm_rank_funk(row):
    return round(row['Ranking']*100/max_rank_by_city[row['City']],5)

norm_rank = data.apply(norm_rank_funk, axis = 1)
norm_rank.name = 'norm_rank'
norm_rank.sort_values()
norm_rank.hist()
plt.figure(figsize = (5,5))
plt.boxplot(norm_rank, vert = False)
df_preproc = pd.concat([data.loc[:,['Rating', 'sample']], number_rew, number_rew_nan, norm_rank], axis = 1)
model_func(df_preproc)
# среднее количество отзывов в городе
mean_rews_by_city = round((data.groupby(['City']).sum()['Number of Reviews']
                           /data.groupby(['City']).max()['Ranking']),2)

mean_rews = data.City.apply(lambda x: mean_rews_by_city[x])
mean_rews.name = 'mean_rews'
mean_rews.sample(5)
# два признака для количество ресторанов в городе, один на основе макс Ranking, другой просто по количеству записей в городе

max_rank = data.City.apply(lambda x: max_rank_by_city[x])
max_rank.name = 'max_rank'

places_counts_by_sity = data.groupby(['City']).count().Ranking
places_counts = data.City.apply(lambda x: places_counts_by_sity[x])
places_counts.name = 'places_counts'

pd.concat([max_rank, places_counts], axis = 1).sample(10)
df_preproc = pd.concat([data.loc[:,['Rating', 'sample']], number_rew, number_rew_nan, norm_rank, mean_rews, places_counts], axis = 1)
model_func(df_preproc)
X = pd.concat([number_rew, norm_rank, rating_by_rank, mean_rews, places_counts, data['Rating']], axis = 1)
X.info()
fig, ax = plt.subplots(1,1, figsize = (10,5))
ax = sns.heatmap(X.corr(),annot = True, cmap = 'coolwarm')
data.info()
all_cities = data.City.value_counts().index

fig, ax = plt.subplots(figsize = (15, 5))

sns.boxplot(x='City', y='Rating',data=data[data.Rating > 0],ax=ax)

plt.xticks(rotation=45)
ax.set_title('Boxplot for City')

plt.show()
# выделим список с городами, боксплоты которых отличаются от остальных
phen_cities = [ i for i in all_cities if data[data.City == i].quantile(q = 0.75).Rating - data[data.City == i].quantile(q = 0.25).Rating != 1.5]
phen_cities  
data.City.unique()
# создадим отдельный признак с городами, у которых боксплоты отличаются от остальных

data['phen_cities'] = data.City.apply(lambda x: x if x in phen_cities else 'other')
data['phen_cities'].value_counts()
# воспользуемся get_dummies

cities = pd.get_dummies(data.City, columns=[ 'City'])
cities.sample(5)
phen_cities_dummy = pd.get_dummies(data.phen_cities, columns=[ 'phen_cities'])
phen_cities_dummy.sample(5)
# проверим дамми признаки на модели

df_preproc = pd.concat([data.loc[:,['Rating', 'sample']], \
                        number_rew, number_rew_nan, norm_rank, mean_rews, places_counts, \
                        cities], axis = 1)
model_func(df_preproc)
data['Price Range'].value_counts()
data['Price Range'].isna().sum()
# Пропусков в цене много. Выделим пропуски в цене в отдельный признак

price_isnan = pd.isna(data['Price Range']).astype('uint8')
price_isnan.name = 'price_isnan'
# Определим функцию для заполнения Price Range

def price_ordinal(price):
    if price == '$':
        result = 1
    elif price == '$$ - $$$':
        result = 2
    elif price == '$$$$':
        result = 3
    else:
        result = 0
    return result

prices = data['Price Range'].apply(price_ordinal)
prices.name = 'prices'

prices.value_counts()
# проверим модель с обработанной ценой

df_preproc = pd.concat([data.loc[:,['Rating', 'sample']], \
                        number_rew, number_rew_nan, norm_rank, mean_rews, places_counts, \
                        cities, prices, price_isnan], axis = 1)
model_func(df_preproc)
# посмотрим, сколько всего стилей кухни встречается датасете

cuisine_styles = Counter()

for i in data['Cuisine Style'].dropna():
    l = re.sub('\s\'|\'','', i)[1:-1].split(',')
    cuisine_styles.update(l)

cuisines = [x[0] for x in cuisine_styles.most_common()]

len(cuisines)

cuisine_styles.most_common()
cuisine_most_common = [x[0] for x in cuisine_styles.most_common()[:10]]
cuisine_most_common
# превратим Cuisine Style в список

cuisine_style = data['Cuisine Style'].apply(lambda x: ['other_style'] if pd.isnull(x) else x[1:-1].split(',') )
cuisine_style.sample(5)
# добавим новый признак "Количество кухонь в ресторане"

cuisine_counts = cuisine_style.apply(lambda x: len(x))
cuisine_counts.name = 'cuisine_counts'

cuisine_counts.sample(5)
for i,k in enumerate(cuisine_style):
    new_list = []
    for j in k:
        j = re.sub('\s\'|\'','', j)
        if j in cuisine_most_common:
            new_list.append(j)
        else:
            new_list.append('other_style')
    cuisine_style.at[i] = new_list
cuisine_style.sample(5)
cuisine_style_df = pd.DataFrame(cuisine_style)
for i in cuisine_most_common + ['other_style']:
    cuisine_style_df[i] = cuisine_style.apply(lambda x: 1 if i in x else 0).astype('uint8')

cuisine_style_df.drop('Cuisine Style', axis = 1, inplace=True)

cuisine_style_df.info()
cuisine_style_df.sample(5)
df_preproc = pd.concat([data.loc[:,['Rating', 'sample']], \
                        number_rew, number_rew_nan, norm_rank, mean_rews, places_counts, \
                        cities, prices, price_isnan, \
                        cuisine_counts, cuisine_style_df], axis = 1)
model_func(df_preproc)
data['Cuisine Style'].isna().sum()
data['Cuisine Style'].fillna('other_style', inplace = True)
data['rew_dates'] = data.Reviews.apply(lambda x : [0] if pd.isna(x) else x[2:-2].split('], [')[1][1:-1].split("', '"))
data['max_rew_date'] = pd.to_datetime(data['rew_dates'].apply(lambda x: max(x)))

data['first_rew'] = pd.to_datetime(data['rew_dates'].apply(lambda x : x[0]))
data['second_rew'] = pd.to_datetime(data['rew_dates'].apply(lambda x: x[1] if len(x) == 2 else ''))

rew_delta = np.abs(data['first_rew'] - data['second_rew'])
rew_delta = rew_delta.apply(lambda x: x.days)

rew_delta.name = 'rew_delta'

rew_delta.sample(5)
rew_delta.isna().sum()
# пустых значений много, сделаем новый признак для NAN
rew_delta_isnan = pd.isna(rew_delta).astype('uint8')

rew_delta_isnan.value_counts()
# Заполним пропуски средним

mean = round(rew_delta.mean(), 2)
rew_delta = rew_delta.fillna(mean)
rew_delta.sample(5)
from datetime import datetime

rew_delta_cur = (datetime.now() - data['max_rew_date'])
rew_delta_cur = rew_delta_cur.fillna(rew_delta_cur.median())

rew_delta_cur = rew_delta_cur.apply(lambda x : x.days)

rew_delta_cur.name = 'rew_delta_cur'

rew_delta_cur.sample(5)
df_preproc = pd.concat([data.loc[:,['Rating', 'sample']], \
                        number_rew, number_rew_nan, norm_rank, mean_rews, places_counts, \
                        cities, prices, price_isnan, \
                        cuisine_counts, cuisine_style_df, \
                        rew_delta, rew_delta_cur,rew_delta_isnan], axis = 1)
model_func(df_preproc)
data_add = pd.read_csv(cities_pop_filename)
data_add.sample(3)
cities_info = pd.DataFrame(data.City.value_counts().index)
cities_info.columns = ['city']
cities_info.head(3)
data_europe = data_add[data_add.iso2.apply(lambda x: x not in ('US','CA','VE'))]
data_europe.head()
cities_country = cities_info.merge(data_europe, how = 'left', on = 'city').loc[:, ['city', 'iso2']]
cities_country.info()
cities_country[cities_country.iso2.isna()]
cities_country.at[23,'iso2'] = 'PT'
cities_country.at[25,'iso2'] = 'PL'
cities_country.at[22,'iso2'] = 'CH'
cities_country.at[19,'iso2'] = 'DK'
cities_country.info()
cities_info = cities_info.merge(data_europe.loc[:,['city','capital', 'population']], how = 'left', on = 'city')
cities_info.sample(5)
cities_info.isna().sum()
cities_info['capital'] = cities_info.capital.fillna('not_cap')
cities_info[cities_info.population.isna()]
# заполним пропуски в населении и странах
cities_info.at[23,'population'] = 237591
cities_info.at[25,'population'] = 769498
cities_info.at[22,'population'] = 428737
cities_info.at[19,'population'] = 615993
cities_info.columns =  ['City', 'capital', 'population']
cities_country.columns = ['City', 'country']
# объединим с исходным датасетом

cities_pop = data.loc[:,['City']].merge(cities_info, how = 'left', on = 'City')

cities_pop.drop(['City'], axis = 1, inplace = True)

cities_pop.info()
cities_capital = pd.get_dummies(cities_pop.capital)
cities_pop.drop(['capital'], axis = 1, inplace = True)

cities_capital.sample(5)
# добавим дамми признаки для стран
countries = data.loc[:,['City']].merge(cities_country, how = 'left', on = 'City')

countries.drop(['City'], axis = 1, inplace = True)
countries.info()
countries = pd.get_dummies(countries)
countries.info()
df_preproc = pd.concat([data.loc[:,['Rating', 'sample']], \
                        number_rew, number_rew_nan, norm_rank, mean_rews, places_counts, \
                        cities, prices, price_isnan, \
                        cuisine_counts, cuisine_style_df, \
                        rew_delta, rew_delta_cur,rew_delta_isnan, \
                        cities_pop, cities_capital, countries], axis = 1)
model_func(df_preproc)
cities_info.info()
cities_info['City']
# заведем словарь для гоордов с новыми данными [кол-во тыс. туристов, место в рейтенге благосостояния] по данным из wiki
th = {
    'London' : [19233, 14],
    'Paris' : [17560, 18],
    'Madrid' : [5440, 19],
    'Barcelona' : [6714, 19],
    'Berlin' : [5959, 15],
    'Milan' : [6481, 24],
    'Rome' : [10065, 24],
    'Prague' : [8949, 22],
    'Lisbon' : [3539, 29],
    'Vienna' : [6410, 2],
    'Amsterdam' : [8354, 7],
    'Brussels' : [3942, 13],
    'Hamburg' : [1450, 15],
    'Munich' : [4067, 15],
    'Lyon' : [6000, 18],
    'Stockholm' : [2605, 8],
    'Budapest' : [3823, 31],
    'Warsaw' : [2850, 27],
    'Dublin' : [5213, 16],
    'Copenhagen' : [3070, 5],
    'Athens' : [5728, 36],
    'Edinburgh' : [1660, 14],
    'Zurich' : [2240, 6],
    'Oporto' : [2341, 29],
    'Geneva' : [1150, 6],
    'Krakow' : [2732, 27],
    'Oslo' : [1400, 1],
    'Helsinki' : [1240, 9],
    'Bratislava' : [126, 26],
    'Luxembourg' : [1139, 11],
    'Ljubljana' : [5900, 20]
}
tourists = data.City.apply(lambda x : th[x][0])
tourists.name = 'tourists'

hapiness = data.City.apply(lambda x : th[x][1])
hapiness.name = 'hapiness'

tourists
hapiness
df_preproc = pd.concat([data.loc[:,['Rating', 'sample']], \
                        number_rew, number_rew_nan, norm_rank, mean_rews, places_counts, \
                        cities, prices, price_isnan, \
                        cuisine_counts, cuisine_style_df, \
                        rew_delta, rew_delta_cur,rew_delta_isnan, \
                        cities_pop, cities_capital, countries, \
                        tourists, hapiness], axis = 1)
model_func(df_preproc)
# выделим текст обзоров для последующего анализа.
data['rew_texts'] = data.Reviews.apply(lambda x : '' if pd.isna(x) else x[2:-2].split('], [')[0])

rew_texts_list = data['rew_texts'].apply(lambda x : [''] if x == '' else x.split("', '"))

data['first_text'] = rew_texts_list.apply(lambda x : x[0][1:-1] if len(x) == 1 else x[0][1:] if len(x) == 2 else '')
data['second_text'] = rew_texts_list.apply(lambda x: x[1][:-1] if len(x) == 2 else '')

data.loc[:,['Reviews', 'rew_texts', 'first_text', 'second_text']].sample(5)
def rew_counts_func(row):
    result = 0 if row['rew_texts'] == ''  else 1 if row['second_text'] == '' else 2
    return result
    
rew_counts = data.apply(rew_counts_func, axis = 1)
rew_counts.name = 'rew_counts'

pd.concat([rew_counts, data['rew_texts']], axis = 1)
rew_counts.value_counts()
# количество слов в отзывах

words_count = data['rew_texts'].apply(lambda x : len(x.split()))

words_count.name = 'words_count'
pd.concat([rew_counts, data['rew_texts'], words_count], axis = 1)
df_preproc = pd.concat([data.loc[:,['Rating', 'sample']], \
                        number_rew, number_rew_nan, norm_rank, mean_rews, places_counts, \
                        cities, prices, price_isnan, \
                        cuisine_counts, cuisine_style_df, \
                        rew_delta, rew_delta_cur,rew_delta_isnan, \
                        cities_pop, cities_capital, countries, \
                        tourists, hapiness, \
                        rew_counts], axis = 1)
model_func(df_preproc)
# посмотрим на корреляцию признаков

X = pd.concat([number_rew, cities_pop, tourists, data['Rating']], axis = 1)

fig, ax = plt.subplots(1,1, figsize = (10,5))
ax = sns.heatmap(X.corr(),annot = True, cmap = 'coolwarm')
ss = StandardScaler()
cities_pop_std = pd.DataFrame(ss.fit_transform(cities_pop))
tourists_std = pd.DataFrame(ss.transform(pd.DataFrame(tourists)))

pd.concat([cities_pop_std,tourists_std], axis = 1).corr()

pop_tourists = cities_pop_std + tourists_std
pop_tourists = pd.DataFrame(ss.fit_transform(pop_tourists))

pop_tourists.name = 'pop_tourists'
df_preproc = pd.concat([data.loc[:,['Rating', 'sample']], \
                        number_rew, number_rew_nan, norm_rank, mean_rews, places_counts, \
                        cities, prices, price_isnan, \
                        cuisine_counts, cuisine_style_df, \
                        rew_delta, rew_delta_cur,rew_delta_isnan, \
                        cities_capital, countries, \
                        hapiness, \
                        rew_counts, \
                        pop_tourists], axis = 1)
model_func(df_preproc)
df_preproc = pd.concat([data.loc[:,['Rating', 'sample']], \
                        number_rew, number_rew_nan, norm_rank, mean_rews, places_counts, \
                        cities, prices, price_isnan, \
                        cuisine_counts, cuisine_style_df, \
                        rew_delta, rew_delta_cur,rew_delta_isnan, \
                        cities_pop, cities_capital, countries, \
                        tourists, hapiness, \
                        rew_counts], axis = 1)
model_func(df_preproc)
# выделим тестовую часть
train_data = df_preproc.query('sample == 1').drop(['sample'], axis=1)
test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1)

y = train_data.Rating.values            # наш таргет
X = train_data.drop(['Rating'], axis=1)
    
RANDOM_SEED = 42
    
# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных
# выделим 20% данных на валидацию (параметр test_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# Создаём модель (НАСТРОЙКИ НЕ ТРОГАЕМ)
model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)
    
# Обучаем модель на тестовом наборе данных
model.fit(X_train, y_train)

# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
# Предсказанные значения записываем в переменную y_pred
y_pred = model.predict(X_test)

y_pred = np.round(y_pred * 2) / 2
print('MAE: ',metrics.mean_absolute_error(y_test, y_pred))
    
# в RandomForestRegressor есть возможность вывести самые важные признаки для модели
plt.rcParams['figure.figsize'] = (10,10)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
test_data = test_data.drop(['Rating'], axis=1)
sample_submission.Rating
predict_submission = model.predict(test_data)
predict_submission = np.round(predict_submission * 2)/2
sample_submission['Rating'] = predict_submission
sample_submission.to_csv('submission.csv', index=False)
sample_submission
