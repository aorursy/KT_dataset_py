import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import missingno as msno

import seaborn as sns

import cufflinks as cf

import plotly.figure_factory as ff

import plotly.graph_objs as go

import re

# Загружаем специальный инструмент для разбивки:

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn.metrics import mean_absolute_error # инструменты для оценки точности модели

from sklearn.preprocessing import OrdinalEncoder # OrdinalEncoder для присвоения порядковых номеров множеству данных

from sklearn import metrics

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# Загружаем данные

DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



df = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
df.sample(5)
# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!

RANDOM_SEED = 42

# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

!pip freeze > requirements.txt

# Постоянный сегодняшний день

date_today = pd.to_datetime('2020-09-18')
def chart(col):

    """Показывает график"""

    df[col].value_counts(ascending=True).plot(kind='barh')





def hist(name, bins):

    """Показывает гистограмму"""

    return name.hist(bins=bins)





def pie_chart(df):

    """Показывает график Pie chart"""

    global_number_rest = df.value_counts(dropna=False)

    total_rest = global_number_rest.sum()

    explode = [0.1 for i in global_number_rest]

    global_number_rest.plot(kind='pie', figsize=(25, 25), explode=explode, fontsize=20, autopct=lambda v: int(v*total_rest/100),

                            title=df.name)

    plt.show()





def get_boxplot(column):

    """Показывает boxplot"""

    fig, ax = plt.subplots(figsize=(14, 4))

    sns.boxplot(x=column, y='Rating',

                data=df.loc[df.loc[:, column].isin(

                    df.loc[:, column].value_counts().index[:10])],

                ax=ax)

    plt.xticks(rotation=45)

    ax.set_title('Boxplot for ' + column)

    plt.show()





def value_counts(name):

    """Показывает уникальные значения"""

    display(pd.DataFrame(name.value_counts(dropna=False)))





def MAE(df_temp):

    y = df_temp['Rating']

    X = df_temp.drop(['Rating'], axis=1)



    # Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.

    # Для тестирования мы будем использовать 20% от исходного датасета.

    X_train, X_test, y_train, y_test = train_test_split(

        X, y, test_size=0.2, random_state=RANDOM_SEED)



    # Создаём модель

    regr = RandomForestRegressor(

        n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)



    # Обучаем модель на тестовом наборе данных

    regr.fit(X_train, y_train)



    # Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

    # Предсказанные значения записываем в переменную y_pred

    y_pred = regr.predict(X_test)



    # Т. к. целевая переменная кратна 0.5, добавим здесь округление y_pred до 0.5

    y_pred = np.round(y_pred*2)/2



    # Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

    # Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

    print()

    print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

    print()



    # в RandomForestRegressor есть возможность вывести самые важные признаки для модели

    plt.rcParams['figure.figsize'] = (10, 10)

    feat_importances = pd.Series(regr.feature_importances_, index=X.columns)



    return regr
df.info()
print(df.isna().sum(),msno.matrix(df, color=(0.905, 0.221, 0.578), sparkline=False))
df['Cuisine_Style_isNAN'] = pd.isna(df['Cuisine Style']).astype('uint8')

df['Price_Range_isNAN'] = pd.isna(df['Price Range']).astype('uint8')

df['Number_of_Reviews_isNAN'] = pd.isna(

    df['Number of Reviews']).astype('uint8')

df['Number of Reviews'].fillna(0, inplace=True)
df.duplicated().sum()
df_tmp = df[['Ranking', 'Number of Reviews', 'Rating']]

MAE(df_tmp)
df.drop(['Restaurant_id'], axis=1, inplace=True)
pie_chart(df['Rating'])

hist(df.Rating,7)
sns.boxplot(df.Rating, color='purple');
print(value_counts(df['City']), pie_chart(df['City']),

      'Кол-во городов: ', df['City'].nunique())
number_of_cities = {

    'London': 8787892,

    'Paris': 2140526,

    'Madrid': 3223334,

    'Barcelona': 1620343,

    'Berlin': 3601131,

    'Milan': 1366180,

    'Rome': 2872800,

    'Prague': 1280508,

    'Lisbon': 505526,

    'Vienna': 1840573,

    'Amsterdam': 859732,

    'Brussels': 144784,

    'Hamburg': 1830584,

    'Munich': 1456039,

    'Lyon': 515695,

    'Stockholm': 961609,

    'Budapest': 1749734,

    'Warsaw': 1758143,

    'Dublin': 553165,

    'Copenhagen': 615993,

    'Athens': 655780,

    'Edinburgh': 476100,

    'Zurich': 402275,

    'Oporto': 221800,

    'Geneva': 196150,

    'Krakow': 766739,

    'Oslo': 673469,

    'Helsinki': 643272,

    'Bratislava': 413192,

    'Luxembourg': 576249,

    'Ljubljana': 277554

}



df['Population'] = df['City'].apply(lambda x: number_of_cities[x])

df_tmp = df['City'].value_counts()

df['Number_of_restaurans'] = df['City'].apply(lambda x: df_tmp[x])
df['Density_restaurans'] = df['Number_of_restaurans'] / df['Population']
capital_or_not = {

    'London': 1,

    'Paris': 1,

    'Madrid': 1,

    'Barcelona': 0,

    'Berlin': 1,

    'Milan': 0,

    'Rome': 1,

    'Prague': 1,

    'Lisbon': 1,

    'Vienna': 1,

    'Amsterdam': 1,

    'Brussels': 1,

    'Hamburg': 0,

    'Munich': 0,

    'Lyon': 0,

    'Stockholm': 1,

    'Budapest': 1,

    'Warsaw': 1,

    'Dublin': 1,

    'Copenhagen': 1,

    'Athens': 1,

    'Edinburgh': 1,

    'Zurich': 1,

    'Oporto': 0,

    'Geneva': 1,

    'Krakow': 0,

    'Oslo': 1,

    'Helsinki': 1,

    'Bratislava': 1,

    'Luxembourg': 1,

    'Ljubljana': 1

}



df['Capital'] = df['City'].apply(lambda x: capital_or_not[x])
country_name = {

    'London': 'England',

    'Paris': 'France',

    'Madrid': 'Spain',

    'Barcelona': 'Spain',

    'Berlin': 'Germany',

    'Milan': 'Italy',

    'Rome': 'Italy',

    'Prague': 'Czech',

    'Lisbon': 'Portugal',

    'Vienna': 'Austria',

    'Amsterdam': 'Nederlands',

    'Brussels': 'Belgium',

    'Hamburg': 'Germany',

    'Munich': 'Germany',

    'Lyon': 'France',

    'Stockholm': 'Sweden',

    'Budapest': 'Hungary',

    'Warsaw': 'Poland',

    'Dublin': 'Ireland',

    'Copenhagen': 'Denmark',

    'Athens': 'Greece',

    'Edinburgh': 'Schotland',

    'Zurich': 'Switzerland',

    'Oporto': 'Portugal',

    'Geneva': 'Switzerland',

    'Krakow': 'Poland',

    'Oslo': 'Norway',

    'Helsinki': 'Finland',

    'Bratislava': 'Slovakia',

    'Luxembourg': 'Luxembourg',

    'Ljubljana': 'Slovenija'

}



df['Country'] = df['City'].apply(lambda x: country_name[x])
print(value_counts(df['Country']), pie_chart(

    df['Country']), 'Кол-во стран: ', df['Country'].nunique())
df_tmp = df['Country'].value_counts()

df['Restaurants_in_Country'] = df['Country'].apply(lambda x: df_tmp[x])
df['Rest_City_concentration'] = df['Number_of_restaurans'] / df['Restaurants_in_Country']
df.dtypes
economies = {

    'England': 45741,

    'France': 45893,

    'Spain': 40172,

    'Germany': 52386, 

    'Italy': 39676, 

    'Czech': 37340,

    'Portugal': 32412,

    'Austria': 52172,

    'Nederlands': 56489,

    'Belgium': 48327,

    'Sweden': 53652, 

    'Hungary': 26448,

    'Poland': 32005,

    'Ireland': 79617, 

    'Denmark': 52279,

    'Greece': 29072,

    'Schotland': 45741, 

    'Switzerland': 65010,

    'Norway': 74357,

    'Finland': 46596, 

    'Slovakia': 35136,

    'Luxembourg': 106372,

    'Slovenija': 36741}

df['Country_economies'] = df['Country'].apply(lambda x: economies[x])
salary = {

    'Spain': 2495,

    'France': 3417,

    'England': 3413,

    'Italy': 2888,

    'Germany': 4415,

    'Portugal': 1670,

    'Czech': 1621,

    'Poland': 1410,

    'Austria': 4141,

    'Nederlands': 3238,

    'Belgium': 3865,

    'Switzerland': 6270,

    'Sweden': 3891,

    'Hungary': 1145,

    'Ireland': 3671,

    'Denmark': 6247,

    'Greece': 1203,

    'Schotland': 3071,

    'Norway': 5458,

    'Finland': 3937,

    'Slovakia': 1.4,

    'Luxembourg': 5948,

    'Slovenija': 2282

}



df['Country_salary'] = df['Country'].apply(lambda x: salary[x])
crime = {

    'Spain': 0.8,

    'France': 1,

    'England': 1,

    'Italy': 0.9,

    'Germany': 0.8,

    'Portugal': 1.2,

    'Czech': 1,

    'Poland': 1.2,

    'Austria': 0.9,

    'Nederlands': 0.9,

    'Belgium': 1.6,

    'Switzerland': 0.6,

    'Sweden': 0.7,

    'Hungary': 1.9,

    'Ireland': 1.2,

    'Denmark': 0.8,

    'Greece': 1.7,

    'Schotland': 1,

    'Norway': 2.2,

    'Finland': 1.6,

    'Slovakia': 1.4,

    'Luxembourg': 0.8,

    'Slovenija': 0.7

}



df['Country_crime'] = df['Country'].apply(lambda x: crime[x])
happiness = {

    'Spain': 6.403,

    'France': 6.442,

    'England': 6.714,

    'Italy': 5.964,

    'Germany': 6.951,

    'Portugal': 5.195,

    'Czech': 6.609,

    'Poland': 5.973,

    'Austria': 7.006,

    'Nederlands': 7.377,

    'Belgium': 6.891,

    'Switzerland': 7.494,

    'Sweden': 7.284,

    'Hungary': 4.714,

    'Ireland': 6.977,

    'Denmark': 7.522,

    'Greece': 5.227,

    'Schotland': 6.714,

    'Norway': 7.537,

    'Finland': 7.469,

    'Slovakia': 6.098,

    'Luxembourg': 6.863,

    'Slovenija': 5.758

}



df['Country_happy'] = df['Country'].apply(lambda x: happiness[x])
df_tmp = df[['Ranking', 'Rating', 'Number of Reviews', 'Cuisine_Style_isNAN', 'Price_Range_isNAN', 'Number_of_Reviews_isNAN',

             'Number_of_restaurans', 'Population', 'Density_restaurans', 'Capital', 'Rest_City_concentration','Country_economies','Country_salary', 'Country_crime', 'Country_happy']]



MAE(df_tmp)
df['Cuisine_Style_list'] = df['Cuisine Style'].str[1:-1].str.strip().str.split(',')

df['Cuisine_Style_list'].fillna('U', inplace=True) # неуказанная кухня станет "U" (Unknown)

df.drop(columns='Cuisine Style', inplace=True)
cuisine_set = set()

for cuis_list in df['Cuisine_Style_list']:

    for cuis in cuis_list:

        cuisine_set.add(cuis.replace("'", '').strip())



print('Всего кухонь: ', len(cuisine_set))
cuisine_count = dict.fromkeys(cuisine_set, 0)

for cuis in cuisine_set:

    for cuis_list in df['Cuisine_Style_list']:

        if cuis in cuis_list:

            cuisine_count[cuis] += 1



cuisine_count = pd.Series(cuisine_count)

cuisine_count.sort_values(ascending=False)
df['Cuisine_count'] = df['Cuisine_Style_list'].apply(

    lambda x: 1 if x == 1 else len(x))
hist(df.Ranking,bins = 100)
for x in (df['City'].value_counts())[0:10].index:

    df['Ranking'][df['City'] == x].hist(bins=100)

plt.show()
df['Relative_rank'] = df['Ranking'] / df['Number_of_restaurans']
df['Price Range'].value_counts()
df['Price Range'].fillna('$$ - $$$', inplace=True)
price_map = {'$': 1, '$$ - $$$': 2, '$$$$': 3}

df['Price Range'].replace(price_map, inplace=True)
df_tmp = df.groupby(['City'])[

    'Number of Reviews'].sum().sort_values(ascending=False)

df_tmp.sort_values().plot(kind='barh')
df['Rev_in_city'] = df['City'].apply(lambda x: df_tmp[x])
df['Relative_rank_rev'] = df['Ranking'] / df['Rev_in_city']
df['Reviews'].fillna('[[], []]', inplace=True)
def processing_reviews(string):

    """Обработка для Reviews"""

    string = string.replace(']]', '')

    string = string.replace("'", '')

    string = string.split('], [')[1]

    string = string.split(', ')

    return string
def find_words(df, word):

    """Создаем колонку с 1/0 для искомого слова в обзоре"""

    col_name = f'Rev_{word}'

    d = {col_name: 0}

    df = df.assign(**d)

    df.loc[df['Reviews'].str.lower().str.contains(word), col_name] = 1

    return df



words = ['terrible',

         'horrible',

         'not good',

         'disappointing',

         'worst',

         'better',

         'bad',

         'excellent',

         'best',

         'amazing',

         'great']



for word in words:

    df = find_words(df, word)
df['Reviews_date_temp'] = df['Reviews'].apply(processing_reviews)

df['Reviews_date_first'] = df['Reviews_date_temp'].apply(

    lambda x: x[1] if len(x) == 2 else None)

df['Reviews_date_last'] = df['Reviews_date_temp'].apply(

    lambda x: x[0] if len(x) > 0 else None)



# Преобразуем в формат дат

df['Reviews_date_first'] = pd.to_datetime(df['Reviews_date_first'])

df['Reviews_date_last'] = pd.to_datetime(df['Reviews_date_last'])
t = df['Reviews_date_last'].mean()



df['Reviews_date_first'].fillna(t, inplace=True)

df['Reviews_date_last'].fillna(t, inplace=True)
df['date_delta'] = (df['Reviews_date_last'] - df['Reviews_date_first']).dt.days
df['date_delta_today'] = (date_today - df['Reviews_date_last']).dt.days
df.drop(['Reviews', 'Reviews_date_temp', 'Reviews_date_first',

         'Reviews_date_last'], axis=1, inplace=True)
def processing_url(string):

    """Обработка для URL_TA"""

    string = string.replace('/Restaurant_Review-g', '')

    string = string.split('-')

    return int(string[0])
df['url_ta_index'] = df['URL_TA'].apply(processing_url)

df.drop(['URL_TA'], axis=1, inplace=True)
df['ID_TA'] = df['ID_TA'].apply(lambda x: int(x[1:]))
plt.rcParams['figure.figsize'] = (20,20)

sns.heatmap(df.corr(), square=True,

            annot=True, fmt=".1f", linewidths=0.1, cmap="RdBu");

print(plt.tight_layout(),pd.DataFrame(df.corr()['Rating']).sort_values('Rating'))
df = pd.get_dummies(df, columns=['City', 'Country'])
for cuis in cuisine_set:

    df[cuis] = 0

    df[cuis] = df['Cuisine_Style_list'].apply(lambda x: 1 if cuis in x else 0)



df.drop(['Cuisine_Style_list'], axis=1, inplace=True)
train_data = df.query('sample == 1').drop(['sample'], axis=1)

test_data = df.query('sample == 0').drop(['sample'], axis=1)
df.info()
model = MAE(train_data)
test_data.sample(10)
test_data.drop(['Rating'], axis=1, inplace=True)
sample_submission
predict_submission = model.predict(test_data)
predict_submission = np.round(predict_submission*2)/2
predict_submission
sample_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)