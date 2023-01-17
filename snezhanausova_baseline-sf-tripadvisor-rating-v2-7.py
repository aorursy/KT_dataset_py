import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



# Загружаем специальный удобный инструмент для разделения датасета:

from sklearn.model_selection import train_test_split



from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.preprocessing import StandardScaler



import re





pd.set_option('display.max_rows', 50)

pd.set_option('display.max_columns', 50)



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
# Посмотрим внимательнее на сложные признаки

data['Cuisine Style'][1]
data.Reviews[1]
data['URL_TA'][1]
# Посмотрим количество пропусков в каждом признаке и какой процент от всего датасета они занимают.

nan_df = pd.DataFrame(data.isna().sum(), columns=['Количество'])



nan_df['%'] = nan_df['Количество'].apply(lambda x: round((x/len(data))*100, 0))

print(nan_df)
# Вынесем наличие пропусков в указанных колонках (кроме Reviews) в отдельные признаки.

data['Number_of_Reviews_isNAN'] = pd.isna(

    data['Number of Reviews']).astype('uint8')

data['Cuisine_style_isNAN'] = pd.isna(data['Cuisine Style']).astype('uint8')

data['Price_range_isNAN'] = pd.isna(data['Price Range']).astype('uint8')
data.describe()
df_train['Rating'].value_counts(ascending=True).plot(kind='barh')
fig = plt.figure()

axes = fig.add_axes([0, 0, 1, 0.8])

axes.hist(data['Number of Reviews'], bins=100)

axes.set_title('Распределение по количеству отзывов')

axes.set_ylabel('Количество ресторанов')

axes.set_xlabel('Количество отзывов')
# Заполняем пропуски в колонке средним количеством отзывов в зависимости от города

data['Number of Reviews'] = data.groupby('City')['Number of Reviews'].transform(

    lambda x: x.fillna(round(x.mean(), 0)))
plt.rcParams['figure.figsize'] = (10,7)

df_train['Ranking'].hist(bins=100)
df_train['City'].value_counts(ascending=True).plot(kind='barh')
df_train['Ranking'][df_train['City'] == 'London'].hist(bins=100)
# посмотрим на топ 10 городов

for x in (df_train['City'].value_counts())[0:10].index:

    df_train['Ranking'][df_train['City'] == x].hist(bins=100)

plt.show()
# нормализуем признак Ranking в пределах каждого города

means = data.groupby('City')['Ranking'].mean()

std = data.groupby('City')['Ranking'].std()

data['Ranking'] = (data.Ranking - data.City.map(means))/(data.City.map(std))
# Загружаем дополнительный датасет для обработки колонки.



df_cities = pd.read_csv('../input/world-cities-datasets/worldcities.csv')
# Приводим назвние 'Porto' к тому, что используется в датасете data - 'Oporto'

df_cities['city_ascii'] = df_cities.city_ascii.apply(

    lambda x: 'Oporto' if x == 'Porto' else x)
# создадим словарь из датасета df_cities, где ключ - город, значение - страна.

df_cities_1 = df_cities.drop(

    ['city', 'lat', 'lng', 'iso2', 'iso3', 'admin_name', 'capital', 'population', 'id'], axis=1)

df_countries = df_cities_1[(df_cities_1['country'] != 'United States') & (

    df_cities_1['country'] != 'Canada') & (df_cities_1['country'] != 'Venezuela')]

df_countries.set_index("city_ascii", drop=True, inplace=True)

country_dict = df_countries.to_dict()

country_dict_n = country_dict['country']
# Добавим новый признак - страна

data['Country'] = data['City'].apply(lambda x: country_dict_n[x])
# создадим словарь из датасета df_cities, где ключ - город, значение - размер населения

df_population = df_cities[(df_cities['country'] != 'United States') & (

    df_cities['country'] != 'Canada')]

df_population = df_population.drop(

    ['city', 'lat', 'lng', 'iso2', 'iso3', 'admin_name', 'capital', 'country', 'id'], axis=1)



df_population.set_index("city_ascii", drop=True, inplace=True)

population_dict = df_population.to_dict()

population_dict_n = population_dict['population']
# Дополним датасет признаком - население города

data['Population'] = data['City'].apply(lambda x: population_dict_n[x])
# создаем множество из названий столиц из датасета df_cities

capitals = set(df_cities[df_cities['capital'] == 'primary']['city_ascii'])
# Функция для определения статуса города: столица или нет

def capital_check(city):

    if city in capitals:

        return 'capital' 

    return 'non_capital'
# Дополним датасет колонками, определяющими является город столицей или нет.

data['Сity_status'] = data['City'].apply(capital_check)
# Добавим словарь с количеством туристов в каждом городе. Нужного датасета не нашлось, данные берем из отчета за 2018 год Euripean Cities Marketing, сайта statista.com и Википедии

tourists_dict = {

    'London': 71.16,

    'Paris': 52.56,

    'Madrid': 19.83,

    'Barcelona': 19.29,

    'Berlin': 32.87,

    'Milan': 12.29,

    'Rome': 28.55,

    'Prague': 18.25,

    'Lisbon': 10.76,

    'Vienna': 17.41,

    'Amsterdam': 16.94,

    'Brussels': 3.91,

    'Hamburg': 14.53,

    'Munich': 17.12,

    'Lyon': 3.5,

    'Stockholm': 14.59,

    'Warsaw': 3.0,

    'Budapest' :12.5,

    'Dublin': 11.2,

    'Copenhagen': 5.9,

    'Athens': 5.7,

    'Edinburgh' :3.85,

    'Zurich': 4.2,

    'Oporto': 1.6,

    'Geneva': 2.6,

    'Krakow': 3.3,

    'Oslo': 3.6,

    'Helsinki': 1.2,

    'Bratislava': 0.88,

    'Luxembourg': 1.1,

    'Ljubljana': 0.39

}
# Добавим колонку с количеством иностранных туристов в каждом городе



data['tourists_qnt'] = data['City'].apply(lambda x: tourists_dict[x])
# Приводим строковые значения типов кухни к спискам

data['Cuisine Style'] = data['Cuisine Style'].astype(str).apply(

    lambda x: str(x).replace('[', '').replace(']', '').replace("'", "").strip())



data['Cuisine Style'] = data['Cuisine Style'].apply(lambda x:  None if x == 'nan' else [

    info.strip() for info in str(x).split(',')])
# Создаем список из 10 самых популярных типов кухни

cuisine_list = pd.DataFrame(data['Cuisine Style'].dropna(

).tolist()).stack().value_counts().reset_index()

top_cuisine = cuisine_list['index'][:10].tolist()
# Заменяем отсутствующие значения в колонке на 'not_define'

data['Cuisine Style'] = data['Cuisine Style'].apply(

    lambda x: 'not_define' if x == None else x)
# Дополняем датасет новым признаком 'cuisine_qnt' - количество типов кухни в каждом ресторане

data['cuisine_qnt'] = data['Cuisine Style'].apply(lambda x: len(x))
# Оставляем только самые популярные кухни, остальные заменим на 'other'

def check_cousine(raw):

    line = []

    top_list = ['Vegetarian Friendly', 'European', 'Mediterranean',

                'Italian', 'Vegan Options', 'Gluten Free Options', 'Bar', 'French', 'Asian']

    for item in raw:

        if item.strip() == 'not_define':

            line.append('not_define')

        elif item.strip() in top_cuisine:

            line.append(item.strip())

        else:

            line.append('other_cuisine')

    return line





data['Cuisine Style'] = data['Cuisine Style'].apply(check_cousine)
# Дополняем датасет колонками с типом кухни

mlb = MultiLabelBinarizer()



data = data.join(pd.DataFrame(mlb.fit_transform(

    data.pop('Cuisine Style')), index=data.index, columns=mlb.classes_))
data['Price Range'].value_counts()
# Заменяем пропуски в 'Price Range' наиболее часто встречающейся категорией

data['Price Range'] = data['Price Range'].fillna('$$ - $$$')
# Переводим значения из номинативного признака в ординарный с помощью словаря

price_dict = {'$': 1, '$$ - $$$': 2, '$$$$': 3}

data['Price Range'] = data['Price Range'].replace(to_replace=price_dict)
data['Reviews'].value_counts()
data['Reviews'] = data.Reviews.apply(lambda x: None if x == '[[], []]' else x)

data['Review_isNAN'] = pd.isna(data['Reviews']).astype('uint8')
# Разобьем на два признака: содержащий отзывы и содержащий даты.

data[['reviews_text', 'reviews_date']

     ] = data['Reviews'].str.split("'],", expand=True)
# выделим даты

data['reviews_date'] = data.reviews_date.dropna().astype(str).apply(

    lambda x: None if pd.isnull(x) else re.compile('\d*/\d*/\d*').findall(x))
# функция для формирования списка из дат в нужном формате

def to_time(line):

    line = [pd.to_datetime(item) for item in line]

    return line





data['reviews_date'] = data.reviews_date.dropna().apply(to_time)
# функция для вычисления количество дней прошедших между первым и вторым отзывом

def find_delta(line):

    return (max(line) - min(line))





data['delta_reviews_date'] = data['reviews_date'].dropna().apply(find_delta).dt.days
# Заполним пропуски 0

data['delta_reviews_date'] = data['delta_reviews_date'].fillna(0)
# приведем все буквы к нижнему регистру

data['reviews_text'] = data.reviews_text.apply(

    lambda x: x if pd.isnull(x) else x.lower())
# Выделим из текста слова состоящие более чем из двух букв

data['reviews_text_1'] = data.reviews_text.astype(str).apply(

    lambda x: re.compile('[a-z][a-z]\w+').findall(x))
# Посмотрим какие слова в отзывах встречаются чаще всего

word_list = pd.DataFrame(data.reviews_text_1.dropna(

).tolist()).stack().value_counts().reset_index()

word_list[:40]
# Создадим список из наиболее часто встречающихся прилагательных, описывающих впечатление.  Также добавим в список частицу "not"

words_list = ['not', 'good', 'nice', 'great', 'very', 'best', 'excellent',

              'delicious', 'friendly', 'lovely', 'amazing', 'tasty', 'fantastic', 'average']
# Функция, которая оставляет в отзывах только наиболее часто встречающиеся слова

def check_words(raw):

    line = []

    for item in raw:

        if item in words_list:

            line.append(item)

        else:

            continue

    return line





data['reviews_text_1'] = data['reviews_text_1'].apply(check_words)
# Используем функцию для получения "One-Hot-Encoded" из списка.

mlb = MultiLabelBinarizer()



data = data.join(pd.DataFrame(mlb.fit_transform(

    data.pop('reviews_text_1')), index=data.index, columns=mlb.classes_))
# Для кодирования оставшихся категориальных признаков через подход One-Hot Encoding используем функцию get_dummies

data = pd.get_dummies(data, columns=['City', 'Country', 'Сity_status'])
data.info()
data_corr = data[data['sample'] == 1]

plt.rcParams['figure.figsize'] = (15,10)

sns.heatmap(data_corr.drop('sample', axis=1).corr())
# Удаляем оставшиеся номинативные признаки

object_columns = [s for s in data.columns if data[s].dtypes == 'object']

data.drop(object_columns, axis=1, inplace=True)
data.info()
# Нормализуем все данные кроме 'Rating','sample', 'Ranking'. Последний был нормализован ранее относительно городов.

def StandardScaler_column(d_col):

    scaler = StandardScaler()

    scaler.fit(data[[d_col]])

    return scaler.transform(data[[d_col]])





for i in list(data.columns):

    if i not in ['Rating', 'sample', 'Ranking']:

        data[i] = StandardScaler_column(i)

        if len(data[data[i].isna()]) < len(data):

            data[i] = data[i].fillna(0)
df_preproc = data



df_preproc.head(10)
df_preproc.info()
# Теперь выделим тестовую часть

train_data = df_preproc.query('sample == 1').drop(['sample'], axis=1)

test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1)



y = train_data.Rating.values            # наш таргет

X = train_data.drop(['Rating'], axis=1)
# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных

# выделим 20% данных на валидацию (параметр test_size)

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=RANDOM_SEED)
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

plt.rcParams['figure.figsize'] = (10, 10)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(20).plot(kind='barh')
test_data.sample(10)
test_data = test_data.drop(['Rating'], axis=1)
sample_submission
predict_submission = model.predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)