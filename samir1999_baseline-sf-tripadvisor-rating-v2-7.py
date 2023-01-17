# Импортируем необходимые библиотеки
import pandas as pd

import numpy as np

import gensim
from gensim import corpora
from pprint import pprint

from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor 
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer

import re

import datetime

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
%matplotlib inline
import seaborn as sns 

import plotly
import plotly.express as px
import plotly.graph_objects as go
import cufflinks as cf
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot

import folium

pyo.init_notebook_mode(connected=True)
cf.go_offline()

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!
RANDOM_SEED = 25
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

df = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
df.info()
df.sample(5)
df.Reviews[1]
plt.figure(figsize=(14, 4))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis');
df['Number_of_Reviews_isNAN'] = pd.isna(
    df['Number of Reviews']).astype('uint8')
df['Cuisine Style_isNAN'] = pd.isna(df['Cuisine Style']).astype('uint8')
df['Price Range_isNAN'] = pd.isna(df['Price Range']).astype('uint8')
df['Reviews_isNAN'] = pd.isna(df['Reviews']).astype('uint8')
# Далее заполняем пропуски 0, вы можете попробовать заполнением средним или средним по городу и тд...
df['Number of Reviews'].fillna(0, inplace=True)
# Напишем функцию, которая выведет всю статистическую инфу по переменной и строит графики
def analyse_numeric(datafr, column):
    '''Визуализирует распределение числовой переменной.
       Принимает параметрами DataFrame и строковое название столбца
       Печатает различные статистические показатели и строит гистограмму.'''
    count = datafr[column].count()
    mean = datafr[column].mean()
    std = datafr[column].std()
    median = datafr[column].median()
    perc25 = datafr[column].quantile(0.25)
    perc75 = datafr[column].quantile(0.75)
    IQR = perc75 - perc25
    range_min = datafr[column].min()
    range_max = datafr[column].max()
    margin = (range_max - range_min)/10
    range_start = range_min - margin
    range_stop = range_max + margin
    range_ = (range_start, range_stop)
    outliers = datafr[column].loc[(
        datafr[column] < perc25 - 1.5*IQR) | (datafr[column] > perc75 + 1.5*IQR)]

    print('Количество: {}, Среднее: {:.3f}, Стандартное отклонение: {:.3f}.'.format(
        count, mean, std))
    print('Минимум: {}, 25-й перцентиль: {}, Медиана: {}, 75-й перцентиль: {}, Максимум: {}, IQR: {}.'
          .format(range_min, perc25, median, perc75, range_max, IQR))
    print('Количество пропусков в столбце: ', pd.isnull(datafr[column]).sum())
    print('Границы выбросов: [{f}, {l}].'.format(
        f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR), 'Количество выбросов: ', len(outliers))

    datafr[column].loc[datafr[column].between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)] \
                  .hist(bins=30, range=range_, label='В границах выбросов')
    outliers.hist(bins=30, range=range_, label='Выбросы')

    plt.legend()
df.nunique(dropna=False)
df.Restaurant_id.nunique()
df_count = df.City.value_counts()
df_count
# Добавим в наш датасет количество ресторанов по городам
df['quantity'] = df['City'].apply(lambda x: df_count[x])
population_dict = {
    'London': 9304016,
    'Paris': 2140526,
    'Madrid': 3348536,
    'Barcelona': 1620343,
    'Berlin': 3748148,
    'Milan': 1404239,
    'Rome': 2856133,
    'Prague': 1324277,
    'Lisbon': 506654,
    'Vienna': 1911728,
    'Amsterdam': 873555,
    'Brussels': 1209000,
    'Hamburg': 1841179,
    'Munich': 1471508,
    'Lyon': 515695,
    'Stockholm': 974073,
    'Budapest': 1752286,
    'Warsaw': 1790658,
    'Dublin': 554554,
    'Copenhagen': 626508,
    'Athens': 664046,
    'Edinburgh': 524930,
    'Zurich': 415367,
    'Oporto': 237559,
    'Geneva': 201818,
    'Krakow': 779115,
    'Oslo': 693491,
    'Helsinki': 648042,
    'Bratislava': 432864,
    'Luxembourg': 613894,
    'Ljubljana': 292988
}
df['population'] = df['City'].map(population_dict)
df['population'] = df['population'] / 1000
df['quantity_density'] = df.quantity / df.population
capitals = ['Paris', 'Stockholm', 'London', 'Berlin',
            'Bratislava', 'Vienna', 'Rome', 'Madrid',
            'Dublin', 'Brussels', 'Warsaw', 'Budapest', 'Copenhagen',
            'Amsterdam', 'Lisbon', 'Prague', 'Oslo',
            'Helsinki', 'Ljubljana', 'Athens', 'Luxembourg']
# Теперь нужно соотнести столицы по странам
countries_dict = {'Amsterdam': 'Netherlands',
                  'Athens': 'Greece',
                  'Barcelona': 'Spain',
                  'Berlin': 'Germany',
                  'Bratislava': 'Slovakia',
                  'Brussels': 'Belgium',
                  'Budapest': 'Hungary',
                  'Copenhagen': 'Denmark',
                  'Dublin': 'Ireland',
                  'Edinburgh': 'UK',
                  'Geneva': 'Switzerland',
                  'Hamburg': 'Germany',
                  'Helsinki': 'Finland',
                  'Krakow': 'Poland',
                  'Lisbon': 'Portugal',
                  'Ljubljana': 'Slovenia',
                  'London': 'UK',
                  'Luxembourg': 'Luxembourg',
                  'Lyon': 'France',
                  'Madrid': 'Spain',
                  'Milan': 'Italy',
                  'Munich': 'Germany',
                  'Oporto': 'Portugal',
                  'Oslo': 'Norway',
                  'Paris': 'France',
                  'Prague': 'Czechia',
                  'Rome': 'Italy',
                  'Stockholm': 'Sweden',
                  'Vienna': 'Austria',
                  'Warsaw': 'Poland',
                  'Zurich': 'Switzerland'}

df['country'] = df.apply(lambda row: countries_dict[row['City']], axis=1)
# Добавим дамми-переменные по городам
df = pd.concat([df, pd.get_dummies(df.City, prefix='City')], axis=1)
df.head(3)
# Добавим столбец Rest_density - сколько ресторанов приходится на человека в городе (плотность ресторанов):
df['quantity_density'] = df['quantity'] / df['population']
# Рассмотрим распределение ранга ресторанов по странам:
df.pivot_table(values=['Ranking'],
               index='country',
               aggfunc='mean').iplot(kind='bar', title='Распределение ранга ресторанов по странам')
# Добавим столбец с количеством представленных стилей кухонь в ресторане:
df['Cuisine Style'] = df['Cuisine Style'].fillna("['Other']")
df['Cuisine_count'] = df['Cuisine Style'].str[2:-2].str.split("', '").str.len()
df.Cuisine_count.describe()
# Рассмотрим расределение ранга ресторанов и количества отзывов в зависимости от количества кухонь:
df.pivot_table(values=['Ranking', 'Number of Reviews'],
               index='Cuisine_count',
               aggfunc='mean').iplot(kind='bar', title='Ранг ресторанов и количество отзывов в зависимости от количества кухонь')
# Начнём с просмотра статистики
analyse_numeric(df, 'Ranking')
# Посмотрим на топ 5 городов:
(df['City'].value_counts())[0:5].index
# Взглянем на график
df[['London', 'Paris', 'Madrid', 'Barcelona', 'Berlin']].iplot(
    kind='hist', title='Распределение ранга по городам', bins=100)
df['Relative_rank'] = df['Ranking'] / df['quantity']
df['mean_ranking'] = df['Ranking'] / \
    df['City'].map(df.groupby(['City'])['Ranking'].max())
df.head(3)
analyse_numeric(df, 'Rating')
# Взглянем на график распределения с помощью более качественного рисунка
df['Rating'].value_counts(ascending=True).iplot(
    kind='bar', title='Распределение рейтинга ресторанов')
df['Price Range'].value_counts(dropna=False)
# Очень много пропусков. Обработаем их и заполним медианой
df['Price Range isna'] = pd.isna(df['Price Range']).astype('uint8')
df['Price Range label'] = df['Price Range'].apply(lambda x: 2.5 if pd.isnull(
    x) else 1.0 if x == '$' else 2.5 if x == '$$ - $$$' else 4.0)
df.head(3)
# Время красивых графиков
analyse_numeric(df, 'Price Range label')
df.pivot_table(values=['Price Range label'],
               index='City',
               aggfunc='mean').iplot(kind='bar', title='Уровень цен по ресторанам')
analyse_numeric(df, 'Number of Reviews')
df.pivot_table(values=['Number of Reviews'],
               index='Rating',
               aggfunc='mean').iplot(kind='bar', title='Распределение ранга от количества отзывов')
# Посмотрим распределение количества отзывов по городам:
df_reviews_by_city = df.groupby(
    ['City'])['Number of Reviews'].sum().sort_values(ascending=False)
df_reviews_by_city
df_reviews_by_city.sort_values().iplot(
    kind='bar', title='Количество отзывов о ресторанах по городам')
# Данные достаточно занимательные. Добавим их в датасет
df['Reviews in city'] = df['City'].apply(lambda x: df_reviews_by_city[x])
# Нормируем количество отзывов к населению города
df['Number of Reviews norm'] = df['Number of Reviews'] / df['population']
# Приведём для начала данные в более приличный вид
df['Reviews'] = df['Reviews'].fillna('[[], []]')
# Извлекем дату из ревью и создадим новые критерии:
df['Date_of_review'] = df['Reviews'].str.findall('\d+/\d+/\d+')
df['Len_date'] = df['Date_of_review'].apply(lambda x: len(x))
df[['Date_of_review', 'Reviews']].head()
# проверим длину даты поля
df.Date_of_review.apply(lambda x: len(x)).value_counts()
# Рассмотрим отзывы с тремя датами:
print("значения Reviews с тремя датами :=")
temp_list = df[df['Len_date'] == 3].Reviews.to_list()
display(df[df['Len_date'] == 3].Reviews.to_list())
display([re.findall('\d+/\d+/\d+', x) for x in temp_list])
# Видим что люди указывали даты в отзывах и эти даты попали в обработку, оставим даты вне отзывов:
df['Len_date'].Date_of_review = df[df['Len_date']
                                   == 3].Date_of_review.apply(lambda x: x.pop(0))
# Заполним перерыв между отзывами (по двум отзывам) и отследим насколько давно был сделан последний отзыв:
def time_to_now(row):
    if row['Date_of_review'] == []:
        return None
    return pd.datetime.now() - pd.to_datetime(row['Date_of_review']).max()


def time_between_reviews(row):
    if row['Date_of_review'] == []:
        return None
    return pd.to_datetime(row['Date_of_review']).max() - pd.to_datetime(row['Date_of_review']).min()


df['Day_to_now'] = df.apply(time_to_now, axis=1).dt.days
df['Day_between_reviews'] = df[df['Len_date'] == 2].apply(
    time_between_reviews, axis=1).dt.days
analyse_numeric(df, 'Day_between_reviews')
analyse_numeric(df, 'Day_to_now')
# Проверим количество пропусков в новых переменных
print('Пропусков в столбце с разницей между сегодняшним днём и последним отзывом: {}'.format(
    df['Day_to_now'].isna().sum()))
print('Пропусков в столбце с разницей между двумя отзывами: {}'.format(
    df['Day_between_reviews'].isna().sum()))
# Заполним пропуски нулями

df['Day_to_now'] = df['Day_to_now'].fillna(0)
df['Day_between_reviews'] = df['Day_between_reviews'].fillna(0)
# Создадим параметр Rev_year - год последнего отзыва.
# Для этого выделим из даты отзыва более свежий отзыв:
def last_review(row):
    if row == []:
        return None
    return pd.to_datetime(row).max()


df['Last_review'] = df['Date_of_review'].apply(last_review)
# Заполним пропуски минимальным значением:
df['Last_review'].min()
df['Last_review'] = df['Last_review'].apply(lambda x: '2004-04-21' if pd.isnull(x) else x)
df['Rev_year'] = df['Last_review'].dt.year
# Создадим параметр Rev_weekday - день недели последнего отзыва:
df['Rev_weekday'] = df['Last_review'].dt.dayofweek
df = df.drop(['Last_review'], axis = 1)
# проверяем
test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape
df_train['Rating'].value_counts(ascending=True).plot(kind='barh')
plt.rcParams['figure.figsize'] = (15,10)
sns.heatmap(df.drop(['sample'], axis=1).corr(),)
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
test_data.sample(10)
test_data = test_data.drop(['Rating'], axis=1)
sample_submission
predict_submission = model.predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head(10)
