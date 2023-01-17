import pandas as pd

# Загружаем специальный удобный инструмент для разделения датасета:

from sklearn.model_selection import train_test_split

import numpy as np

import re

import datetime

import seaborn as sns

import matplotlib.pyplot as plt

import math

%matplotlib inline



pd.set_option('display.max_rows', 500)

pd.set_option('display.width', 10000)



path='../input/rests-and-words/'



# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!

RANDOM_SEED = 42

# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

!pip freeze > requirements.txt
# Набор констант для тестирования гипотез

MaxInterval = 1500    # Максимальный интервал между отзывами в днях



INTERVAL = -10000     # Констатнта (дней) для заполнения отсутстующих интервалов между отзывами (nan)

INTERNOW = -10000     # То же самое для интервала между отзывом и текущей датой



MAX_or_MIN = 'min'    # Минимальный или максимальный ранг: критерий отбора для удаления дублирующихся строк



POS =  1              # Вес положительных отзывов в оценке 

NEG = -3              # Вес негативных отзывов в оценке



NET_SIZE = 3          # Минимальное количество ресторанов для маркирования как сетевой



MINUS = -2            # Снижение баллов по полю num_of_top_cuis для ресторанов, не указавших кухни (там, где nan)

TOPCUIS = 5           # Количество наиболее полпулярных кухонь для преобразования в dummy-переменные

# Функция для поиска интервала между датами

def interval (d_list):

    interval = interNow = None

    

    if len(d_list) == 1:

        interNow = (pd.to_datetime('01-01-2020') - pd.to_datetime(d_list[0])).days

        if interNow > MaxInterval: interNow = None   

    

    if len(d_list) == 2:

        d1=pd.to_datetime(d_list[0])

        d2=pd.to_datetime(d_list[1])

        interval = abs((d1-d2).days)

        interNow = (pd.to_datetime('01-01-2020') - max(d1, d2)).days

        if interval > MaxInterval: interval = None   # отсекаем хвост (вылеты)

        if interNow > MaxInterval: interNow = None   



    return interval, interNow





# Функция для создания набора значений (словаря) из серии (из отдельного поля датасета).

def make_collection(df, col):

    collection = {}

    for i in df[col]:

        for j in i:

            if j in collection.keys():

                collection[j]+=1

            else:

                collection[j] =1

    return collection





# Код для создания словарей из слов, использованных в отзывах (для последующего ранжирования отзывов)

df=pd.read_csv(path+'main_task.csv')

df['Reviews']=df['Reviews'].apply(lambda x: x.lower())

pattern = re.compile('[a-z]*')

df['new'] = df['Reviews'].apply(pattern.findall)

words = pd.Series(make_collection(df,'new')).drop('').sort_values(ascending=False)



# Из полученного словаря берем 1000 наиболее употребимых слов и создаем списки "хороших" и "плохих" слов в отзывах.

# Список сохраняем в csv файл: Dict. После того, как забрали слова - убираем поле new.

df.drop(columns=['new'], inplace=True)





# Функция обработки строки => делает из строки список строк

def simple_list(string):

    string = string.lower()

    str_list = [i.strip("'[]") for i in string.split(', ')]

    return str_list





# Формируем набор полей из значений в поле (в том числе, из значений внутри списков)

def list_to_columns(df, col, not_a_list=False):

    if not_a_list:

        collection = df[col].unique()

    else:

        collection = make_collection(df,col)

    for i in collection:

        df[i] = df[col].apply(lambda x: 1 if (i in x) or (i=='other') else 0)

    return 0





# Функция, обратная list_to_columns: собирает для отдельного города все стили, которые там есть.

def make_cuis_list (city, Cities, pos):

    cuislist=[]

    for cuis in Cities.columns[pos:]:

        if Cities.loc[city.name,cuis] != 0:

            cuislist.append(cuis)

    return cuislist





# Определяем два наиболее популярных стиля в конкретном городе для заполнения nan значений в поле Cuisin Style

def make_pop_cuis (city, Cities, pos):

    result = list(Cities.loc[city.name, Cities.columns[pos:]].sort_values(ascending=False).head(5).index)

    return result





def isnan(x):

    if x!=x: return True

    return False
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample']  = 0 # помечаем где у нас тест

df_test['Rating']  = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



df = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
# 1. смотрим на данные



display (df.info())

display (df.describe())

display (df.head(3))
for col in df.columns:

    print (col+':', len(df[col].value_counts()))
# Создаем признак сетевого ресторана

rests = df['Restaurant_id'].value_counts()

df['net_size'] = df['Restaurant_id'].apply(lambda x: rests[x])
id_ta = df['ID_TA'].value_counts()

id_ta = id_ta[id_ta>1].copy()



dub_s = pd.DataFrame()

for item in id_ta.index.to_list():

    dub_s = dub_s.append(df[(df['ID_TA']==item)])

dub_s
def clean_dubs(df, max_or_min):

    id_ta = df['ID_TA'].value_counts()

    id_ta = id_ta[id_ta>1].copy()

    

    dub_s = pd.DataFrame()

    for item in id_ta.index.to_list():

        dub_s = dub_s.append(df[(df['ID_TA']==item)][['Ranking','ID_TA']])

    

    dub = dub_s.groupby(by='ID_TA')['Ranking'].agg(['max','min'])

    name = max_or_min + '_index'

    

    dub[name] = dub.apply(lambda x: dub_s.loc[dub_s.Ranking==x[max_or_min]].index[0], axis=1)

    df.drop(index = dub[name], inplace=True)

    return df



#clean_dubs(df, MAX_or_MIN)
# Обработка поля 'Price Range'

df['Price Range'] = df['Price Range'].apply(lambda x: 0 if isnan(x) \

                                           else 1 if x=='$' \

                                           else 3 if x=='$$$$' \

                                           else 2)
# Нормализация поля Ranking по максимальному рангу в городе

rang_by_city = df[['City','Ranking']].groupby(by='City').max()



df['Max_Rang'] = df.apply(lambda x: rang_by_city.loc[x.City], axis=1) 



df['Ranking']  = df['Ranking']/df['Max_Rang']

df.drop(columns=['Max_Rang'],inplace=True)



# Гипотеза: учесть количество ресторанов в каждом городе. Чем больше ресторанов - тем больше вес у соответствующего ранга.
# Находим даты

pattern = re.compile('\d\d/\d\d/\d{4}')

df['Reviews'].fillna('[[], []]',inplace=True)

df['dates']   = df['Reviews'].apply(pattern.findall)



# Находим интервалы (interval - между двумя отзывами, InterNow - между последним (или единственным) отзывом и текущей датой)

df['dates'] = df['dates'].apply(interval)

df['interval'] = df['dates'].apply(lambda x: x[0])

df['interNow'] = df['dates'].apply(lambda x: x[1])

df.drop(columns=['dates'], inplace=True)



# Уменьшаем всю серию interNow нс скаляр (моду) и убираем получившиеся отрицательные значения

df['interNow']-= df['interNow'].value_counts().idxmax()

df['interNow'] = df['interNow'].apply(lambda x: -x if x<0 else x)





# Обработка nan

def my_fillna(col):

    temp = df[[GRUPBY,col]].groupby(by=GRUPBY).mean()

    df[col] = df.apply(lambda x: temp.loc[x[GRUPBY]], axis=1)

    return 0





df['interval'].fillna(INTERVAL,inplace=True)

df['interNow'].fillna(INTERNOW,inplace=True)



# Рисуем, что получилось

df['interval'].hist(bins=50, legend=True)

df['interNow'].hist(bins=50, legend=True)



plt.title('Распределение интервала между отзывами')

plt.xlabel('Дни')

plt.ylabel('Количество ресторанов')
# Отзывы переводим в нижний регистр и анализируем встречающиеся слова - сравниваем со слвварем позитивных и негативных слов

df['Reviews'] = df['Reviews'].apply(lambda x: x.lower())

w_dict = pd.read_csv(path+'Dict.csv',sep=';',index_col='word')              # Считывае словарь хороших и плохих слов



w_dict['mark'] = w_dict['mark'].apply(lambda x: POS if x=='positive' else NEG) 

df['mark']=0

for w in w_dict.index:

    df.loc[(df['Reviews'].str.contains(w)) & ~(df['Reviews'].str.contains('not')),'mark'] += w_dict.loc[w]['mark']   # Анализируем отзывы (у негативных вес больше)



df.mark.hist(bins=10)

plt.title('Распределение тональности положительных и негатиных отзывов')

plt.xlabel('Суммарная тональность')

plt.ylabel('Количество ресторанов')
# Подгрузим данные из Википедии по количесву жителей в европейских городах и подтянрем данные к нашему датасету

dfCities = pd.read_csv(path+'Cities_plus.csv', sep=';')

df = df.merge(dfCities, how='left')



# Заполняем пропуски в количестве отзвов средним погороду

Num_of_Rev = df[~df['Number of Reviews'].isna()][['City','Number of Reviews']].groupby(by='City').median()



df['Number of Reviews'] = df.apply(

    lambda x: Num_of_Rev.loc[x.City]['Number of Reviews'] \

    if isnan(x['Number of Reviews']) \

    else x['Number of Reviews'], axis=1)



# Взвесим на количество жителей

#df['Num_of_Rev'] = (df['Number of Reviews']/df['Population']) * 10000



# И логарифимируем, что бы уйти от слишком большого разброса

# df['Num_of_Rev'] = df['Num_of_Rev'].apply(math.log)

# df['Num_of_Rev'].hist(bins=50)





df['Number of Reviews']=(df['Number of Reviews']-df['Number of Reviews'].min())/(df['Number of Reviews'].max()-df['Number of Reviews'].min())
Cities_best_rating = df[df['sample']==1][['City','Rating']].groupby(by='City').mean().sort_values('Rating',ascending=False)

df['City_rating'] = df.apply(lambda x: Cities_best_rating.loc[x.City],axis=1)



# # Создаем признак рейтинга сети

# rate_by_net = df[df['sample']==1][['Restaurant_id','Rating']].groupby(by='Restaurant_id').mean()

# df['Net_Rating'] = df.apply(lambda x: rate_by_net.loc[x.Restaurant_id] if x.net_size >= NET_SIZE else 0, axis=1) 

# Заполнение пропусков в поле 'Cuisine Style'.

# Работаем только с строками, где 'Cuisine Style' заполнено (в датасете dx).

dx = pd.read_csv(path+'main_task.csv')

dx = dx[~dx['Cuisine Style'].isna()].copy()

dx['Cuisine Style'] = dx['Cuisine Style'].apply(simple_list)



# Добавляем поле из количества различных стилей в каждом ресторане

dx.insert(3,'num_of_cuis', dx['Cuisine Style'].apply(lambda x: len(x)))



# Разносим каждый из стилей кухонь по столбцам

list_to_columns (dx,'Cuisine Style')



# Группируем датасет по городам и формируем для каждого города наиболее харакерные стили

Cities = pd.DataFrame(dx.groupby('City').sum())

Cities.drop(['Ranking','Rating','Number of Reviews'], axis=1, inplace=True)



Cities.insert(1, 'num_of_rest', dx['City'].value_counts())                          # Количество ресторанов в каждом городе

Cities.insert(2, 'cuis_per_rest', Cities['num_of_cuis']/Cities['num_of_rest'])      # Среднее количество стилей в ресторанах

Cities['cuis_per_rest'] = Cities['cuis_per_rest'].apply(lambda x: int(round(x,0)))



Cities.insert(3, 'all_uniq_cuis', Cities.apply(make_cuis_list, axis=1, args=[Cities,3]))  # Список уникальных стилей

Cities.insert(4, 'num_of_uniq_cuis', Cities['all_uniq_cuis'].apply(lambda x: len(x)))     # Количество уникальных стилей

Cities.insert(5, 'most_pop_cuis', Cities.apply(make_pop_cuis, axis=1, args=[Cities,6]))   # Топ самых популярных стилей



# Фиксируем рестораны с пропусками в поле Cuisine Style

df['nan_in_CS'] = df['Cuisine Style'].apply(lambda x: 1 if x!=x else 0) 



# Заполняем отсутствующие данные в боевом датасете df стилями, наиболее характерными для каждого города

df['Cuisine Style'] = df.apply(

    lambda x: Cities.loc[x.City]['most_pop_cuis'] \

    if  isnan(x['Cuisine Style']) \

    else simple_list(x['Cuisine Style']), axis=1)



# Создаем поле, характеризующее количество наиболее популярных стилей в городе в меню конкретного ресторана

#df['num_of_top_cuis'] = df.apply(

#    lambda x: len(list(set(x['Cuisine Style']) & set(Cities.loc[x.City]['most_pop_cuis']))), axis=1)

    



# В заполненных ранее пропусках будут сидеть все популярные стили, поэтому мы уменьшаем баллы для таких ресторанов

#df['num_of_top_cuis'] = df.apply(

#    lambda x: MINUS if x['nan_in_CS']==1 else x['num_of_top_cuis'], axis=1)



df.drop(['nan_in_CS'], axis=1, inplace=True)





df['num_of_cuis'] = df['Cuisine Style'].apply(lambda x: len(x))
#df.num_of_top_cuis.hist(bins=6)
# Переводим города в dummy

dummy_City = pd.get_dummies(df['City'])

df = df.join(dummy_City,how='inner')



# Переводим самые популярные кухни в dummy

collection = pd.Series(make_collection(df,'Cuisine Style'))

top_cuisins = collection.sort_values(ascending=False).head(TOPCUIS).index.to_list()

for cuisin in top_cuisins:

    df[cuisin] = df['Cuisine Style'].apply(lambda x: 1 if cuisin in x else 0)



# Создадим признак по количеству в ресторане наиболее популярных стилей кухонь

#df['num_of_top_cuis_ever'] = df['Cuisine Style'].apply(lambda x: len(list(set(x) & set(top_cuisins))))
max_population = df['Population'].max()

df['Population'] = df['Population']/max_population
plt.rcParams['figure.figsize'] = (20,15)

sns.heatmap(df[df.columns[:20]].corr())
df.head()
df.info()
# Убираем все, кроме цифровых признаков

object_columns = [s for s in df.columns if df[s].dtypes == 'object']

df.drop(object_columns, axis = 1, inplace=True)
# Теперь выделим тестовую часть

# Х - данные с информацией о ресторанах, у - целевая переменная (рейтинги ресторанов)



train_data = df.query('sample == 1').drop(['sample'], axis=1)

test_data  = df.query('sample == 0').drop(['sample'], axis=1)



y = train_data.Rating.values                                   # наш таргет

X = train_data.drop(['Rating','Capital'], axis=1)
display (X.sample(5))

display (X.info())

display (len(y))
# Импортируем необходимые библиотеки:

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели

from sklearn.model_selection import train_test_split

# Загружаем специальный инструмент для разбивки:  

from sklearn.model_selection import train_test_split  

  



    

# Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.  

# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных

# выделим 20% данных на валидацию (параметр test_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED)
# Создаём модель (НАСТРОЙКИ НЕ ТРОГАЕМ)

regr = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)
# Обучаем модель на тестовом наборе данных

regr.fit(X_train, y_train)
# в RandomForestRegressor есть возможность вывести самые важные признаки для модели

plt.rcParams['figure.figsize'] = (10,10)

feat_importances = pd.Series(regr.feature_importances_, index=X.columns)

feat_importances.nlargest(15).plot(kind='barh')
# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

y_pred = regr.predict(X_test)
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
feat_importances.sort_values(ascending=False).head(20)
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
X_test
predict_submission = regr.predict(X_test)
predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)