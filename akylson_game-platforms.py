# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Подключим необходимые библиотеки для проведения анализа

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import numpy as np

from scipy import stats as st
# Назначим к переменной df наш датасет games.csv

dataset_filepath = "/kaggle/input/games.csv" # Путь к файлу с данными

df = pd.read_csv(dataset_filepath)

sns.set_style("darkgrid")
#Выведем полностью наш датасет

df
# Посмотрим информацию о нашей таблице

df.info()
#Посмотрим какие столбцы у нас имеются и все ли с ними впорядке

df.columns
#Посчитаем количество пустых значений

df.isna().sum()
#Посмотрим на таблицу, где имеются пустые значения в столбце critic_score

df[df['Critic_Score'].isna()]
#Посмотрим на таблицу, где имеются пустые значения в столбце user_score

df[df['User_Score'].isna()]
#Посмотрим на таблицу, где имеются пустые значения в столбце rating

df[df['Rating'].isna()]
#Посмотрим на таблицу, где имеются пустые значения в столбце year_of_release

df[df['Year_of_Release'].isna()]
#Посмотрим какие платформы для игры у нас имеются

df['Platform'].value_counts()
#Посмотрим какие жанры игр мы имеем и нет ли повторений

df['Genre'].value_counts()
#Посчитаем количество дубликатов

df.duplicated().sum()
# Заменим названия столбцов (приведем к нижнему регистру) 

df.columns = df.columns.str.lower()

df.columns
# Приведем к нижнему регистру следующие столбцы:

for column in df[['name','platform','genre','rating']]:

    df[column] = df[column].str.lower()
# Изменим тип данных в столбце year_of_release 

df['year_of_release'] = df['year_of_release'].astype('Int64')
#Посмотрим как выглядит теперь наша таблица

df.sample(20)
# Посчитаем количество пустых значений в столбце year_of_release

df['year_of_release'].isna().sum()
#Напишем цикл где будем делать замену на года тех у кого совпадает название с игрой но в графе year_of_release пусто

for i in df[df['year_of_release'].isnull() == True].index:  

    df['year_of_release'][i] = df.loc[df['name'] == df['name'][i], 'year_of_release'].max()
#Посмотрим сколько удалось заполнить пустых значений

df['year_of_release'].isna().sum()
#Заменим tbd на Nan

df['user_score'] = df['user_score'].replace('tbd', np.nan, regex=True)
# Поменяем формат столбца user_score на float

df['user_score'] = df['user_score'].astype(float)

df['user_score'].dtype
#Создадим новый столбец total_sales и прибавим продажи всех столбцов

df['total_sales'] = df['na_sales'] + df['eu_sales'] + df['jp_sales'] + df['other_sales']
df
# Методом пивот отсортируем таблицы и отрисуем график, чтобы просмотреть как менялось количество выпускаемых игр

games_on_period = df.pivot_table(index='year_of_release', values='name', aggfunc='count')

plt.figure(figsize=(12,6))

sns.lineplot(data=games_on_period)

plt.title("Количество игр выпускаемые в разные года")

plt.xlabel("Год выпуска")

plt.ylabel("Количество выпущенных игр")

plt.legend('')
platform_on_sales = df.pivot_table(

    index='platform', values='total_sales', aggfunc='sum').sort_values(by='total_sales', ascending=False)



plt.figure(figsize=(13,6))

sns.barplot(x=platform_on_sales.index,y=platform_on_sales['total_sales'])

plt.title("Продажи по платформам за весь период")

plt.xlabel("Название платформы")

plt.ylabel("Количество продаж")
# Напишем функцию, которая будет возвращать нужную сводную таблицу и выводить данные с 2005 года

def year_total_sale_for_platform(name, data):

    slicee = data[(data['platform'] == name) & (data['year_of_release'] > 2005)]

    total = slicee.pivot_table(index='year_of_release', values='total_sales', aggfunc='sum').sort_values('year_of_release', ascending=False)

    return total
# Создадим свою таблицу по платформам и их обшим продажам. отсортируем их по убыванию и оставим только топ 5.

top_5_platforms = df.pivot_table(index='platform', values='total_sales', aggfunc='sum').sort_values(by='total_sales', ascending=False).head(5)

top_5_platforms = top_5_platforms.reset_index().rename_axis(None, axis=1)
#Выведем топ 5 продаваемых платформ

top_5_platforms
#Отрисуем все игровые платформы и их поведение за последние 10 лет

plt.figure(figsize=(12,6))

plt.title('Количество продаж популярных игровых платформ')

plt.xlabel('Годы игровых релизов')

plt.ylabel('Продажи')



for i in list(top_5_platforms['platform']):

    sns.lineplot(data=year_total_sale_for_platform(i,df)['total_sales'], label=i)

    plt.legend()
#Сохраним в переменной df_top_5_platforms топ 5 платформ и избавимся от выбросов

list_of_top5 = ['ps2','x360','ps3','wii','ds']

df_top_5_platforms = df[df['platform'].isin(['ps2','x360','ps3','wii','ds'])]

df_top_5_platforms = df_top_5_platforms[df_top_5_platforms['total_sales']<1.4]
df_top_5_platforms['total_sales'].describe()
#Отрисуем ящики с усами 

plt.figure(figsize=(12,6))

sns.boxplot(data=df_top_5_platforms, x='platform', y='total_sales')

plt.title('Ящик с усами', fontsize=15)

plt.xlabel('Платформа', fontsize=12)

plt.ylabel('Глобальные продажи',fontsize=12)
#Корреляция между оценками пользователей и продажами 

sony_play_station2 = df[df['platform']=='ps2']

sony_play_station2['user_score'].corr(sony_play_station2['total_sales'])
#Построим диаграмму рассеяния

plt.figure(figsize=(12,6))

sns.scatterplot(x='user_score', y='total_sales', data=sony_play_station2)

plt.title('test')
#Построим диаграмму рассеяния по оценкам критиков 

plt.figure(figsize=(12,6))

sns.scatterplot(x='critic_score', y='total_sales', data=sony_play_station2)

plt.title('test')
#Корреляция между оценкой критиков и продажам

sony_play_station2['critic_score'].corr(sony_play_station2['total_sales'])
#Напишем функцию, которая будет отрисовывать графики рассеивания и считать корреляции

def other_platform_matrix(name_of_platform):

    platform = df[df['platform']==name_of_platform]

    fig, ax = plt.subplots(1 ,2, figsize=(15,5))

    sns.scatterplot(x='user_score', y='total_sales', data=platform, ax=ax[0])

    sns.scatterplot(x='critic_score', y='total_sales', data=platform, ax=ax[1])

    fig.suptitle(name_of_platform, fontsize=15)

    ax[0].set(xlabel='Оценка пользователей')

    ax[1].set(xlabel='Оценка критиков')

    ax[0].set(ylabel='Количество продаж')

    ax[1].set(ylabel='Количество продаж')

    plt.show()

    

    correl = platform['user_score'].corr(platform['total_sales'])

    critic_correl = platform['critic_score'].corr(platform['total_sales'])

    

    

    if 0.3 >= critic_correl >= 0.1:

        print('Корреляция между отзывами критиков и игровой платформой ', name_of_platform.upper(), ': Слабая', critic_correl)

    if 0.5 >= critic_correl >= 0.3:

        print('Корреляция между отзывами критиков и игровой платформой ', name_of_platform.upper(), ': Умеренная', critic_correl)

    if 0.7 >= critic_correl >= 0.5:

        print('Корреляция между отзывами критиков и игровой платформой ', name_of_platform.upper(), ': Высокая', critic_correl)

    if 0.9 >= critic_correl >= 0.7:

        print('Корреляция между отзывами критиков и игровой платформой ', name_of_platform.upper(), ': Весьма высокая', critic_correl)

    if 1 >= critic_correl >= 0.9:

        print('Корреляция между отзывами критиков и игровой платформой ', name_of_platform.upper(), ': Сильная', critic_correl)

    

    if 0.3 >= correl >= 0.1:

        print('Корреляция между отзывами пользователей и продажами ', name_of_platform.upper(), ': Слабая', correl)

    if 0.5 >= correl >= 0.3:

        print('Корреляция между отзывами пользователей и продажами ', name_of_platform.upper(), ': Умеренная', correl)

    if 0.7 >= correl >= 0.5:

        print('Корреляция между отзывами пользователей и продажами ', name_of_platform.upper(), ': Высокая', correl)

    if 0.9 >= correl >= 0.7:

        print('Корреляция между отзывами пользователей и продажами ', name_of_platform.upper(), ': Весьма высокая', correl)

    if 1 >= correl >= 0.9:

        print('Корреляция между отзывами пользователей и продажами ', name_of_platform.upper(), ': Сильная', correl)

    print('\n')
#С помощью цикла выведем все 5 графиков

for platform in list_of_top5:

    other_platform_matrix(platform)
#Посчитаем дисперсию, стандартное отклонение, среднее и медиану у топ 5 платформ к оценкам пользователей

for platform in list_of_top5:

    print('Дисперсия', platform.upper(),':', np.var(df[df['platform']==platform]['user_score']))

    print('Стандартное отклонение', platform.upper(),':', np.std(df[df['platform']==platform]['user_score']))

    print('Среднее',platform.upper(),':',  df[df['platform']==platform]['user_score'].mean())

    print('Медиана',platform.upper(),':',  df[df['platform']==platform]['user_score'].median())

    print('\n')
#Посчитаем дисперсию, стандартное отклонение, среднее и медиану у топ 5 платформ к оценкам критиков

for platform in list_of_top5:

    print('Дисперсия', platform.upper(),':', np.var(df[df['platform']==platform]['critic_score']))

    print('Стандартное отклонение', platform.upper(),':', np.std(df[df['platform']==platform]['critic_score']))

    print('Среднее',platform.upper(),':',  df[df['platform']==platform]['critic_score'].mean())

    print('Медиана',platform.upper(),':',  df[df['platform']==platform]['critic_score'].median())

    print('\n')
# Методом сводных таблиц выведем жанры и их продажи.  отсортируем по убыванию. 

distr_genre = df.pivot_table(

    index='genre', values='total_sales', aggfunc='sum').sort_values(by='total_sales', ascending=False)

distr_genre = distr_genre.reset_index().rename_axis(None, axis=1)

distr_genre
#Отрисуем барплот чтобы наглядно посмотреть какие жанры лидирует, а какие остаются внизу

plt.figure(figsize=(12,6))

plt.title('Распределение игр по жанрам ',fontsize=15)

sns.barplot(data=distr_genre, x='genre', y='total_sales')

plt.xlabel('Жанры игр',fontsize=12)

plt.ylabel('Продажи',fontsize=12)
#Напишем функции для создания сводных таблиц и отсривоки барплотов



#Функция для создания сводбных таблиц за весь период

def forpivot(row, title):

    fig, axes = plt.subplots(1, 3, figsize=(20, 4))

    for pivot, ax in zip(list(['platform','genre','rating']),axes.flatten()[:3]):

        ppivot = df.pivot_table(index=pivot, values=row, aggfunc='sum'

                  ).sort_values(by=row, ascending=False).reset_index().rename_axis(None, axis=1).head(5)

        print(ppivot)

        print('\n\n') 

        sns.set_palette("Blues")

        sns.barplot(data=ppivot, x=pivot, y=row, ax=ax)

        fig.suptitle(title, fontsize=15)

        



plt.show()

sns.set()



#Функция для создания сводных таблиц за последний год

def for_pivot_2016(row, title):

    temp = df[df['year_of_release']>2015]

    fig, axes = plt.subplots(1, 3, figsize=(20, 4))

    for pivot, ax in zip(list(['platform','genre','rating']), axes.flatten()[:3]):

        ppivot = temp.pivot_table(index=pivot, values=row, aggfunc='sum').sort_values(by=row, ascending=False).reset_index().rename_axis(None, axis=1).head(5)

        print(ppivot)

        print('\n\n')

        sns.set_palette("BuGn_r")

        sns.barplot(data=ppivot, x=pivot, y=row, ax=ax)

        fig.suptitle(title, fontsize=15)

        

#Выведем топ 5 платформ, жанров и рейтингов за весь период

forpivot('na_sales','Топ 5 платформ, жанров и рейтингов за весь период')
#Выведем топ 5 платформ, жанров и рейтингов за последний год

for_pivot_2016('na_sales','Топ 5 платформ, жанров и рейтингов за последний год')
#Выведем топ 5 платформ, жанров и рейтингов за весь период для Европейского союза

forpivot('eu_sales','Топ 5 платформ, жанров и рейтингов за весь период для Европейского союза')
#Выведем топ 5 платформ, жанров и рейтингов за последний год для европейского союза

for_pivot_2016('eu_sales','Топ 5 платформ, жанров и рейтингов за последний год для европейского союза')
#Выведем топ 5 платформ, жанров и рейтингов за весь период для Японии

forpivot('jp_sales','Топ 5 платформ, жанров и рейтингов за весь период для Японии')
#Выведем топ 5 платформ, жанров и рейтингов за последний год для японии

for_pivot_2016('jp_sales','Топ 5 платформ, жанров и рейтингов за последний год для японии')
# Сохраним в переменных xbox_hyp и pc_hyp соответствующие данные (актуальные данные за последние 10 лет)

xone_hyp = df[(df['platform']=='xone') & (df['year_of_release']>2006)]['user_score']

pc_hyp = df[(df['platform']=='pc') & (df['year_of_release']>2006)]['user_score']



#Посчитаем средний рейтинг пользователя для xbox платформ

xone_hyp.mean()
#Посчитаем средний рейтинг пользователя для PC платформ

pc_hyp.mean()
#Выполним проверку гипотезы. Будем использовать метод ttest_ind



alpha = .01



results = st.ttest_ind(xone_hyp.dropna(), pc_hyp.dropna(), equal_var=False)



print('p-значение:', results.pvalue)





if (results.pvalue < alpha):

    print("Отвергаем нулевую гипотезу")

else:

    print("Не получилось отвергнуть нулевую гипотезу")
# Сохраним в переменных genre_action_hyp и genre_sports_hyp соответствующие данные с пользовательскими оценками

genre_action_hyp = df[(df['genre']=='action') & (df['year_of_release']>2006)]['user_score']

genre_sports_hyp = df[(df['genre']=='sports') & (df['year_of_release']>2006)]['user_score']



#выведем среднюю оценку по жанру экшн

genre_action_hyp.mean()
#выведем среднюю оценку по жанру спорт

genre_sports_hyp.mean()
#Выполним проверку гипотезы. Будем использовать метод ttest_ind



alpha = .01



results = st.ttest_ind(genre_action_hyp.dropna(), genre_sports_hyp.dropna(), equal_var=False)



print('p-значение:', results.pvalue)





if (results.pvalue < alpha):

    print("Отвергаем нулевую гипотезу")

else:

    print("Не получилось отвергнуть нулевую гипотезу")