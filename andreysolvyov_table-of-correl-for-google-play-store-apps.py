#цель - построить функцию, которая составляет таблицу корреляции фичей

#перед тем, как подавать в функцию данные важно, чтобы все значения в таблице были однотипные и небыло пустых ячеек

#работа функции: переводит все данные в числа и считает корреляцию

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import math

import datetime

def date_normalize(s):

    s = s.replace(',', '').split(' ')

    s[0] = ['January','February','March','April','May','June','July','August','September','October','November','December'].index(s[0]) + 1

    return int((datetime.datetime(2019, 7, 27) - datetime.datetime(int(s[2]), int(s[0]), int(s[1]))).days) 

def normalize_cur_ver(s):

    if s != 'Varies with device':

        a = (''.join(filter(lambda x: (x.isdigit() or (x == '.')), s)) + '.0').replace('..','.').replace('³.0', '3')

        return float(a[:a.find('.') + 2])

    else:

        return 'Varies with device'

def si(s):

    if s[-1] == 'M':

        return float(s[:-1])*1024

    elif s[-1] == 'k':

        return float(s[:-1])

    else:

        return 0

print(os.listdir("../input"));
#избавляемся от мусора

df = pd.read_csv('../input/googleplaystore.csv')

df = df.drop(df[df['Rating'] == 19].index, axis = 0)

df['Size in KB'] = df['Size'].apply(si)

df = df.drop('Size', axis = 1)

df = df.drop('App', axis = 1)

df = df.drop('Content Rating', axis = 1)
head = str(df.head(0).T)

head = head[head.index('Category'):-1].split(',')

head = [head[i].strip() for i in range(len(head))]

inf = pd.DataFrame.from_items({head.index(i): [i, type(df[i][0]), round(100*(len(df[i].value_counts())/len(df)))] for i in head}.items(), 

    orient='index', 

    columns=['In head','type','Percent of uniques'])

#чтобы перевести все данные в числа нужно их либо разбить на категории либо удалить, потому что, очевидно, бессмысленно искать корреляцию между названием приложения и чем либо еще

#можно разбить на категории или нет определим по уникальности данных - если все данные разные, значит, это точно не категория, иначе - категория
#проверим, однотипные ли данные, и, если нет - найдем номер строки, где они сидят,а заодно и узнаем сколько выделившихся данных и можем ли мы ими пренебречь

c = 0

for i in range(len(inf['In head'])):

    true = True

    a = -1

    for j in df[inf['In head'][i]]:

        a = a + 1

        true = true*(type(j) == inf['type'][i])

        if (not true):

            c = c + 1

            print()

            print('номер строки: ', a)

            print('значение: ', j)

            print('столбец: ', inf['In head'][i])

            print('тип: ', type(j))

            print('ожидался тип: ', inf['type'][i])

            print('проверяем: ',  j,   df[inf['In head'][i]][a])

            print('проверка:',    j == df[inf['In head'][i]][a])

            true = True

c
#видно, что перепутались классы <class 'float'> и <class 'numpy.float64'> - это не проблема, а так же кое где встречаются неопределенные значения. Проверим, много ли неопределенных значений

#а заодно и переведем наши данные в <class 'float'>

c = 0

for i in range(len(inf['In head'])):

    true = True

    a = -1

    for j in df[inf['In head'][i]]:

        a = a + 1

        true = true*(type(j) == inf['type'][i])

        if (not true) and (inf['type'][i] != np.float64):

            c = c + 1

            print()

            print('номер строки: ', a)

            print('значение: ', j)

            print('столбец: ', inf['In head'][i])

            print('тип: ', type(j))

            print('ожидался тип: ', inf['type'][i])

            true = True

df['Reviews'] = df['Reviews'].astype(int)

df['Size in KB'] = df['Size in KB'].astype(float)

100*c/len(df)
#обновим инфо

inf = pd.DataFrame.from_items({head.index(i): [i, type(df[i][0]), round(100*(len(df[i].value_counts())/len(df)))] for i in head}.items(), 

    orient='index', 

    columns=['In head','type','Percent of uniques'])

inf

#как оказалось, неопределенных значений всего 0.1%, поэтому мы можем ими пренебречь, удалим эти строки:

indexes_to_drop = []

for i in range(len(inf['In head'])):

    true = True

    a = -1

    for j in df[inf['In head'][i]]:

        a = a + 1

        true = true*(type(j) == inf['type'][i])

        if (not true) and (inf['type'][i] != np.float64) and (inf['type'][i] != np.int64):

            indexes_to_drop = indexes_to_drop  + [a]

            true = True

indexes_to_keep = set(range(df.shape[0])) - set(indexes_to_drop)

df_sliced = df.take(list(indexes_to_keep))
#проверим, все ли мы сделали правильно, если да, то ответ должен = 0

c = 0

for i in range(len(inf['In head'])):

    true = True

    a = -1

    for j in df_sliced[inf['In head'][i]]:

        a = a + 1

        true = true*(type(j) == inf['type'][i])

        if (not true) and (inf['type'][i] != np.float64)  and (inf['type'][i] != np.int64):

            c = c + 1

            true = True

100*c/len(df)
#мы все сделали правильно, поэтому смело обновляем наш датасет

df = df_sliced

#почти все данные, кроме Current Ver и Last Updated готовы к тому, чтобы перевести их в числа.

inf
#решим вопрос с последним обновлением - посчитаем сколько дней назад было обновление 

df['Last Update in delta days'] = df['Last Updated'].apply(date_normalize)

df = df.drop('Last Updated', axis = 1)

#обновим инфо

head = str(df.head(0).T)

head = head[head.index('Category'):-1].split(',')

head = [head[i].strip() for i in range(len(head))]

inf = pd.DataFrame.from_items({head.index(i): [i, type(df[i][0]), round(100*(len(df[i].value_counts())/len(df)))] for i in head}.items(), 

    orient='index', 

    columns=['In head','type','Percent of uniques'])

inf
#осталось самое сложное - обработать Current Ver

#посмотрим, какие бывают значения без точек, с несколькими точками, посторонними символами - не точками и не числами, убирем из рассмотрения Varies with device

for i in df['Current Ver']:

    if ((i.count('.') > 1) or (i.count('.') == 0) or (not(i.replace('.', '').isdigit()))) and (not (i == 'Varies with device')):

        print(i)
#все не так уж и плохо, как могло показаться - для начала уберем из датасета все версии, в которых нет чисел - по таким данным нельзя определить версию:

a = -1

indexes_to_drop = []

for i in df['Current Ver']:

    a = a + 1

    if df['Current Ver'][df['Current Ver'].index[a]].isalpha() and (df['Current Ver'][df['Current Ver'].index[a]] != 'Varies with device'):

        indexes_to_drop = indexes_to_drop  + [a]

indexes_to_keep = set(range(df.shape[0])) - set(indexes_to_drop)

df_sliced = df.take(list(indexes_to_keep))
#перепишем наш датасет:

df = df_sliced

df = df.drop(df[df['Current Ver'] == 'Human Dx'].index, axis = 0)

df = df.drop(df[df['Current Ver'] == 'DH-Security Camera'].index, axis = 0)

df = df.drop(df[df['Current Ver'] == 'App copyright'].index, axis = 0)

df = df.drop(df[df['Current Ver'] == 'Natalia Studio Development'].index, axis = 0)

df['New Current Ver'] = df['Current Ver'].apply(normalize_cur_ver)

df['New Android Ver'] = df['Android Ver'].apply(normalize_cur_ver)

df = df.drop('Current Ver', axis = 1)

df = df.drop('Android Ver', axis = 1)

#обновим инфо

head = str(df.head(0).T)

head = head[head.index('Category'):-1].split(',')

head = [head[i].strip() for i in range(len(head))]

inf = pd.DataFrame.from_items({head.index(i): [i, type(df[i][0]), round(100*(len(df[i].value_counts())/len(df)))] for i in head}.items(), 

    orient='index', 

    columns=['In head','type','Percent of uniques'])

inf
#датасет готов к обработке! осталось только закатегорить столбцы(а оставшиеся столбцы типа <class 'str'> категорить можно, потому что уникальных среди них не более 2%)

for i in range(len(inf)):

    if not(inf['type'][i] in [np.float64, np.int64, float]):

        df['New ' + inf['In head'][i]] = df[inf['In head'][i]].map({pd.unique(df[inf['In head'][i]])[j]: j for j in range(len(pd.unique(df[inf['In head'][i]])))})

#обновим инфо

head = str(df.head(0).T)

head = head[head.index('Category'):-1].split(',')

head = [head[i].strip() for i in range(len(head))]

inf = pd.DataFrame.from_items({head.index(i): [i, type(df[i][0]), round(100*(len(df[i].value_counts())/len(df)))] for i in head}.items(), 

    orient='index', 

    columns=['In head','type','Percent of uniques'])

inf
dfn = df[['Rating', 'Reviews', 'Size in KB', 'Last Update in delta days', 'New Current Ver', 'New Android Ver', 'New Category', 'New Installs', 'New Type', 'New Price', 'New Genres']]

#было бы неплохо рассмотреть отдельные датасеты с участием и без Varies with device, но я рассмотрю только без

dfn_varnot = dfn[(dfn['New Current Ver'] != 'Varies with device') & (dfn['New Android Ver'] != 'Varies with device')]

head = str(dfn_varnot.head(0).T)

head = head[head.index('Rating'):-1].split(',')

head = [head[i].strip() for i in range(len(head))]

infn = pd.DataFrame.from_items({head.index(i): [i, type(dfn_varnot[i][0]), round(100*(len(dfn_varnot[i].value_counts())/len(dfn_varnot)))] for i in head}.items(), 

    orient='index', 

    columns=['In head','type','Percent of uniques'])

infn
#наконец, долгожданная таблица

dfn_varnot.corr(method='pearson')
