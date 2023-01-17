import numpy as np

import pandas as pd

from pandas import DataFrame

import datetime as dt

import matplotlib as mpl

import matplotlib.pyplot as plt

import os



!cp ../input/lowess/lowess.py ./

import lowess as lo
# Preprocess the data

# Create additional DataFrame with time only without date



df_source = pd.read_csv("../input/periodic-traffic-data/periodic_traffic.csv")

df_source['rep_date'] = pd.to_datetime(df_source['_time'])

df_source.drop(['_time'], axis=1, inplace=True)



df_source_time = df_source.copy()

df_source_time['rep_time'] = df_source_time['rep_date'].apply(lambda x: dt.datetime.strptime(x.strftime('%H:%M'),'%H:%M'))

df_source_time.drop(['rep_date'], axis=1, inplace=True)



df_source=df_source.set_index('rep_date')

df_source_time=df_source_time.set_index('rep_time')



print("Rows found in the DataFrame:\n{}\n".format(len(df_source.index)))

display(df_source.tail(3))

display(df_source_time.tail(3))
df_source_time['C9'].plot()

plt.show()
#Задаем необходимые переменные

#Window для расчета скользящего стандартного отклонения берем как, например, 1/12 от периода



v_window=8 #скользящее окно

k_out=1.5 #Коэффициент для умножения на std для расчета границ фильтрации выбросов для первого прогона

k_norm=1.5 #Коэффициент для умножения на std для расчета границ "нормального" трафика для второго прогона



i=df_source_time.index.shape[0]

x=np.linspace(-10,10,i)



#Вспомогательная функция для отсечения значений, выходящих за границы фильтрации выборосов



def f_out(x):

    name=x.index[0]

    if x[name] > x[name+'_lo']+k_out*x[name+'_std_first_step']:

        x[name+'_adj']=np.nan

    elif x[name] < x[name+'_lo']-k_out*x[name+'_std_first_step']:

        x[name+'_adj']=np.nan

    else:

        x[name+'_adj']=x[name]

    return x



#Функция для обработки данных.

#На вход функции подается объект Series из исходных данных.

#На выходе получаем данные с отсеченными выбросами ['lo'] и со стандартным отколонением для обработанных данных ['std'].



def f_low(df_x):

    df_res=DataFrame(df_x)

    name=df_res.columns[0]

    i=df_x.index.shape[0]

    x=np.linspace(-10,10,i)

    df_res[name+'_lo'] = lo.lowess(x, df_x.values, x)

    df_res[name+'_std_first_step'] = df_x.rolling(window=v_window,min_periods=0).std().fillna(method='bfill').shift(-int(v_window/2))

    df_res=df_res.apply(f_out,axis=1)

    df_res[name+'_adj_first_step']=df_res[name+'_adj'].fillna(method='bfill')

    df_res[name+'_adj'] = lo.lowess(x, np.array(df_res[name+'_adj_first_step']), x)

    df_res[name+'_std'] = df_res[name+'_adj_first_step'].rolling(window=v_window,min_periods=0).std().fillna(method='bfill').shift(-int(v_window/2))

    return df_res



l=list(df_source_time.columns)

print( "Список полученных для анализа фич:\n{}".format(l) )



for name in l:



    df=f_low(df_source_time[name].sort_index(axis=0))

    display(df.head())



    fig,ax = plt.subplots(1,figsize=(12,9))

    ax.plot(df[name],'b.',label='Original') #исходный график

    ax.plot(df[name+'_lo']+k_out*df[name+'_std_first_step'],'g',label='Границы фильтрации выбросов') #Верхняя граница для фильтрации выборосов

    ax.plot(df[name+'_lo']-k_out*df[name+'_std_first_step'],'g',label='Границы фильтрации выбросов') #Нижняя граница для фильтрации выборосов

    ax.plot(df[name+'_lo'],'r', label='Восстановленный график на первом шаге') #Восстановленный график методом lowess на первом шаге

    ax.plot(df[name+'_adj']+k_norm*df[name+'_std'],'k', label='Верхняя граница нормального трафика') #Верхняя граница нормального трафика

    ax.plot(df[name+'_adj']-k_norm*df[name+'_std'],'k', label='Нижняя граница нормального трафика') #Нижняя граница нормального трафика

    ax.plot(df[name+'_adj'],'y', label='Восстановленный график на втором шаге') #Восстановленный график методом lowess на втором шаге

    ax.set_title(name)

    plt.legend()

    plt.show()
for name in ['NAKA']:



    df=f_low(df_source_time[name].sort_index(axis=0))

    

    df[name+'_adj_avg'] = DataFrame(df[name+'_adj'].groupby(level=0).mean())

    

    display(df.head())



    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(24,9))

    ax1.plot(df[name],'b.',label='Original') #исходный график

    #ax1.plot(df[name+'_lo']+k_out*df[name+'_std_first_step'],'g',label='Границы фильтрации выбросов') #Верхняя граница для фильтрации выборосов

    #ax1.plot(df[name+'_lo']-k_out*df[name+'_std_first_step'],'g',label='Границы фильтрации выбросов') #Нижняя граница для фильтрации выборосов

    #ax1.plot(df[name+'_lo'],'r', label='Восстановленный график на первом шаге') #Восстановленный график методом lowess на первом шаге

    ax1.plot(df[name+'_adj_avg']+k_norm*df[name+'_std'],'k', label='Верхняя граница нормального трафика') #Верхняя граница нормального трафика

    ax1.plot(df[name+'_adj_avg']-k_norm*df[name+'_std'],'k', label='Нижняя граница нормального трафика') #Нижняя граница нормального трафика

    #ax1.plot(df[name+'_adj'],'y', label='Восстановленный график на втором шаге') #Восстановленный график методом lowess на втором шаге

    ax1.plot(df[name+'_adj_avg'],'r', label='Восстановленный график на втором шаге') #Восстановленный график методом lowess на втором шаге

    ax1.set_title(name)

    ax1.legend()

    ax2.plot(df_source[name])

    ax2.legend()

    plt.show()
df_s1=df_source.copy()

df_s1['rep_time']=df_source.index.values

df_s1['rep_time']=df_s1['rep_time'].apply(lambda x: dt.datetime.strptime(x.strftime('%H:%M'),'%H:%M'))

df_s2=pd.merge(df_s1,df,how='left',left_on='rep_time',right_index=True)

display(df_s2.head())

df_s2['lower'] = df_s2['NAKA_adj_avg']-df_s2['NAKA_std']*k_norm

df_s2['upper'] = df_s2['NAKA_adj_avg']+df_s2['NAKA_std']*k_norm

df_s2[['NAKA_x','NAKA_adj_avg','lower','upper']].plot(figsize=(12,9))

plt.show()