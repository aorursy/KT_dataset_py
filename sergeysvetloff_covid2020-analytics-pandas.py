#!/usr/bin/env python
# coding: utf-8

# This work is aimed at practicing the use of the -pandas- etc. library. \
# Many solutions are not optimal. \
# The task I set in this work is to find dependencies that affect the spread of covid-19 * and find the expected horizon for the end of the peak of the pandemic \
# To solve this problem, hypotheses were used **: \
# - the spread of the virus is affected by the country's tourism rating (share in the global volume of tourist traffic) \
# - the country's share in the world trade balance affects the growth of the local pandemic \
# - the country's weight in world oil production / consumption affects the growth of a local pandemic \
# Open sites and statistics of the World Bank act as data sources (links are indicated in the section loaders). \
# Covid-19 data: https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases# \
# \ * - not all conclusions are final and accurate. The data is loaded when this Jupyter notebook is launched! \
# \ ** - not all data may be displayed correctly!

# In[1]:


# ! pip install chart_studio
# !pip install html5lib
# ! pip install bs4
# ! pip install sys
# ! pip install graphviz


# In[2]:


import pandas as pd
import seaborn as sns
import numpy as np
import re as re
import bs4 as bs
import lxml.html as html
import html5lib
import sys
import requests
import matplotlib.pyplot as plt
import cufflinks as cf


# In[3]:


# Настройка графики
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [15, 5]
# plt.rcParams['figure.figsize'] = (15, 5)  # Размер картинок
plt.rcParams['font.family'] = 'sans-serif' # Шрифт
plt.style.use('ggplot')  # Красивые графики

# pd.options.display.max_rows = 10 # Количество строк?


# ## Covid,
# #### данные по статистике погибших от  Института Хопкинса 

# In[4]:


# # Парсим сайт со страницей по медалям и странам:
# # АДрес сайта:
# url = 'https://ru.wikipedia.org/wiki/%D0%9F%D0%B0%D0%BD%D0%B4%D0%B5%D0%BC%D0%B8%D1%8F_COVID-19'

# header = {
#   "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
#   "X-Requested-With": "XMLHttpRequest"
# }
# r = requests.get(url, headers=header)
# # создаем фрейм и делаем обертки для нужных колонок
# Deaths_raw = pd.read_html(url, header=0)[2].iloc[2:, [0,15]].rename(columns={ 'All-time countries statistics':'country', 'All-time countries statistics.15':'medals'})
# Deaths_raw = pd.read_html(url, header=0)
# Deaths_raw.head()
# url_old = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSe-8lf6l_ShJHvd126J-jGti992SUbNLu-kmJfx1IRkvma_r4DHi0bwEW89opArs8ZkSY5G2-Bc1yT/pub?gid=0&single=true&output=csv'
# url = 'https://ourworldindata.org/c6e32fd6-7862-4471-8010-849a5f2b156d'
# url = 'https://github.com/owid/covid-19-data/blob/master/public/data/ecdc/total_deaths.csv'
# Cases_raw = pd.read_csv(url , sep='\t', comment='#')
# Cases_raw.head(20)


# In[5]:


# загрузим данные о подтвержденных случаях covid-19
# источник данных - https://data.humdata.org/
url_old = 'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv&filename=time_series_covid19_confirmed_global.csv'
# url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSe-8lf6l_ShJHvd126J-jGti992SUbNLu-kmJfx1IRkvma_r4DHi0bwEW89opArs8ZkSY5G2-Bc1yT/pub?gid=0&single=true&output=csv'
url = 'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv&filename=time_series_covid19_confirmed_global.csv'
Cases_raw = pd.read_csv(url)

# загружаем данные на текущую дату по количеству умерших:
Cases_raw = Cases_raw.rename(columns={'Country/Region': 'country'})
# удаляем провинции и штаты, так как нас интересует только в целом страна:
Cases_raw = Cases_raw.drop('Province/State', axis=1)
Cases_raw = Cases_raw.rename({'Lat':'X', 'Long':'Y'}, axis=1)

# суммируем строки по странам (убираем разделение на провинции и экстерритории)
data_c = Cases_raw.groupby('country').sum()
data_c = data_c.reset_index()

# Данные получаются из разных источников, поэтому названия стран не совпадают и их приходится корретировать в ручную
#!!! Переименовать страну, если не совпадает с другой таблицей
data_c.country[data_c.country == 'US'] = 'United States'
data_c.country[data_c.country == 'Korea, South'] = 'Korea Republic'
data_c.country[data_c.country == 'Czechia'] = 'Czech Republic'
# нас интересует, какое количество зараженных было 14 дней назад:
data_c = data_c.pop(data_c.columns[-15])
data_c = pd.DataFrame(data_c)


# #### Данные по инфицированным за 14 дней до текущей даты:

# In[6]:


data_c.head()


# In[7]:


# источник данных - https://data.humdata.org/
url = 'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_deaths_global.csv&filename=time_series_covid19_deaths_global.csv'
Deaths_raw = pd.read_csv(url)

# загружаем данные на текущую дату по количеству умерших:
Deaths_raw = Deaths_raw.rename(columns={'Country/Region': 'country'})
# удаляем провинции и штаты, так как нас интересует только в целом страна:
Deaths_raw = Deaths_raw.drop('Province/State', axis=1)
Deaths_raw = Deaths_raw.rename({'Lat':'X', 'Long':'Y'}, axis=1)

# суммируем строки по странам (убираем разделение на провинции и экстерритории)
data_d = Deaths_raw.groupby('country').sum()
data_d = data_d.reset_index()

# Данные получаются из разных источников, поэтому названия стран не совпадают и их приходится корретировать в ручную
#!!! Переименовать страну, чтобы названия стран из разных источников совпадали
data_d.country[data_d.country == 'US'] = 'United States'
data_d.country[data_d.country == 'Korea, South'] = 'Korea Republic'
data_d.country[data_d.country == 'Czechia'] = 'Czech Republic'

# добавляем в таблицу колонку с зараженными -14дней от текущей даты:
# data_c = data_c.fillna(0)
data_d['cases'] = data_c
# считаем долю погибших на текущую дату  от числа зараженных 14 дней назад 
# данная гипотеза основана на том, что срок инкубации >= 14 дней:
data_d['rate_d_c'] = data_d.iloc[:,-2] / data_d.iloc[:,-1]
# переставим колонку Страна на первое место
# data_d = data_d.reindex(columns=([['country'], ['rate_c']] + list([a for a in data_d.columns if (a != 'country' or a != 'rate_c')]) ))
data_d = data_d.reindex(columns=(['country', 'rate_d_c'] + list([a for a in data_d.columns if  a != 'rate_d_c' and a != 'country']) ))
data_d = data_d.drop(['cases', 'X', 'Y'], axis=1)
data_d.sort_values(by=data_d.columns[-1], ascending=False).head(10)


# In[8]:


# проверим наличие повторов по стране
data_d[data_d.country == 'United Kingdom']


# In[9]:


data_d.head()


# #### Посмотрим на графики по датам для топ-10 (страны, где максимум по потерям от COVID)

# In[10]:


# Топ-10
# Ось Х - дата
# Ось У - количество потерь
data_top = data_d.sort_values(by=data_d.columns[-1], ascending=False)
data_top.head()


# In[11]:


data_top.shape


# In[12]:


data_top.iloc[0, 2]


# In[13]:


x = data_top.columns[2:]
x[0]
len(x)


# In[14]:


tmp = data_top.iloc[0, 2+1] - data_top.iloc[0, 2]
# tmp


# In[15]:


# выведем логарифмический график по топ-10 стран по потерям за весь период наблюдений
country_name = data_top.iloc[:20,0]


y = data_top
plt.figure(figsize=(18, 8))
z = []
data_days_num = []
tmp = 0
for ii in range(20):
    for jj in range(2, len(x)):
        tmp = data_top.iloc[ii, jj+1] - data_top.iloc[ii, jj]
        z.append(tmp)
    data_days_num.append(z)
    z = []
# data_days_num = data_days_num.reshape(185,-1)
data_days_num = pd.DataFrame(data_days_num)

x = data_days_num.columns
data_days_num.shape, x.shape


for i in range(len(data_days_num)):
#     plt.plot(x, data_days_num.iloc[i, :], label=country_name)
    plt.plot(x, data_days_num.iloc[i, :], label=country_name)
    
    
plt.legend(country_name)
plt.xlabel('Days', size = 15)
plt.ylabel('Num_death', size = 15)
plt.title('Num death country all times', size = 20)


# In[16]:


data_days_num.head()


# In[17]:


df = data_days_num
# df = data_top
fig = df.iplot(asFigure=True, xTitle="The X Axis",
                    yTitle="The Y Axis", title="The Figure Title")
fig.show()


# In[18]:



# матрица значений для построения графика и взятие логарифма по значениям потерь у ТОП-10
data_graf = pd.DataFrame(data = data_top.iloc[:10,2:])
# список Топ-10 стран по потерям
country_name = data_top.iloc[:10,0]
data_graf.shape


# In[19]:


# для удобства анализа прологарифмируем значения по столбцам
# data_graf_log = np.log(data_graf.iloc[:,:]) / 10
data_graf_log = np.log10(data_graf.iloc[:,:]) / 10

# data_graf_log = np.log(data_graf.iloc[:,:]) / np.log(data_graf.iloc[:,:].max())


# In[20]:


data_graf_log.head()


# In[21]:


# выведем логарифмический график по топ-10 стран по потерям за весь период наблюдений
country_name = data_top.iloc[:,0]

x = data_graf_log.columns
y = data_graf_log
plt.figure(figsize=(18, 8))
for i in range(len(y)):
    plt.plot(x, data_graf_log.iloc[i, :], label=country_name)

plt.legend(country_name)
plt.xlabel('Score', size = 15)
plt.ylabel('log_death', size = 15)
plt.title('Rate death country all times', size = 20)


# In[22]:


# выведем график по топ-10 стран по потерям за 7 дней до текущей даты
x = data_graf_log.columns[-9:]
y = data_graf_log.iloc[:, -9:]
plt.figure(figsize=(18, 8))
for i in range(len(y)):
    plt.plot(x, data_graf_log.iloc[i, -9:], label=country_name)

plt.legend(country_name)
plt.xlabel('Day', size = 20)
plt.ylabel('log_death', size = 20)
plt.title('Rate death country last week', size = 20)


# In[23]:


data_d[data_d['country'] == 'China']


# In[24]:


# выведем график по Корее и Китаю - у них пандемия закончилась ранее всех

xK = data_d.columns[2:]
yK = data_d.iloc[[90], 2:]
yC = data_d.iloc[[36], 2:]

yK_log = np.log10((data_d.iloc[90, 2:]).astype(float)) / 10
yC_log = np.log10((data_d.iloc[36, 2:]).astype(float)) / 10

plt.figure(figsize=(18, 8))
# plt.plot(xK, yK, label=country_name)
plt.plot(xK, yK_log, label = 'Korea Republic')
plt.plot(xK, yC_log, label = 'China')

plt.legend()
plt.xlabel('Day', size = 20)
plt.ylabel('log_death', size = 20)
plt.title('China and Korea Republic Rate death country', size = 20)


# In[25]:


len(data_d.iloc[0])


# In[26]:


data_d.iloc[36,-15:
           ]


# In[27]:


# выведем график по Корее и Китаю - у них пандемия закончилась ранее всех
plt.figure(figsize=(18, 8))
# xK = data_d.columns[2:]
# yK = data_d.iloc[[90], 2:]
# yC = data_d.iloc[[36], 2:]

for i in range(2, len(data_d.iloc[0])-1):
    xK = data_d.columns[i]
#     yK = ((data_d.iloc[90, (i+1)]).astype(float) - (data_d.iloc[90, i]).astype(float))
#     print(yK)
    yC = ((data_d.iloc[36, (i+1)]).astype(float) - (data_d.iloc[36, i]).astype(float))
#     plt.scatter(xK, yK, label = 'Korea Republic')
    plt.scatter(xK, yC, label = 'China')

# plt.legend()
plt.xlabel('Day', size = 20)
plt.ylabel('Num_death', size = 20)
plt.title('China Rate death country', size = 20)


# ### Russia

# In[28]:


# выведем график по топ-10 стран по потерям за 7 дней до текущей даты
country_name = 'Russia'
xR = data_d.columns[2:]
# yR = data_d.iloc[138,2: ]
yR = np.log10((data_d.iloc[138, 2: ]).astype(float)) / 10
plt.figure(figsize=(18, 8))
plt.plot(xR, yR, label=country_name)

plt.legend(country_name)
plt.xlabel('Day', size = 20)
plt.ylabel('log_death', size = 20)
plt.title('Russia Rate death country', size = 20)


# Графики выше показывают как выходят на "плато" страны. Характерный график у Китая - он уже на стабильном плато. \
# Все, кроме США - только подходят к плато. \
# Ниже посмотрим на страны, у которых пока уровень потерь невысок, однако, может быть из-за того, что страна только "входит" в процесс

# У нас есть матрицы:
#     - самые высокие потери =Топ-10
#     - самые быстрорастущие
#     - замедляющиеся или на плато

# ### Посмотрим на производную роста deaths

# In[30]:


data_d.info


# In[35]:


data_rate_ = data_d.iloc[:,:]

data_rate_ = data_rate_.to_numpy()
data_rate_summary = []
tmp = []
# print(data_d.shape, data_rate_[0,0], len(data_rate_), len(data_rate_[0,]))
for i in range(len(data_rate_)):
    for j in range(3, len(data_rate_[0,])-1):
        if data_rate_[i, j] == 0 or data_rate_[i, j+1] == 0:
            rate_=0
        else:
            rate_ = (data_rate_[i, j+1] - data_rate_[i, j]) / data_rate_[i, j+1]

        data_rate_summary.append(rate_)
data_rate_summary = np.array(data_rate_summary)
print(data_rate_summary.size)
tmp = data_rate_summary.resize(185,-1)


# сформируем фрейм с данным:
data_rate_summary = pd.DataFrame(tmp)

# добавим столбец с количеством умерших
data_rate_summary['num_cumulative_count'] = data_d.iloc[:,-1]
data_rate_summary['country'] = data_rate_[:,0]
# переставим колонку Страна на первое место
data_rate_summary = data_rate_summary.reindex(columns=(['country'] + list([a for a in data_rate_summary.columns if a != 'country']) ))

data_rate_summary.sort_values('num_cumulative_count', ascending = False).head(15)


# In[ ]:


data_d[data_d['country']=='Mexico']


# In[ ]:


# отсортируем страны Топ-20 по суммарному количеству умерших на текущую дату
data_rate_summary = data_rate_summary.sort_values(by = data_rate_summary.columns[-1], ascending=False).head(20)
# а теперь удалим столбец "потери"
data_rate_summary = data_rate_summary.drop(['num_cumulative_count'], axis=1)
# и теперь отсортируем Топ-10 по коэффициенту роста погибших
data_rate_summary = data_rate_summary.sort_values(by =  data_rate_summary.columns[-1], ascending=False).head(10)
data_rate_summary.head(10)


# ### График "Наибольший рост погибших за период наблюдений"

# In[ ]:


country_name = data_rate_summary.iloc[:,0]
xx = data_rate_summary.columns[1:]
yy = data_rate_summary
plt.figure(figsize=(18, 10))
for i in range(len(yy)):
#     plt.kdeplot(x, data_rate_summary.iloc[i, 1:], label=country_name)
    sns.kdeplot(xx, data_rate_summary.iloc[i, 1:], label=country_name)
plt.legend(country_name)
plt.xlabel('???', size = 20)
plt.ylabel('Density?', size = 20)
plt.title('Density death country', size = 20)


# In[ ]:


for i in range(len(yy)):
#     plt.kdeplot(x, data_rate_summary.iloc[i, 1:], label=country_name)
    sns.kdeplot(data_rate_summary.iloc[i, 1:], label=country_name)
plt.legend(country_name)
plt.xlabel('Score', size = 20)
plt.ylabel('Rate_death', size = 20)
plt.title('Density death country', size = 20)


# In[ ]:


for i in range(len(yy)):
#     plt.kdeplot(x, data_rate_summary.iloc[i, 1:], label=country_name)
    plt.plot(data_rate_summary.iloc[i, 1:], label=country_name)
plt.legend(country_name)
plt.xlabel('Days', size = 20)
plt.ylabel('Rate_death', size = 20)
plt.title('Divirgenz death country', size = 20)


# In[ ]:


# последняя декада наблюдений
country_name = data_rate_summary.iloc[:,0]
xx = data_rate_summary.columns[-10:]
yy = data_rate_summary.iloc[:,-10:]

for i in range(len(yy)):
#     plt.kdeplot(x, data_rate_summary.iloc[i, 1:], label=country_name)
    plt.plot(data_rate_summary.iloc[i, -10:], label=country_name)
plt.legend(country_name, loc='upper left')
plt.xlabel('Day', size = 15)
plt.ylabel('Rate_death', size = 15)
plt.title('Last 10 days death country', size = 20)


# In[ ]:


# data_d.iloc[:,0:-1]
data_columns = data_d.iloc[:,0]
result = pd.DataFrame(data=data_d.iloc[:,[0,1,-1]])
result = result.rename(columns={result.columns[-1]: 'max_d'})
# data_dd['country'] = data_d.iloc[1:,0]
# data_dd['max_num_deatch'] = data_d.iloc[1:,-1]
# data_dd
result.sort_values(by = 'rate_d_c', ascending = False).head(15)


# # Olymp

# #### спарсим данные с МОК по всем странам и получим таблицу завоеванных медалей по каждой стране с 1896 по 2018 гг

# In[ ]:


# Парсим сайт со страницей по медалям и странам:
# АДрес сайта:
url = 'http://olympanalyt.com/OlympAnalytics.php?param_pagetype=MedalsByCountries'
# создаем фрейм и делаем обертки для нужных колонок
olympic = pd.read_html(url, header=0)[2].iloc[2:, [0,15]].rename(columns={ 'All-time countries statistics':'country', 'All-time countries statistics.15':'medals'})
# print(olympic.sort_values('medals'))
olympic = olympic.reset_index(drop=True)
olympic['name_country'] = olympic['country'].str.split("\ \(").str.get(0)
olympic = olympic.drop('country', axis=1) 
olympic = olympic.rename({'name_country':'country'}, axis=1)
olympic['medals_int'] = olympic['medals'].astype(int)
olympic = olympic.drop(['medals'], axis=1)
olympic = olympic.rename({'medals_int': 'medals'}, axis=1)
olympic.head(5)


# ### Почистим названия стран , 
# #### которые менялись и суммируем общее значение медалей по каждой строке (бывш. названия стран) не менее 1% от общего значения

# In[ ]:


(olympic[olympic.country == 'Republic of China'])


# In[ ]:


rf = (olympic.iloc[165:170, 1]).sum()
olympic.loc[olympic.country == 'Russian Federation'] = ['Russian Federation', rf]
# удалим Гон-Конг и Республика Китай - их значения =>0
olympic = olympic.drop([166, 167, 168, 169, 42, 89], axis=0).reset_index(drop=True)
olympic.country[olympic.country == 'Russian Federation'] = 'Russia'

ch = (olympic.iloc[53:56, 1]).sum()
olympic.loc[olympic.country == 'Czech Republic'] = ['Czech Republic', ch]
olympic = olympic.drop([54, 55], axis=0).reset_index(drop=True)

ge = (olympic.iloc[71:76, 1]).sum()
olympic.loc[olympic.country == 'Germany'] = ['Germany', ge]
olympic = olympic.drop([72, 73, 74, 75], axis=0).reset_index(drop=True)

olympic = olympic.drop([138], axis=0).reset_index(drop=True)


# In[ ]:


# проверяем на ошибки и повторы
(olympic[olympic.country == 'Republic of China'])


# Результат вверху =None - значит все норм.

# ### Посчитаем
# #### вес страны в общем количестве медалей

# In[ ]:


olympic.tail()


# In[ ]:


total = olympic.medals.max()
olympic['rate_med'] = round((olympic['medals'] / total), 3)


# In[ ]:


olympic = olympic.drop('medals', axis=1)
olympic = olympic.drop([215,216,217,218], axis=0)


# #### Финальный список по странам -  в долях от общего кол-ва медалей за период 1896-2018гг

# In[ ]:


# проверка значений рейтинга
olympic.sort_values('rate_med', ascending=False).head(5)


# In[ ]:


# # проверка на максимум
# olympic.rate_m.max()


# ### Connect Olymp

# In[ ]:


left = result
right = olympic
result = pd.merge(right, left, on='country')
# переставим на первое место название страны
result = result.reindex(columns=(['country'] + list([a for a in result.columns if a != 'country']) ))
result.sort_values('rate_med', ascending=False)
# result


# In[ ]:


result.reset_index(drop=True)


# ## Economics
# #### экономические данные: доля страны в мировом ВП

# In[ ]:


# Парсим сайт со страницей по медалям и странам:
# АДрес сайта:
url = 'http://wdi.worldbank.org/table/WV.1'
# создаем фрейм и делаем обертки для нужных фичей
economic = pd.read_html(url)[2].iloc[0:,0:5]
economic = economic.rename({0:'country', 1:'popul', 2:'area', 3:'pop_density', 4:'gni'}, axis=1)
# удаляем ненужные строки
economic = economic.drop([216,217,218,219,220,221,222,223,224,225,226], axis=0)
economic = economic.reset_index(drop=True)
economic.country[economic.country == 'Russian Federation'] = 'Russia'
economic.country[economic.country == 'Kyrgyz Republic'] = 'Kyrgyzstan'
economic.country[economic.country == 'Korea, Dem. Peopleâs Rep.'] = 'Korea Republic'
economic.country[economic.country == 'Iran, Islamic Rep.'] = 'Iran'
economic.country[economic.country == 'Congo, Dem. Rep.'] = 'Congo DR'
economic.country[economic.country == 'Congo, Rep.'] = 'Congo'
economic.country[economic.country == 'Hong Kong SAR, China'] = 'Hong Kong'
# заемняем пустые значения на 0:
economic.popul[economic.popul=='..'] = 0
economic.area[economic.area=='..'] = 0
economic.pop_density[economic.pop_density=='..'] = 0
economic.gni[economic.gni=='..'] = 0

# преобразуем данные в формат float:
economic['popul_int'] = economic['popul'].astype(float)
economic['area_int'] = economic['area'].astype(float)
economic['pop_density_int'] = economic['pop_density'].astype(float)
economic['gni_int'] = economic['gni'].astype(float)
economic = economic.drop(['popul', 'area', 'pop_density', 'gni'], axis=1)
# найдем максимумы для подсчета весов - это итоговые строки по указанным параметрам, потом м их удаляем:
pop_max = economic.popul_int.max() # в таблице исходной есть строка с общим числом жителей планеты - она выступает как максимум
area_max = economic.area_int.max()
den_max = economic.pop_density_int.max()
gni_max = economic.gni_int.max()
# посчитаем веса по каждой стране и по каждому критерию
economic['rate_pop'] = round((economic['popul_int'] / pop_max), 3)
economic['rate_area'] = round((economic['area_int'] / area_max), 3)
economic['rate_den'] = round((economic['pop_density_int'] / den_max), 3)
economic['rate_gni'] = round((economic['gni_int'] / gni_max), 3)
# удаляем ненужные больше столбцы и строки
economic = economic.drop([215], axis=0)
economic = economic.drop(['popul_int','area_int','pop_density_int','gni_int'], axis=1)
economic = economic.drop([41], axis=0)

economic.head()


# In[ ]:


economic.shape


# #### Connect Economic

# In[ ]:


left = result
right = economic
result = pd.merge(right, left, on='country')
result.sort_values(by= 'rate_pop', ascending=False).head()


# In[ ]:


# переставим на первое место название страны
# result = result.reindex(columns=(['country'] + list([a for a in result.columns if a != 'country']) ))
# result.sort_values('rate_gni', ascending=False).head(10)


# # TOURISM
# #### количество входяших и исходяших в млн. на 2017г.

# In[ ]:


# Парсим сайт со страницей по медалям и странам:
# АДрес сайта:
url = 'http://wdi.worldbank.org/table/6.14'
# создаем фрейм и делаем обертки для нужных колонок
travel = pd.read_html(url)[2].iloc[0:,[0,2,4]]
travel = travel.rename({0 : 'country', 2 : 't_in', 4 : 't_out'}, axis=1)
# после визуального анализа (он здесь уже опущен за ненадобностью), удаляем ненужные нам колонки:
travel = travel.drop([215,216,217,218,219,220,221,222,223,224,225])

travel.country[travel.country == 'Russian Federation'] = 'Russia'
travel.country[travel.country == 'Kyrgyz Republic'] = 'Kyrgyzstan'
travel.country[travel.country == 'Korea, Dem. Peopleâs Rep.'] = 'Korea Republic'
travel.country[travel.country == 'Iran, Islamic Rep.'] = 'Iran'
travel.country[travel.country == 'Congo, Dem. Rep.'] = 'Congo DR'
travel.country[travel.country == 'Congo, Rep.'] = 'Congo'
travel.country[travel.country == 'Hong Kong SAR, China'] = 'Hong Kong'

travel.t_in[travel.t_in=='..'] = 0
travel.t_out[travel.t_out=='..'] = 0

travel['t_in_int'] = travel['t_in'].astype(int)
travel['t_out_int'] = travel['t_out'].astype(int)
travel = travel.drop(['t_in', 't_out'], axis=1)

# найдем максимумы для подсчета весов:
# удаляем строку с суммарным оборотом:
travel = travel.drop([214], axis=0)
com_in_max = travel.t_in_int.sum()
com_out_max = travel.t_out_int.sum()

# посчитаем веса по каждой стране и по каждому критерию
travel['rate_t_in'] = round((travel['t_in_int'] / com_in_max), 3)
travel['rate_t_out'] = round((travel['t_out_int'] / com_out_max), 3)

# удаляем ненужные больше столбцы и строки
travel = travel.drop(['t_in_int','t_out_int'], axis=1)

travel.sort_values('rate_t_out', ascending=False).head(10)


# In[ ]:


travel.sort_values(by='rate_t_in', ascending=False).head(10)


# #### Connect Tourism

# In[ ]:


left = result
right = travel
result = pd.merge(right, left, on='country')
result.head()


# In[ ]:


# result = result.drop(['X', 'Y'], axis=1)


# In[ ]:


result.shape


# ## FUEL RATE
# #### World Development Indicators: Contribution of natural resources to gross domestic product

# ### Fuel export

# In[ ]:


# Парсим сайт: доля страны в мировом торговом балансе экспорта за 2018 year по экспорту топливных ресурсов:
# АДрес сайта:
url = 'http://wdi.worldbank.org/table/4.4'
# создаем фрейм и делаем обертки для нужных колонок
oil_world_export = pd.read_html(url)[2].iloc[0:-12,[0,2,8]]
oil_world_export = oil_world_export.rename({0 : 'country', 2 : 'w_export', 8: 'part_oil_export_percent'}, axis=1)
oil_world_export.w_export[oil_world_export.w_export == '..'] = 0
oil_world_export.part_oil_export_percent[oil_world_export.part_oil_export_percent == '..'] = 0
oil_world_export['w_export'] = oil_world_export['w_export'].astype(float)
oil_world_export['part_oil_export_percent'] = oil_world_export['part_oil_export_percent'].astype(float)
oil_world_export['rate_oil'] = oil_world_export.w_export * oil_world_export.part_oil_export_percent / 100
# oil['gdp'] = oil['gdp'].astype(float)
total_oil = oil_world_export.rate_oil.sum()
# рейтинг страны в мировом экспорте топлива (нефть, газ, уголь и пр) = we - world_export
oil_world_export['rate_we_fuel'] = round(oil_world_export.rate_oil / total_oil, 2)

oil_world_export = oil_world_export.drop(['w_export', 'part_oil_export_percent', 'rate_oil'], axis=1).reset_index(drop=True)
oil_world_export.sort_values(by=oil_world_export.columns[1],ascending=False).head(10)


# ### Fuel import

# In[ ]:


# Парсим сайт: доля страны в мировом торговом балансе по импорту за 2018 year по импорту топливных ресурсов:
# АДрес сайта:
url = 'http://wdi.worldbank.org/table/4.5'
# создаем фрейм и делаем обертки для нужных колонок
oil_world_import = pd.read_html(url)[2].iloc[0:-12,[0,2,8]].rename({0 : 'country', 2 : 'w_import', 8: 'part_fuel_import_percent'}, axis=1)
# oil_world_import = oil_world_import.rename({0 : 'country', 2 : 'w_import', 8: 'part_fuel_import_percent'}, axis=1)
oil_world_import.w_import[oil_world_import.w_import == '..'] = 0
oil_world_import.part_fuel_import_percent[oil_world_import.part_fuel_import_percent == '..'] = 0
oil_world_import['w_import'] = oil_world_import['w_import'].astype(float)
oil_world_import['part_fuel_import_percent'] = oil_world_import['part_fuel_import_percent'].astype(float)
oil_world_import['rate_fuel'] = oil_world_import.w_import * oil_world_import.part_fuel_import_percent / 100
# oil['gdp'] = oil['gdp'].astype(float)
total_fuel_import = oil_world_import.rate_fuel.sum()
# рейтинг страны в мировом импорте топлива (нефть, газ, уголь и пр) = wi - world_import
oil_world_import['rate_wi_fuel'] = round(oil_world_import.rate_fuel / total_fuel_import, 2)

oil_world_import = oil_world_import.drop(['w_import', 'part_fuel_import_percent', 'rate_fuel'], axis=1).reset_index(drop=True)
oil_world_import.sort_values(by=oil_world_import.columns[1],ascending=False).head(10)


# #### Connect export & import data

# In[ ]:


data_oil = pd.DataFrame(oil_world_export)
# перемиенуем столбцы rate_wfi = 'rate world fuel import' / rate_wfe = 'rate world fuel export'
data_oil = data_oil.rename({'rate_we_fuel' : 'rate_wfi'}, axis=1)
data_oil['rate_wfe'] = oil_world_import.iloc[:,-1]
data_oil.sort_values(by='country', ascending=True).head()


# In[ ]:


data_oil.country[data_oil.country == 'Russian Federation'] = 'Russia'
data_oil.country[data_oil.country == 'Kyrgyz Republic'] = 'Kyrgyzstan'
data_oil.country[data_oil.country == 'Korea, Dem. Peopleâs Rep.'] = 'Korea Republic'
data_oil.country[data_oil.country == 'Iran, Islamic Rep.'] = 'Iran'
data_oil.country[data_oil.country == 'Congo, Dem. Rep.'] = 'Congo DR'
data_oil.country[data_oil.country == 'Congo, Rep.'] = 'Congo'
data_oil.country[data_oil.country == 'Hong Kong SAR, China'] = 'Hong Kong'

data_oil.rate_wfi[data_oil.rate_wfi == '..'] = 0
data_oil.rate_wfe[data_oil.rate_wfe == '..'] = 0

data_oil['rate_wfi'] = data_oil['rate_wfi'].astype(float)
data_oil['rate_wfe'] = data_oil['rate_wfe'].astype(float)

data_oil.head()


# In[ ]:


data_oil.shape


# ### Connect data

# In[ ]:


result.head()


# In[ ]:


left = result
right = data_oil
result = pd.merge(right, left, on='country')
result.head()


# # MORTALITY
# #### Уровень смертности в стране в 2018

# In[ ]:


# количество дней наблюдений
count_days = len(data_d.loc[0])-2


# In[ ]:


# Парсим сайт со страницей по уровню смертности:
# Адрес сайта:
url = 'http://wdi.worldbank.org/table/2.1'
# создаем фрейм и делаем обертки для нужных колонок
mort = pd.read_html(url)[2].iloc[0:,[0,9]]
mort = mort.rename({0 : 'country', 9 : 'rate_mort'}, axis=1)
mort = mort.drop([214,215,216,217,218,219,220,221,222,223,224,225])

mort.country[mort.country == 'Russian Federation'] = 'Russia'
mort.country[mort.country == 'Kyrgyz Republic'] = 'Kyrgyzstan'
mort.country[mort.country == 'Korea, Dem. Peopleâs Rep.'] = 'Korea Republic'
mort.country[mort.country == 'Iran, Islamic Rep.'] = 'Iran'
mort.country[mort.country == 'Congo, Dem. Rep.'] = 'Congo DR'
mort.country[mort.country == 'Congo, Rep.'] = 'Congo'
mort.country[mort.country == 'Hong Kong SAR, China'] = 'Hong Kong'

mort.rate_mort[mort.rate_mort=='..'] = 0
mort['rate_mort'] = mort['rate_mort'].astype(int)

# найдем макс для подсчета весов:
mort_r_max = mort.rate_mort.max()
# посчитаем значение коэффициента естесственной убыли населения в стране за текущий период наблюдений пандемии:
mort['rate_mort'] = round((mort['rate_mort'] / (12*30)) * (count_days), 3)
# mort = mort.drop('rate_mort', axis=1)

mort.sort_values('rate_mort', ascending=False)
# mort.shape


# #### Connect Mort

# In[ ]:


left = result.copy()
right = mort
result = pd.merge(right, left, on='country')


# In[ ]:


result['rate_mort'] = result['rate_mort'] / 4
result.head()


# In[ ]:


result.shape


# # AGE
# #### доля населения 15-65 лет

# In[ ]:


# Парсим сайт со страницей по медалям и странам:
# АДрес сайта:
url = 'http://wdi.worldbank.org/table/2.1'
# создаем фрейм и делаем обертки для нужных колонок
age = pd.read_html(url)[2].iloc[0:,[0,5]]
age = age.rename({0 : 'country', 5 : 'rate_age'}, axis=1)
age = age.drop([214,215,216,217,218,219,220,221,222,223,224,225])

age.country[age.country == 'Russian Federation'] = 'Russia'
age.country[age.country == 'Kyrgyz Republic'] = 'Kyrgyzstan'
age.country[age.country == 'Korea, Dem. Peopleâs Rep.'] = 'Korea Republic'
age.country[age.country == 'Iran, Islamic Rep.'] = 'Iran'
age.country[age.country == 'Congo, Dem. Rep.'] = 'Congo DR'
age.country[age.country == 'Congo, Rep.'] = 'Congo'
age.country[age.country == 'Hong Kong SAR, China'] = 'Hong Kong'

age.rate_age[age.rate_age=='..'] = 0
age['rate_age'] = age['rate_age'].astype(int)

# найдем макс для подсчета весов:
age_r_max = age.rate_age.max()
# посчитаем веса по каждой стране и по каждому критерию
age['rate_age'] = round((age['rate_age'] / 100), 3)
# age = age.drop('rate_age', axis=1)

# age.sort_values('rate_age', ascending=False)
age.shape


# #### Connect Age

# In[ ]:


left = result
right = age
result = pd.merge(right, left, on='country')
result.head()


# In[ ]:


result[result.country == 'China']


# In[ ]:


result.head()


# In[ ]:


result.sort_values(by= 'rate_pop', ascending=True).head(10)


# In[ ]:


result.info()


# # Преобразуем данные 
# ### проведем подготовку данных

# In[ ]:


result_finale_data = result.copy()
result_finale_data.sort_values(by= 'rate_pop', ascending=True).head()


# In[ ]:


# дропнем строки с 0 по фиче rate_pop = нам не интересны эти страны
result_finale_data = result_finale_data.drop(result_finale_data[result_finale_data['rate_pop'] == 0].index)


# In[ ]:


result_finale_data.sort_values(by='rate_pop', ascending=True).head()


# In[ ]:


# сумма = число погибших в мире
death_sum = (result_finale_data.iloc[:,-1]).sum()
death_sum


# In[ ]:


# доля погибших от covid к населению страны
result_finale_data['rate_d_pop'] = (result_finale_data.iloc[:,-1] / (pop_max * result_finale_data.rate_pop * 1000))

result_finale_data['rate_covid_mort'] = result_finale_data.iloc[:,-3] / result_finale_data.iloc[:,2]


# In[ ]:


# доля смертей к числу жителей в стране
result_finale_data.sort_values('rate_d_pop', ascending=False).head()


# In[ ]:


result_finale_data.shape


# In[ ]:


# доля страны от общего числа погибших в мире
# result_finale_data['rate_d'] = result_finale_data.iloc[:,-2] / data_c


# In[ ]:


result_data = result_finale_data.drop('max_d' , axis=1)


# In[ ]:


result_data.sort_values(by=result_data.columns[-1], ascending=False).head()


# In[ ]:


result_data.describe()


# In[ ]:


result_data = result_data.dropna()


# In[ ]:


result_data.sort_values(by = 'rate_d_pop', ascending=False)


# In[ ]:


result_data.reset_index(drop=True)


# #### Проведем нормировку данных

# In[ ]:


result_norm = result_data.iloc[:,1:]
result_norm['rate_mort'] = result_norm['rate_mort'] / 4
result_norm.head()


# In[ ]:


result_norm.describe()


# In[ ]:


# сохраним данные во фрейм
result_norm = (result_norm - result_norm.mean(axis=0)) / (result_norm.std(axis=0))
result_norm.shape


# In[ ]:


result_norm.head(10)


# In[ ]:


result_norm.tail(10)


# ## Result Graf

# In[ ]:





# In[ ]:





# In[ ]:


country_name = result_norm.iloc[:,0]
num_col = result_norm.columns[1:]
# print(num_col)
for col in num_col:
    x = pd.Series(result_norm[col], name=col)
#    sns.distplot(x)
    sns.kdeplot(x)


# In[ ]:


result_norm_T = result_norm.T
result_norm_T.shape


# In[ ]:


result_norm.hist(figsize=(12,12))


# In[ ]:


# найдем макс для подсчета весов:
num_d_max = result_finale_data.iloc[:,-1].max()
num_d_max


# In[ ]:


h = sns.pairplot(result_norm, hue=None, height=5, aspect=1)


# In[ ]:


plt.scatter(result_norm['rate_d_c'], result_norm['rate_covid_mort'])


# In[ ]:


plt.scatter(result_norm['rate_t_out'], result_norm['rate_gni'])


# In[ ]:


plt.scatter(result_norm['rate_gni'], result_norm['rate_t_out'])


# In[ ]:


result_norm.head()


# In[ ]:


result_agg_weight = result_norm.copy()
result_agg_country = result_finale_data.pop('country')


# In[ ]:





# In[ ]:


for column in result_agg_weight:
    x = pd.Series(result_agg_weight[column], name=column)
#    sns.distplot(x)
    sns.kdeplot(x)


# In[ ]:





# In[ ]:


f = sns.kdeplot(pd.Series(result_finale_data['rate_gni'], name='rate_gni'))


# In[ ]:


# result_finale_data.iloc[:,-1]


# In[ ]:


data = result_finale_data.sort_values(by = result.columns[-1], ascending=False).head(20)
# data = np.array(result.head(20))
type(data), data.shape


# In[ ]:


# plt.plot(range(len(data)), (data), label='data')
# plt.plot(data=data[0,1:], label='data')
# sns.kdeplot(data[1:,:])
# columns = data.columns[9:-1]
# # print(columns, '&',data.iloc[0,8:], len(data))
# for i in range(0, len(data)):
#     x = pd.Series(data.iloc[i,9:])
# #    sns.distplot(x)
#     sns.kdeplot(x)


# In[ ]:


result.sort_values(by = 'rate_covid_mort', ascending=False).head()


# In[ ]:


result_norm.head()


# In[ ]:


result_finale_data.head()


# In[ ]:


result_finale_data.sort_values(by = 'rate_covid_mort', ascending=False).head()


# In[ ]:


result_finale_data.sort_values(by = 'rate_d_pop', ascending=False).head()


# In[ ]:


result_data[result_data.country =='Russia']


# In[ ]:


result_finale_data.info()


# In[ ]:


correlations_data = result_norm.corr(method='pearson', min_periods=1)['rate_d_pop'].sort_values()
correlations_data


# ## Write to CSV dataset

# In[ ]:


result_data.to_csv('covid_cumulative_dataset_rates.csv')


# # Result dataset

# In[ ]:


dataset = pd.read_csv('covid_cumulative_dataset_rates.csv', index_col=0)
dataset.info()


# In[ ]:


dataset.sort_values(by=['rate_d_pop'], ascending=False).head(10)


# In[ ]:


dataset.sort_values(by=['rate_covid_mort'], ascending=False).head(10)


# ## Graph

# In[ ]:


# !pip install cufflinks


# In[ ]:


dataset.describe()


# In[ ]:


z = sns.pairplot(dataset, hue=None, height=5, aspect=1)


# In[ ]:


columns = dataset.columns
# df = pd.DataFrame(dataset).cumsum()
df = pd.DataFrame(dataset.iloc[1:,1:])

fig = df.iplot(asFigure=True, xTitle="The X Axis",
                    yTitle="The Y Axis", title="The Figure Title")
fig.show()


# In[ ]:


dataset.iloc[76:79,:4]


# In[ ]:


dataset.shape


# In[ ]:


plt.figure(figsize=(24, 24))
df=dataset.iloc[:,1:15]
fig = df.iplot(asFigure=True, subplots=True, shape=(15,1), shared_xaxes=True, fill=True)
fig.show()


# # ML

# In[ ]:


from sklearn import metrics
from sklearn.svm import SVC


# In[ ]:


dataset = pd.read_csv('covid_cumulative_dataset_rates.csv', index_col=0)

X = dataset.iloc[:,1:]
y = dataset.pop('country')


# In[ ]:


y


# In[ ]:


# fit a SVM model to the data
model = SVC()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# In[ ]: