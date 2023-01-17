# загружаем требуемые библиотеки и пребразуем ранее сохраненные файлы csv в датафреймы

import pandas as pd

import numpy as np

registered_users_count = pd.read_csv('../input/retention-rate/registered_users_count.csv')

active_users_count_with_cohorts = pd.read_csv('../input/retention-rate/active_users_count_with_cohorts.csv')
registered_users_count.info()
active_users_count_with_cohorts.info()
registered_users_count['registration_date'] = pd.to_datetime(registered_users_count['registration_date'])

# преобразование строковых данных в формат datetime в таблице registered_users_count

for col in ['activity_date','registration_date']:

    active_users_count_with_cohorts[col] = pd.to_datetime(active_users_count_with_cohorts[col])

# преобразование строковых данных в формат datetime в таблице active_users_count_with_cohorts
retention_table = active_users_count_with_cohorts.merge(registered_users_count,on=['registration_date'],how='left')

retention_table.head()
retention_table['retention_rate'] = retention_table['active_users_count'] / retention_table['registered_users_count']
retention_table['lifetime'] = retention_table['activity_date'] - retention_table['registration_date']

retention_table['lifetime'] = retention_table['lifetime']/np.timedelta64(1,'D')

retention_table['lifetime'] = retention_table['lifetime'].astype(int) # Приведем тип к целому числу

retention_table.head()
retention_pivot = retention_table.pivot_table(index='registration_date',columns='lifetime',values='retention_rate',aggfunc='sum')

retention_pivot.head()
import seaborn as sns

from matplotlib import pyplot as plt



sns.set(style='white')

plt.figure(figsize=(20, 9))

plt.title('Cohorts: User Retention')

sns.heatmap(retention_pivot,mask=retention_pivot.isnull(),annot=True,fmt='.0%',

            linewidths=0.5,linecolor='black',center=-0.1)
# для построения графика Retention Rate по всем когортам июня 2019г., для каждого значения Lifetime, вычисляем список средних значений значений

June_mean=[retention_pivot[i].mean() for i in retention_pivot]
import matplotlib.pyplot as plt

import matplotlib.ticker as ticker



fig, ax = plt.subplots()

ax.plot(June_mean, marker = 'o', color = 'blue') # формируем график и его формат, определяем маркеры

retention_rate = np.arange(-1,len(June_mean)) # числовые значения для оси X



fig.set_figwidth(12) # ширина фигуры

fig.set_figheight(6) # высота фигуры



ax.set_title('Изменение среднего Retention Rate в июне 2019 г.\nв зависимости от времени жизни пользователя', fontsize = 16)

#  Устанавливаем позиции тиков для всех осей

ax.tick_params(axis = 'both', which = 'major', bottom = True, left = True)

#  Устанавливаем интервал основных делений по оси X

ax.set_xticks(retention_rate)

ax.set_xlim([-0.5, 30.5])



#  Устанавливаем подписи тиков оси X

lable=['Day '+ str(x) for x in retention_rate] # строки для подписей оси X

ax.set_xticklabels(lable, rotation = 45)



# изменяем формат значений по Y

ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0)) # формат значений по Y в %

ax.set_ylabel('Retention Rate')

ax.grid(axis='y') # линии сетки только по Y  



plt.show()
revenue = pd.read_csv('../input/retention-rate/revenue.csv')

revenue.info()
# преобразуем значения столбцов date и registration_date в формат дат и просмотрим результат

for col in ['date','registration_date']:

    revenue[col] = pd.to_datetime(revenue[col])

revenue.head()
revenue = revenue.rename(columns={'date':'activity_date'})

# переименование колонок

retention_table_with_revenue = retention_table.merge(revenue,on=['registration_date','activity_date'],how='left')

retention_table_with_revenue.info()

# объединение таблиц и просмотр общих сведений об объединенной таблице
for col in ['revenue','users_count_with_revenue']:

    retention_table_with_revenue[col] = retention_table_with_revenue[col].fillna(0)

    

retention_table_with_revenue['users_count_with_revenue'] = retention_table_with_revenue['users_count_with_revenue'].astype(int)
retention_table_with_revenue['arpu'] = retention_table_with_revenue['revenue'] / retention_table_with_revenue['active_users_count']

retention_table_with_revenue.head()
pivot_arpu = retention_table_with_revenue.pivot_table(index='registration_date',columns='lifetime',values='arpu',aggfunc='mean')

pivot_arpu.head()
sns.set(style='white')

plt.figure(figsize=(20, 9))

plt.title('Зависимость среднего значения ARPU по когортам от lifetime (июнь 2019 г.)', fontsize=16)

# формат значений оси Y

Y_index=[pivot_arpu.index[i].strftime('%d-%m-%Y') for i in range(len(pivot_arpu.index))] 

sns.heatmap(pivot_arpu,

            annot=True, 

            fmt='.2g', 

            linewidths=0.1, 

            center=-0.25, # смещение цветовой палитры в сторону светлого цвета (темные только 0, все значащие элем. светлее)

            linecolor='w', # белые линии сетки

            yticklabels = Y_index)
fig, ax = plt.subplots()

ax.plot(pivot_arpu[0], marker = 'o', color = 'red')

# формируем график и его формат, определяем маркеры

lifetime_rate = pivot_arpu.index # числовые значения для оси X

fig.set_figwidth(18) # ширина фигуры

fig.set_figheight(6) # высота фигуры



#  тиков для оси X

ax.set_xticks(lifetime_rate)

ax.tick_params(axis = 'x', which = 'major', bottom = True)



#  подписи тиков оси X

lable=[lifetime_rate[i].strftime('%d-%m-%Y') for i in range(len(lifetime_rate))] # формат подписей оси X

ax.set_xticklabels(lable, rotation = 45)

ax.set_title('График значений ARPU\nпо когортам июня 2019 г. в нулевой день жизни', fontsize = 16)

ax.set_ylabel('ARPU, $', fontsize = 14)

ax.set_xlabel('Когорты', fontsize = 14)

ax.grid(axis='y') # линии сетки только по Y



plt.show()
# список средних по когортам значений ARPU для каждого значения lifetime

Cohorts_mean = [pivot_arpu[i].mean() for i in pivot_arpu.columns]



fig, ax = plt.subplots()

ax.plot(Cohorts_mean, marker = 'X', color = 'g', markersize = 10) 

# формируем график и его формат, определяем маркеры

lifetime_long = np.arange(-1,len(Cohorts_mean)) 

# числовые значения для оси X



fig.set_figwidth(16) # ширина фигуры

fig.set_figheight(6) # высота фигуры



ax.set_xticks(lifetime_long)

ax.set_xlim([-0.5, 30.5])

ax.tick_params(axis = 'both', which = 'major', bottom = True, left = True)

ax.set_xlabel('Lifetime')

ax.set_ylabel('ARPU, $')

ax.set_title('Зависимость среднего значения ARPU по когортам\nот времени жизни пользовтеля', 

             fontsize = 16)

#  Устанавливаем подписи тиков оси X

lable=['Day '+ str(x) for x in retention_rate] # строки для подписей оси X

ax.set_xticklabels(lable, rotation = 45)

ax.grid() 



plt.show()