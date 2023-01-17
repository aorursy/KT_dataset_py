import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import folium

from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# информация о локациях, координаты и популяция. С базовой таблицей связь по title
file = '/kaggle/input/tuva-populaton/populat_tuva.xlsx'
geodata = pd.read_excel(file, sheet_name=0, header=0, index_col='Unnamed: 0')

#Get points
points = (geodata.lat.fillna(0),geodata.lot.fillna(0), geodata['population_2010'].fillna(0))
lat = points[0]
long = points[1]
populat = points[2]

# Map
map_tuva = folium.Map(location=[51.719082, 94.433983],width=600, height=400,max_zoom=6)

for la,lo, populat in zip(lat,long, populat):    
    
    if populat <= 100:
        size = 0.5
        color='#c3c793'
    elif populat <= 1000:
        size = 1
        color='#e3c139'
    elif populat <= 10000:
        size = 1
        color='#e38b39'
    elif populat <= 50000:
        size = 3
        color='#ff0000'
    else:
        size = 0.5
        color='#D6ACD0'  
        
    folium.CircleMarker(
        location=[la,lo],
        radius=size,
        popup= 'Population {}'.format(populat),
        color=color,
        fill=True,
        fill_color='green'
    ).add_to(map_tuva)
    
map_tuva 
"""
Легенда: В Кызыл приезжает студент зараженный COVID-19. Приезжает на маршрутном такси из Новосибирска. 
Позже, а именно 30/03/2020 вирус обнаруживается у водителя, который проживает в Чадане. 
Студент проживает в семье с мамой и бабушкой, у которых позже также диагностировали заражение.  
Бабушка в возрасте 67 лет умирает 10/04/2020.  Необходимо составить математическую модель 
распространения COVID-19. Дополнительно, необходимо посчитать смертность, при наличии 80 аппаратов ИВЛ.

UDP данных очень мало, поэтому используйте компартментальные модели SIR 
https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology

Базовая статистика по Кызылу: https://www.gks.ru/bgd/regl/b12_14t/IssWWW.exe/Stg/sibir/03-00.htm

id - id пациента
sex - пол
birth - день рождения
city_id - место рождения или место фактического местонахождения
infection_reason - тип заражения
contact_number - потенциальное число контактов за весь период, включая до момента confirmed_date
confirmed_date - дата подтверждения
released_date - дата выписки в случае полного выздоровления
dead_day - дата смерти
predict_count - неопределенный коэффициент
"""
test_data = pd.DataFrame(columns=['id', 'sex', 'birth', 'city_id', 'infection_reason', 'contact_number', 
                                  'confirmed_date', 'released_date', 'dead_day', 'predict_count'])

test_data.loc[len(test_data)] = ['1', 'man', '10/02/2000', 'Кызыл', 'student', '40', '15/03/2020', '05/04/2020', '', '5']
test_data.loc[len(test_data)] = ['2', 'woman', '25/08/1975', 'Кызыл', 'contact', '20', '25/03/2020', '', '', '3']
test_data.loc[len(test_data)] = ['3', 'woman', '13/05/1952', 'Кызыл', 'contact', '10', '20/03/2020', '', '10/04/2020', '2']
test_data.loc[len(test_data)] = ['4', 'man', '10/02/1985', 'Чадан', 'contact', '80', '30/03/2020', '', '', '5']

test_data.birth = pd.to_datetime(test_data.birth)
test_data.confirmed_date = pd.to_datetime(test_data.confirmed_date)
test_data.released_date = pd.to_datetime(test_data.released_date)
test_data.dead_day = pd.to_datetime(test_data.dead_day)

test_data
# формируем данные по зараженным населеным пунктам и наличию ИВЛ
geo_infection = geodata.loc[geodata['title'].isin(test_data['city_id'])].reset_index().drop(['index', 'full_adress', 'id', 'region'], axis='columns', inplace=False)
geo_infection['count_IVL'] = 0
geo_infection.at[0,'count_IVL'] = 80
geo_infection.at[1,'count_IVL'] = 5 # В Чадане для теста добавлено значение  - 5 ИВЛ


#Get points
points = (geo_infection.lat.fillna(0),geo_infection.lot.fillna(0))
lat = points[0]
long = points[1]

# Map
title_html = '''<h3 align="center" style="font-size:20px"><b></b></h3>'''
map_tuva = folium.Map(location=[51.719082, 94.433983],width=600, height=400,max_zoom=6)
pop_title = geo_infection['population_2010']


for la,lo in zip(lat,long):
    folium.CircleMarker(
        location=[la,lo],
        radius=5,
        #popup='Population',
        color='red',
        fill=True,
        fill_color='red'
    ).add_to(map_tuva)
    

point = [[la,lo] for la,lo in zip(lat,long)]  
folium.PolyLine(point, color='red').add_to(map_tuva)

map_tuva
geo_infection
# добавляем данные по возрастному составу заражённых населеных пунктов в процентах https://vpngis.gks.ru:8443/ShowReport.aspx?DataSourceName=pub-02-02 (copy-past - привет росстату)
str_data = """
0 – 4								36519
5 – 9								27512
10 – 14								24373
15 - 19								25048
20 - 24								29137
25 - 29								27346
30 - 34								23894
35 - 39								22583
40 - 44								20912
45 - 49								19616
50 - 54								16066
55 - 59								11119
60 - 64								8895
65 - 69								5665
70 - 74								5036
75 - 79								2246
80 - 84								1399
85								563
"""
age_region = pd.DataFrame([x.split('\t') for x in str_data.split('\n')])
age_region = age_region.drop(0)
age_region = age_region.drop(19)
age_region = age_region[[0,8]]
age_region.columns = ['age_group', 'population']
age_region['population'] = age_region['population'].astype('int')

age_region
plt.figure(figsize=(15,10))
plt.title("Возрастные категории населения Республики Тыва (2010 г.)",fontsize=20)
plt.xlabel("Количество людей",fontsize=18)
plt.ylabel("Возрастная группа (лет)",fontsize=18)
plt.bar(age_region['age_group'], age_region['population'])
plt.show()
# формируем коэфицент смертности по данным "Китайский вариант" https://www.dw.com/ru/%D0%B8%D1%81%D1%81%D0%BB%D0%B5%D0%B4%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5-%D0%BA%D1%82%D0%BE-%D1%87%D0%B0%D1%89%D0%B5-%D0%B2%D1%81%D0%B5%D0%B3%D0%BE-%D1%83%D0%BC%D0%B8%D1%80%D0%B0%D0%B5%D1%82-%D0%BE%D1%82-%D0%BA%D0%BE%D1%80%D0%BE%D0%BD%D0%B0%D0%B2%D0%B8%D1%80%D1%83%D1%81%D0%B0/a-52434059
dead = """
0 – 9	0
10 – 39	0.2
40 – 49	0.4
50 – 59	1.3
60 – 69	3.6
70 – 79	8.0
80	14.8
"""

dead_age = pd.DataFrame([x.split('\t') for x in dead.split('\n')])
dead_age = dead_age.drop(0)
dead_age = dead_age.drop(8)
dead_age.columns = ['age_group', 'dead_persent']
dead_age['dead_persent'] = dead_age['dead_persent'].astype('float32')

nol = age_region.loc[1:2]['population'].sum()
desat = age_region.loc[3:8]['population'].sum()
sorok = age_region.loc[9:10]['population'].sum()
padesyat = age_region.loc[11:12]['population'].sum()
chestdesat = age_region.loc[13:14]['population'].sum()
semdesat = age_region.loc[15:16]['population'].sum()
vosemdesay = age_region.loc[17:]['population'].sum()

dead_age['population'] = ''
dead_age.at[1,'population'] = nol
dead_age.at[2,'population'] = desat
dead_age.at[3,'population'] = sorok
dead_age.at[4,'population'] = padesyat
dead_age.at[5,'population'] = chestdesat
dead_age.at[6,'population'] = semdesat
dead_age.at[7,'population'] = vosemdesay

# Абсолютное значение на всю возрастную популяцию! В реальности эти данные применимы тоьлко на количество зараженных.
dead_age['dead_people'] = ''
dead_age['dead_people'] = dead_age['population'] * dead_age['dead_persent'] // 100

dead_age
# Условные данные без учёта реального количества зараженных!
plt.figure(figsize=(13,7))
plt.title('Суммарный коэффициент "смертность/возраст" от COVID-19 среди зараженных \n на территории Республики Тыва',fontsize=20)
plt.ylabel('Популяция')
plt.bar(dead_age['age_group'], dead_age['population'],label='Плотность по возрасту')
plt.plot(dead_age['age_group'], dead_age['dead_persent']*10000,label='Коэффициент смертности', color="crimson") # перемножил для визуальной оценки! Условные данные без учёта реального количества зараженных!
plt.bar(dead_age['age_group'], dead_age['dead_people']*10,label='Плотность умерших (условное)', color="crimson") # перемножил для визуальной оценки! Условные данные без учёта реального количества зараженных!
plt.legend()
# Абсолютное значение на всю возрастную популяцию! В реальности эти данные применимы только на количество зараженных.
dead_people = dead_age['dead_people']
age_group = dead_age['age_group']

plt.figure(figsize=(13,7))
plt.title("Показатели рисков смертности (см. пояснение) от Covid-19 \n по Республике Тыва с учётом возрастных групп (уханьский сценарий)",fontsize=20)
plt.xlabel("Количество (Внимание! Данные на всю популяцию, только для тестирования)",fontsize=16)
plt.ylabel("Возрастная группа (лет)",fontsize=18)
plt.barh(age_group,dead_people, color="crimson")
plt.show()
# добавим возраст и период
today = datetime.datetime.today()
test_data['age'] = (pd.to_datetime('2020', format='%Y') - pd.to_datetime(test_data['birth'], format='%Y')).astype('timedelta64[Y]')
test_data['post_infection_period'] = today - test_data['confirmed_date']

test_data
print (test_data) # данные по заболевшим
print (dead_age) # возрастные группы и коэффициент смертности (уханьский сценарий)
print (geo_infection) # заражённые локации
"""
S = S(t) 	группа риска,
I = I(t) 	заболевших
R = R(t) 	выздоровевших
"""
# определяем группу риска susceptible dead (S)
susceptible = dead_age['dead_persent'].sum() #процент смертности людей из df.dead_age
geo_infection['susceptible'] = ''
geo_infection['susceptible'] = geo_infection['population_2010'].apply(lambda x : susceptible * x // 100)
geo_infection['infected_people'] = ''

# Вносим в таблицу количество заболевших I
def infection_counts(x):
    data = test_data.loc[test_data['city_id'] == x]
    return data['city_id'].value_counts()
geo_infection['infected_people'] = geo_infection['title'].apply(lambda x : infection_counts(x))
print ('Процент смертности среди заболевших', susceptible)
geo_infection

test_data
# dead_people - Абсолютное значение на всю возрастную популяцию! В реальности эти данные применимы только на количество зараженных.
dead_age
# https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Total population, N.
N = 109918
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 4, 3
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma = 0.2, 1./10 
# A grid of time points (in days)
t = np.linspace(0, 160, 160)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w',figsize=(15,7))
ax = fig.add_subplot(111, axisbelow=True)

ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Группа риска')
ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Зараженные')
ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Выздоровевших')
ax.set_xlabel('Время /дней')
ax.set_ylabel('Число (1000s)')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()
"""
N - популяция населенного пункта
I0 - начальное количество заражённых
R0 - начальное количество выздоровевших
S0 - группы риска
beta - скорость контакта
gamma - скорость выздоравливания
t - тета временных рядов
y - 
"""
sir_data = pd.DataFrame(columns=['N', 'I0', 'R0', 'S0', 'beta', 'gamma', 't', 'y'])
sir_data
