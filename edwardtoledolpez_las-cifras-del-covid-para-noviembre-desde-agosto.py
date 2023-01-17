#For the system

import os



#Manage of time

from datetime import datetime, timedelta, date

import calendar



#Manage of files

import pandas as pd

import csv

import numpy as np



#Graph tools

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import dates as mdates

import matplotlib.patches as mpatches



#interactive visualization

import plotly.express as px

import plotly.graph_objs as go



#Gauss-Jordan

import math
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

df.dtypes
df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])

df = df[['ObservationDate',

         'Country/Region',

         'Confirmed',

         'Deaths',

         'Recovered']]
#Hola extraño, si te preguntas que es esto, es para hacer que los colores hagan match con el estilo. Sí, TOC

colors = [u'#348ABD', u'#A60628', u'#7A68A6', u'#467821', u'#D55E00', u'#CC79A7', u'#56B4E9', u'#009E73', u'#F0E442', u'#0072B2']

names = ['blue', 'red', 'purple', 'green', 'orange', 'pink', 'light_blue', 'intense_green', 'pastel_yellow', 'dark_blue']

bmh_colors = dict(zip(names, colors))





x = df['ObservationDate']

y_df = ['Confirmed', 'Deaths', 'Recovered']

titles = ['COVID-19 Confirmed Cases by date', 'COVID-19 Deaths Cases by date', 'COVID-19 Recovered Cases by date']

colors = [bmh_colors['blue'], 'k', bmh_colors['green']]



fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(25,5), sharey=False, sharex=True)

plt.style.use('ggplot')

n = 0



for ax in fig.get_axes():

    ax.scatter(x, df[ y_df[n] ], s=1, c=colors[n])

    ax.set_title(titles[n])



    date_format = mdates.DateFormatter('%m')

    ax.set_xlabel('Dates')

    ax.xaxis.set_major_formatter(date_format)



    n = n + 1



plt.show()
start_day = datetime(year=2020, month=1, day=22)

end_day = datetime(year=2020, month=8, day=12)

days = end_day - start_day

n_days = days.days



list_days = [start_day + timedelta(days=x) for x in range(n_days)]
dates = df['ObservationDate'].to_list()

confirmed = df['Confirmed'].to_list()

deaths = df['Deaths'].to_list()

recovered = df['Recovered'].to_list()

countries = df['Country/Region'].to_list()
sum_confirmed = [0] * (n_days)

sum_deaths = [0] * (n_days)

sum_recovered = [0] * (n_days)



day_subregion = [None] * (n_days)

day_continent = [None] * (n_days)



plus = 0

for d in range(len(list_days)):



    while dates[plus] == list_days[d] and (d + plus) < len(dates):

        '''---BY DAY---'''

        #General

        sum_confirmed[d] = int(sum_confirmed[d]) + int(confirmed[d + plus])

        sum_deaths[d] = int(sum_deaths[d]) + int(deaths[d + plus])

        sum_recovered[d] = int(sum_recovered[d]) + int(recovered[d + plus])

        

        plus += 1
ys = [sum_confirmed, sum_deaths, sum_recovered]

titles = ['COVID-19 Confirmed Cases by date', 'COVID-19 Deaths Cases by date', 'COVID-19 Recovered Cases by date']

colors = [bmh_colors['blue'], 'k', bmh_colors['green']]
x = list_days

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(25,5), sharey=False, sharex=True)

plt.style.use('ggplot')

    

n = 0



for ax in fig.get_axes():

    ax.scatter(x, ys[n], s=1, color=colors[n])

    ax.set_title(titles[n])



    date_format = mdates.DateFormatter('%m')

    ax.set_xlabel('Dates')

    ax.xaxis.set_major_formatter(date_format)



    n = n + 1



plt.show()
def estimate_b0_b1(x, y):

    n = np.size(x)



    #Obtain average of x and y

    mean_x, mean_y = np.mean(x), np.mean(y)



    #Calculate sumatory of XY and XX

    sum_xy = np.sum((x - mean_x)*(y - mean_y))

    sum_xx = np.sum((x*(x - mean_x)))



    #Regresion coeficents

    b_1 = sum_xy / sum_xx

    b_0 = mean_y - b_1*mean_x



    b = [b_0, b_1]



    return b



def linear_regression(x, y):

    if type(x[0]) == datetime:

        x = map(datetime.toordinal, x)

        x = list(x)



    b = estimate_b0_b1(x, y)



    y_pred = [b[0] + b[1]*x[x_i] for x_i in range(len(x))]



    #Graficación

    plot_regression(x, y,y_pred)





def plot_regression(x, y, y_pred, x_labels=None, y_labels=None, titles=None):

    fig, ax = plt.subplots(figsize=(7,5))

    plt.style.use('ggplot')

    

    plt1 = ax.scatter(x, y, label='Data', color = 'g', marker='o', s=10)



    try:

        plt2 = sns.regplot(x, y_pred, label='Linear Regresion Model', 

                        scatter_kws={'s':1},

                        line_kws={"color":"r","alpha":0.3,"lw":10})

        

    except:

        plt.cla()

        plt1 = ax.scatter(x, y, label='Data', color = 'g', marker='o', s=10)

        plt3 = ax.plot(x, y_pred, label='Linear Regresion Model', color ='b')

        



    #X axis format

    date_format = mdates.DateFormatter('%m')

    ax.set_xlabel('Dates')

    ax.xaxis.set_major_formatter(date_format)



    #Labels

    if x_labels == None:

        ax.set_xlabel('2020 months')

    else:

        pass



    if y_labels == None:  

        ax.set_ylabel('Confirmed Cases / 10,000,000')

    else:

        pass



    if titles == None: 

        ax.set_title('COVID-19 Confirmed Cases')

    

    plt.legend()



    plt.show()
x = list_days

y = ys[0]



linear_regression(x,y)
april_pos = 0

while list_days[april_pos].month < 4:

    april_pos += 1



april_pos
x = list_days[70:]

y = ys[0][70:]



linear_regression(x,y)
def obtain_alpha_beta(x, y):

    #Iterables

    x_2 = [x[i]**2 for i in range(len(x))]

    log_y = [math.log(y[i], 10) for i in range(len(y))]

    x_log_y = [x[i]*log_y[i] for i in range(len(log_y))]

    n = len(x)



    #Summatories

    sum_x = sum(x)

    sum_x_2 = sum(x_2)



    sum_log_y = sum(log_y)

    sum_x_log_y = sum(x_log_y)



    #ALPHA



    #Matrix Solver

    #Numerador

    Mn_00_11_alpha = sum_log_y * sum_x_2

    Mn_01_10_alpha = sum_x_log_y * sum_x

    #Denominador

    Md_00_11 = n * sum_x_2

    Md_01_10 = sum_x * sum_x



    numerador_alpha = Mn_00_11_alpha - Mn_01_10_alpha

    denominador = Md_00_11 - Md_01_10

    log_alpha = numerador_alpha/denominador



    #BETHA

    #Numerador

    Mn_00_11_betha = n * sum_log_y

    Mn_01_10_betha = sum_x * sum_x_log_y



    numerador_betha = Mn_00_11_betha - Mn_01_10_betha

    log_betha = numerador_betha/denominador





    alpha = 10**log_alpha

    betha = 10**log_betha



    return alpha, betha, log_alpha, log_betha



def exponencial_regression(x, y):

    if type(x[0]) == datetime:

        ordinal_x = map(datetime.toordinal, x)

        ordinal_x = list(ordinal_x)

    else:

        ordinal_x = x



    alpha, betha, log_alpha, log_betha = obtain_alpha_beta(ordinal_x, y)



    y_pred = [alpha*betha**ordinal_x[x_i] for x_i in range(len(ordinal_x))]



    #Graficación

    plot_regression(x, y,y_pred)

    

    print(f'Los valores de log_alpha = {log_alpha}, log_betha = {log_betha}')

    print(f'Los valores de alpha = {alpha}, betha = {betha}')
x = list_days[70:]

y = ys[0][70:]



exponencial_regression(x,y)
def solve_for_m_2(x, y):

    if type(x[0]) == datetime:

        x = map(datetime.toordinal, x)

        x = list(x)



    #SUM X

    n = len(x)

    sum_x = sum(x)

    sum_x2 = sum([x[i]**2 for i in range(len(x))])

    sum_x3 = sum([x[i]**3 for i in range(len(x))])

    sum_x4 = sum([x[i]**4 for i in range(len(x))])



    #SUM Y

    sum_y = sum(y)

    sum_xy = sum([x[i]*y[i] for i in range(len(x))])

    sum_x2y = sum([x[i]**2*y[i] for i in range(len(x))])



    #Matrix row

    r1 = [n , sum_x, sum_x2]

    r2 = [sum_x, sum_x2, sum_x3]

    r3 = [sum_x2, sum_x3, sum_x4]



    A = np.matrix([r1, r2, r3], dtype='float')

    b = np.matrix([sum_y, sum_xy, sum_x2y], dtype='float').reshape(-1,1)

    

    A_prime = np.linalg.solve(A,b)



    return A_prime



def polinomial_regression(x, y):



    if type(x[0]) == datetime:

        x_m = [n for n in range(len(x))]

    else:

        x_m = x



    A_prime = solve_for_m_2(x_m, y)

    a0 = A_prime[0,0]

    a1 = A_prime[1,0]

    a2 = A_prime[2,0]



    y_pred_pol = [(a0 + a1*x_m[i] + a2*x_m[i]**2) for i in range(len(x_m))]



    #Graficación

    plot_pol_regression(x, y, y_pred_pol)



def plot_pol_regression(x, y, y_pred, x_labels=None, y_labels=None, titles=None):

    

    fig, ax = plt.subplots(figsize=(7,5))

    plt.style.use('ggplot')

    

    data_sctr = ax.scatter(x, y, label='Data', color = 'g', marker='o', s=10)

    mdl_plt = ax.plot(x, y_pred, label='Model',marker='o', markersize=1 , color = 'r', alpha=0.2, linewidth=10)

    #X axis format

    date_format = mdates.DateFormatter('%m')

    ax.set_xlabel('Dates')

    ax.xaxis.set_major_formatter(date_format)



    #Labels

    if x_labels == None:

        ax.set_xlabel('2020 months')

    else:

        pass



    if y_labels == None:  

        ax.set_ylabel('Confirmed Cases / 10,000,000')

    else:

        pass



    if titles == None: 

        ax.set_title('COVID-19 Confirmed Cases')

    

    plt.legend()



    plt.show()
x = list_days[70:]

y = ys[0][70:]



polinomial_regression(x,y)

print('\n')



x = list_days

y = ys[0]



polinomial_regression(x,y)

year = end_day.year

month = end_day.month + 3

pred_month = calendar.month_name[month]



day = end_day.day



prediction_day = date(year, month, day)

start_day = date(year=2020, month=1, day=22)



prediction_days = prediction_day - start_day

n_prediction_days = prediction_days.days



list_days_pred = [start_day + timedelta(days=x) for x in range(n_prediction_days)]
def polinomial_prediction(x, y, x_prediction):



    x_m = [n for n in range(len(x))]



    A_prime = solve_for_m_2(x_m, y)

    a0 = A_prime[0,0]

    a1 = A_prime[1,0]

    a2 = A_prime[2,0]



    x_m_p = [n for n in range(len(x_prediction))]



    y_pred_pol = [(a0 + a1*x_m_p[i] + a2*x_m_p[i]**2) for i in range(len(x_m_p))]



    return y_pred_pol



def plot_prediction(x, y, y_pred, x_pred, x_labels=None, y_labels=None, titles=None):

    

    fig, ax = plt.subplots(figsize=(10,6))



    mdl_plt = ax.plot(x_pred, y_pred, label='Model',marker='o', markersize=1 , color = 'r', alpha=0.2, linewidth=10)

    data_sctr = ax.scatter(x, y, label='Data', color = 'g', marker='o', s=10)



    #Labels

    ax.set_ylabel('Confirmed Cases / 10,000,000')

    ax.set_title('COVID-19 Confirmed Cases')

    

    pred_number = '{:,}'.format(int(y_pred[-1]))

    pred_date = datetime.strftime(x_pred[-1], '%d-%B-%Y')

    

    #X axis format

    txt = f'''2020 Months



Pronósitco: {pred_number}* Para el día: {pred_date}

*En caso de no haber una vacuna'''



    date_format = mdates.DateFormatter('%m')

    ax.xaxis.set_major_formatter(date_format)

    ax.set_xlabel(txt)



    plt.legend(loc='best')



    plt.show()
def plot_cases(x, ys, y_pred, x_pred, labels, colors, country=None):

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(25,5), sharey=False, sharex=True)

    n = 0

    plt.style.use('ggplot')



    for ax in fig.get_axes():

        mdl_plt = ax.plot(x_pred, y_pred[n], label='Model',marker='o', markersize=1 , color = 'r', alpha=0.2, linewidth=10)

        data_sctr = ax.scatter(x, ys[n], label='Data', color = colors[n], marker='o', s=2)

        

        ax.set_ylabel(f'{labels[n]} Cases / 1e^n')

        if country == None:

            ax.set_title(f'COVID-19 {labels[n]} World Cases')

        else:

            ax.set_title(f'COVID-19 {labels[n]} {country} Cases ')



        pred_number = '{:,}'.format(int(y_pred[n][-1]))

        pred_date = datetime.strftime(x_pred[-1], '%d-%B-%Y')



        if country == None:

            txt = f'''2020 Months



    World Prediction: {pred_number}* {labels[n]} at: {pred_date}

    *In case there is no vaccine'''



        else:

            txt = f'''2020 Months



    {country} Prediction: {pred_number}* {labels[n]} at: {pred_date}

    *In case there is no vaccine'''



        date_format = mdates.DateFormatter('%m')

        ax.xaxis.set_major_formatter(date_format)

        ax.set_xlabel(txt)



        mdl_plt = mpatches.Patch(color='red', label='Model', alpha=0.2)

        ax.legend([mdl_plt, data_sctr], ['Model', 'Data'], loc='best')



        n = n + 1



    plt.show()
x = list_days

ys = [sum_confirmed, sum_deaths, sum_recovered]



x_pred = list_days_pred

y_pred = [polinomial_prediction(x, ys[n], x_pred) for n in range(len(ys))]



labels = ['Confirmed', 'Deaths', 'Recovered']



plot_cases(x, ys, y_pred, x_pred, labels, colors)
country = df['Country/Region'].to_list()



mx_confirmed = [0] * (n_days)

mx_deaths = [0] * (n_days)

mx_recovered = [0] * (n_days)



day_subregion = [None] * (n_days)

day_continent = [None] * (n_days)



plus = 0

for d in range(len(list_days)):



    while dates[plus] == list_days[d] and (d + plus) < len(dates):

        '''---BY DAY---'''

        #General

        if country[d + plus] == 'Mexico':

            mx_confirmed[d] = int(mx_confirmed[d]) + int(confirmed[d + plus])

            mx_deaths[d] = int(mx_deaths[d]) + int(deaths[d + plus])

            mx_recovered[d] = int(mx_recovered[d]) + int(recovered[d + plus])

        

        plus += 1
x = list_days[70:]

y_mx = [mx_confirmed[70:], mx_deaths[70:], mx_recovered[70:]]



x_pred = list_days_pred[70:]

y_pred_mx = [polinomial_prediction(x, y_mx[n], x_pred) for n in range(len(y_mx))]



plot_cases(x, y_mx, y_pred_mx, x_pred, labels, colors, 'Mexico')
x = list_days[70:]

y_mx = [mx_confirmed[70:], mx_deaths[70:], mx_recovered[70:]]



x_pred = list_days_pred[70:]

ys = [sum_confirmed[70:], sum_deaths[70:], sum_recovered[70:]]

y_pred = [polinomial_prediction(x, ys[n], x_pred) for n in range(len(ys))]

y_pred_mx = [polinomial_prediction(x, y_mx[n], x_pred) for n in range(len(y_mx))]



labels = ['Confirmed', 'Deaths', 'Recovered']

mx_colors = [bmh_colors['blue'], 'k', bmh_colors['green']]



fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(25,5), sharey=False, sharex=True)

n = 0

plt.style.use('ggplot')



for ax in fig.get_axes():

    world_mld_plt = ax.plot(x_pred, y_pred[n], label='World Model',marker='o', markersize=1 , color = 'r', alpha=0.2, linewidth=10)

    mx_mdl_plt = ax.plot(x_pred, y_pred_mx[n], label='Mx Model',marker='o', markersize=1 , color = bmh_colors['orange'], alpha=0.2, linewidth=10)



    data_sctr = ax.scatter(x, ys[n], label='World Data', color = colors[n], marker='o', s=2)

    mx_sctr = ax.scatter(x, y_mx[n], label='Mx Data', color = mx_colors[n], marker='o', s=2)

    

    ax.set_ylabel(f'{labels[n]} Cases / 1e^n')

    ax.set_title(f'COVID-19 {labels[n]} Cases')



    pred_number = '{:,}'.format(int(y_pred[n][-1]))

    pred_date = datetime.strftime(x_pred[-1], '%d-%B-%Y')



    pred_number_mx = '{:,}'.format(int(y_pred_mx[n][-1]))



    txt = f'''2020 Months



World Prediction: {pred_number}* {labels[n]}

Mx Prediction:  {pred_number_mx}* {labels[n]}

At: {pred_date}



*In case there is no vaccine'''



    date_format = mdates.DateFormatter('%m')

    ax.xaxis.set_major_formatter(date_format)

    ax.set_xlabel(txt)





    n = n + 1

    

    ax.legend(loc='best')



plt.show()