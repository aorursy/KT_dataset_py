

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import statsmodels.api as sm

import math





confirmed = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

recovered = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

open_line_list = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")

data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

deaths = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

line_list_data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

data = data.rename(columns={'SNo': 'sno', 'ObservationDate': 'obs_date', 'Province/State': 'state', 

                            'Country/Region': 'country', 'Last Update': 'last_update', 

                            'Confirmed': 'confirmed', 'Deaths': 'deaths', 'Recovered': 'recovered'})
data['obs_date'] = pd.to_datetime(data['obs_date'], format='%m/%d/%Y')
data
first_case_per_country = data.groupby('country').obs_date.agg(['min']).rename(columns={'min': 'first_case'})

data = data.merge(first_case_per_country, on='country')

data['day_num_dt'] = data['obs_date'] - data['first_case']

data['day_num'] = data['day_num_dt'].dt.days

data = data.drop(columns=['day_num_dt'])



plt.style.use('seaborn-darkgrid')

palette = plt.get_cmap('Set1')

palette_num = 0



def draw_chart(data, x_col, y_col, x_col_lbl, y_col_lbl, legend, title):

    global palette_num

    plt.plot(data[x_col], data[y_col], marker='', color=palette(palette_num), linewidth=1, alpha=0.9, label=y_col)

    plt.legend(loc=2, ncol=2)

    plt.title(title, loc='left', fontsize=12, fontweight=0, color='orange')

    plt.xlabel(x_col_lbl)

    plt.ylabel(y_col_lbl)

    palette_num = 1 + palette_num

    
beijing = data[(data['country'] == 'Mainland China') & (data['state'] == 'Beijing')]

beijing
draw_chart(beijing, 'day_num', 'confirmed', 'Dia #', 'Casos Confirmados', 'Beijing', 'Beijing')
hubei = data[(data['country'] == 'Mainland China') & (data['state'] == 'Hubei')]

hubei

draw_chart(hubei, 'day_num', 'confirmed', 'Dia #', 'Casos Confirmados', 'Hubei', 'Hubei')
italy = data[(data['country'] == 'Italy') ]

italy
draw_chart(italy, 'day_num', 'confirmed', 'Dia #', 'Casos Confirmados', 'Italy', 'Italy')
spain = data[(data['country'] == 'Spain') ]

spain
draw_chart(spain, 'day_num', 'confirmed', 'Dia #', 'Casos Confirmados', 'Spain', 'Spain')
arg = data[(data['country'] == 'Argentina') ]

arg
draw_chart(arg, 'day_num', 'confirmed', 'Dia #', 'Casos Confirmados', 'Argentina', 'Argentina')
arg


def predict_confirmed(df, day_num):

    df.ln_confirmed = np.log(df.confirmed)

    X = df.day_num

    X = sm.add_constant(X)

    y = df.ln_confirmed

    mod = sm.OLS(y, X)

    res = mod.fit()

    x0 = res.params[0]

    b = res.params[1]

    xt = x0 + b * day_num

    return math.exp(xt)

    



print("Casos en Argentina para el 01 de Julio 2020 (dia 119) <-- extrapolacion espantosa, solo para jugar!!!", predict_confirmed(arg, 119))

print("Casos en Argentina para el 17 de Marzo 2020 (dia 14)", predict_confirmed(arg, 14))

print("Casos en Argentina para el 12 de Marzo 2020 (dia 9)", predict_confirmed(arg, 9))



pred = []

for i in range(0, len(arg)):

     pred.append(predict_confirmed(arg, i))



arg['pred'] = pred

draw_chart(arg, 'day_num', 'pred', 'Dia #', 'Casos Predichos', 'Argentina', 'Argentina')    


'''

arg.ln_confirmed = np.log(arg.confirmed)

X = arg.day_num.dt.days

X = sm.add_constant(X)

y = arg.ln_confirmed

mod = sm.OLS(y, X)

res = mod.fit()

print(res.summary())

'''