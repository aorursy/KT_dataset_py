import os

print(os.listdir("../input"))

import pandas as pd

import plotly

from plotly.offline import iplot, init_notebook_mode

import plotly.express as px

from datetime import datetime



init_notebook_mode()
df_all_history = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')



df_all_history.head()
df_all_history.info()
df_all = df_all_history





df_all['Dates'] = df_all_history['Date']





df_all['Date'] = pd.to_datetime(df_all['Date'])
covid_19_countries = (df_all_history.groupby("Country/Region")["ConfirmedCases","Fatalities"].sum()

                      .sort_values(by = "Fatalities",ascending = False))

print(covid_19_countries)
df_oversea = df_all

df_oversea.fillna(value="", inplace=True)



df_oversea
fig_oversea = px.line(df_oversea, x='Dates', y='ConfirmedCases',

                      line_group='Country/Region',

                      color='Country/Region',

                      color_discrete_sequence=px.colors.qualitative.D3,

                      hover_name='Country/Region',

)



fig_oversea.show()
df_all1 = df_all_history

df_all1['Country'] = df_all_history['Country/Region']
df_oversea1 = df_all1

df_oversea1.fillna(value="", inplace=True)



df_oversea1
%matplotlib inline

import scipy.integrate as spi

import numpy as np

import pylab as pl

alpha=1.4247

beta=0.14286

TS=1.0 #观察间隔

ND=15.0 #观察结束日期

S0=1-1e-6 #初始易感人数

I0=1e-6 #初始感染人数

INPUT = (S0, I0, 0.0)

def diff_eqs(INP,t):

    '''The main set of equations'''

    Y=np.zeros((3))

    V = INP

    Y[0] = - alpha * V[0] * V[1]

    Y[1] = alpha * V[0] * V[1] - beta * V[1]

    Y[2] = beta * V[1]

    return Y

t_start = 0.0

t_end = ND

t_inc = TS

t_range = np.arange(t_start, t_end+t_inc, t_inc) #生成日期范围

RES = spi.odeint(diff_eqs,INPUT,t_range)



pl.subplot(111)

pl.plot(RES[:,0], '-g', label='Susceptible')

pl.plot(RES[:,1], '-r', label='Infective')

pl.plot(RES[:,2], '-k', label='Removal')

pl.legend(loc=0)

pl.title('SIR_Model')

pl.xlabel('Time')

pl.ylabel('Numbers')

pl.xlabel('Time')

        