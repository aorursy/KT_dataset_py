# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from IPython.core.display import HTML

import folium

import datetime

from datetime import datetime

import requests

from bs4 import BeautifulSoup

import lxml.html as lh

import pandas as pd

import re

import time

import psutil

import json



import numpy as np

from PIL import Image

import os

from os import path

import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp



import plotly.graph_objects as go

from pandas.plotting import register_matplotlib_converters

import plotly.express as px

from IPython.display import display, Markdown, Latex

import matplotlib as plot

from matplotlib.pyplot import figure

import seaborn as sns



register_matplotlib_converters()

from IPython.display import Markdown

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
titanicdf = pd.read_csv('/kaggle/input/titanic/train.csv')



#3,711

totalTitanicPop = 3300

totalDiamondPop = 3711

display(Markdown("#### **Total population in Titanic :**{}".format(totalTitanicPop)))

display(Markdown("#### **but we have details of only **{} **people**".format(max(titanicdf['PassengerId'])," people!!!")))



#titanicPeople = []

titanicPeopledf = pd.DataFrame()





titanicPeopledf['id'] = titanicdf['PassengerId']

titanicPeopledf['pClass'] = titanicdf['Pclass'] 

titanicPeopledf['sex'] = titanicdf['Sex']

titanicPeopledf['age'] = titanicdf['Age']

titanicPeopledf['cabin'] = titanicdf['Cabin']



immunitylist = []

incubationlist = []

infectedlist = []

quarantinedlist = []

alivelist = []



for person in range(max(titanicdf['PassengerId'])):



    if titanicdf.iloc[person]['Age'] <20:

        immunity = 0.9

    elif titanicdf.iloc[person]['Age'] <40 and titanicdf.iloc[person]['Age'] >20:

        immunity = 0.7

    elif titanicdf.iloc[person]['Age'] <60 and titanicdf.iloc[person]['Age'] >40:

        immunity = 0.5

    elif titanicdf.iloc[person]['Age'] <100 and titanicdf.iloc[person]['Age'] >60:

        immunity = 0.2

    immunitylist.append(immunity)

    incubationlist.append(0)

    infectedlist.append(False)

    quarantinedlist.append(False)

    alivelist.append(True)

    

    

titanicPeopledf['immunity'] = immunitylist

titanicPeopledf['incubation'] = incubationlist

titanicPeopledf['infected'] = infectedlist

titanicPeopledf['quarantined'] = quarantinedlist

titanicPeopledf['alive'] = alivelist

titanicPeopledf
totalTitanicHosp = 2

totalTItanicBed = 12+6+4

totalTitanicMedicalQulified = 300
def base_seir_model(init_vals, params, t):

    S_0, E_0, I_0, R_0 = init_vals

    S, E, I, R = [S_0], [E_0], [I_0], [R_0]

    alpha, beta, gamma = params

    dt = t[1] - t[0]

    for _ in t[1:]:

        next_S = S[-1] - (beta*S[-1]*I[-1])*dt

        next_E = E[-1] + (beta*S[-1]*I[-1] - alpha*E[-1])*dt

        next_I = I[-1] + (alpha*E[-1] - gamma*I[-1])*dt

        next_R = R[-1] + (gamma*I[-1])*dt

        S.append(next_S)

        E.append(next_E)

        I.append(next_I)

        R.append(next_R)

    return [S,E, I,R]
# Define parameters

t_max = 50

dt = .1

t = np.linspace(0, t_max, int(t_max/dt) + 1)

N = 3300

init_vals = 1 - 1/N, 1/N, 0, 0

alpha = 0.2

beta = 10

gamma = 0.5

params = alpha, beta, gamma

# Run simulation

results = base_seir_model(init_vals, params, t)

#results



fig = go.Figure()

#fig = px.scatter(testingHistory,x="time_stamp", y="testing_no")



fig = fig.add_trace(go.Scatter(x=t, y=results[0],

                    mode='lines',

                    name='Susceptibles'))

fig = fig.add_trace(go.Scatter(x=t, y=results[1],

                    mode='lines',

                    name='Exposed'))

fig = fig.add_trace(go.Scatter(x=t, y=results[2],

                    mode='lines',

                    name='Infectious'))

fig = fig.add_trace(go.Scatter(x=t, y=results[3],

                    mode='lines',

                    name='Recovered'))

fig = fig.update_layout(

    yaxis_title ="Populaton Fraction",

    xaxis_title ="Days"  

)



fig.show()
def base_seird_model(init_vals, params, t):

    S_0, E_0, I_0, R_0,D_0 = init_vals

    S, E, I, R,D = [S_0], [E_0], [I_0], [R_0], [D_0]

    alpha, beta, gamma, myu = params

    dt = t[1] - t[0]

    for _ in t[1:]:

        next_S = S[-1] - (beta*S[-1]*I[-1])*dt

        next_E = E[-1] + (beta*S[-1]*I[-1] - alpha*E[-1])*dt

        next_I = I[-1] + (alpha*E[-1] - gamma*I[-1])*dt - (myu*I[-1])*dt

        next_R = R[-1] + (gamma*I[-1])*dt

        next_D = D[-1] + (myu*I[-1])*dt

        S.append(next_S)

        E.append(next_E)

        I.append(next_I)

        R.append(next_R)

        D.append(next_D)

    return [S,E, I,R,D]
# Define parameters

t_max = 50

dt = .1

t = np.linspace(0, t_max, int(t_max/dt) + 1)

N = 3300

init_vals = 1 - 1/N, 1/N, 0, 0,0

alpha = 0.2

beta = 10

gamma = 0.5

myu = 0.1

params = alpha, beta, gamma, myu

# Run simulation

results = base_seird_model(init_vals, params, t)

#results



fig = go.Figure()

#fig = px.scatter(testingHistory,x="time_stamp", y="testing_no")



fig = fig.add_trace(go.Scatter(x=t, y=results[0],

                    mode='lines',

                    name='Susceptibles'))

fig = fig.add_trace(go.Scatter(x=t, y=results[1],

                    mode='lines',

                    name='Exposed'))

fig = fig.add_trace(go.Scatter(x=t, y=results[2],

                    mode='lines',

                    name='Infectious'))

fig = fig.add_trace(go.Scatter(x=t, y=results[3],

                    mode='lines',

                    name='Recovered'))

fig = fig.add_trace(go.Scatter(x=t, y=results[4],

                    mode='lines',

                    name='Dead'))

fig = fig.update_layout(

    yaxis_title ="Populaton Fraction",

    xaxis_title ="Days"  

)



fig.show()
# Define parameters

t_max = 100

dt = .1

t = np.linspace(0, t_max, int(t_max/dt) + 1)

N = 3300

init_vals = 1 - 1/N, 1/N, 0, 0

alpha = 0.2

beta = 1

gamma = 0.5

params = alpha, beta, gamma

# Run simulation

results = base_seir_model(init_vals, params, t)

#results



fig = go.Figure()

#fig = px.scatter(testingHistory,x="time_stamp", y="testing_no")



fig = fig.add_trace(go.Scatter(x=t, y=results[0],

                    mode='lines',

                    name='Susceptibles'))

fig = fig.add_trace(go.Scatter(x=t, y=results[1],

                    mode='lines',

                    name='Exposed'))

fig = fig.add_trace(go.Scatter(x=t, y=results[2],

                    mode='lines',

                    name='Infectious'))

fig = fig.add_trace(go.Scatter(x=t, y=results[3],

                    mode='lines',

                    name='Recovered'))

fig = fig.update_layout(

    yaxis_title ="Populaton Fraction",

    xaxis_title ="Days"  

)



fig.show()
# Define parameters

t_max = 150

dt = .1

t = np.linspace(0, t_max, int(t_max/dt) + 1)

N = 3300

init_vals = 1 - 1/N, 1/N, 0, 0,0

alpha = 0.2

beta = 1

gamma = 0.5

myu = 0.1

params = alpha, beta, gamma, myu

# Run simulation

results = base_seird_model(init_vals, params, t)

#results



fig = go.Figure()

#fig = px.scatter(testingHistory,x="time_stamp", y="testing_no")



fig = fig.add_trace(go.Scatter(x=t, y=results[0],

                    mode='lines',

                    name='Susceptibles'))

fig = fig.add_trace(go.Scatter(x=t, y=results[1],

                    mode='lines',

                    name='Exposed'))

fig = fig.add_trace(go.Scatter(x=t, y=results[2],

                    mode='lines',

                    name='Infectious'))

fig = fig.add_trace(go.Scatter(x=t, y=results[3],

                    mode='lines',

                    name='Recovered'))

fig = fig.add_trace(go.Scatter(x=t, y=results[4],

                    mode='lines',

                    name='Dead'))

fig = fig.update_layout(

    yaxis_title ="Populaton Fraction",

    xaxis_title ="Days"  

)



fig.show()