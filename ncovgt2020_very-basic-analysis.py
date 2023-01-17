# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot  as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dir_pacientes = '/kaggle/input/covid19-guatemala/patients.csv'

pacientes = pd.read_csv(dir_pacientes, header = 0, parse_dates = True)



# format dates

cols_fecha = ["birth_date","confirmation_date", "recovery_date", "deceased_date"]

for col in cols_fecha:

    pacientes[col] = pd.to_datetime(pacientes[col], yearfirst = True)



# display columns and their datatypes

print(pacientes.info())
# display some data

print(pacientes.head())
# Some estatistics

confirmados = pacientes['confirmation_date'].value_counts().sort_index()

recuperados = pacientes['recovery_date'].value_counts().sort_index()

fallecidos = pacientes['deceased_date'].value_counts().sort_index()

print("Confirmed: {}".format(confirmados.sum()))

print("Recovered: {}".format(recuperados.sum()))

print("Deceased: {}".format(fallecidos.sum()))
fig1 = plt.figure(figsize=(5,5))

fig1ax1 = fig1.add_subplot(1,1,1)

fig1ax1.set_title('Confirmed Cases')

confirmados.cumsum().plot(ax = fig1ax1, color = 'r', label = 'cumulative', marker='.',markersize=10)

confirmados.plot( ax = fig1ax1, color = 'b', label = 'daily')#, marker='x',markersize=10)

fig1ax1.set_xticks(confirmados.index.to_list())

fig1ax1.set_yticks(range(confirmados.sum()+2))

fig1ax1.legend(loc="best")
fig2 = plt.figure(figsize=(5,5))

fig2ax1 = fig2.add_subplot(1,1,1)

fig2ax1.set_title('Recovered cases')

recuperados.cumsum().plot(ax = fig2ax1, color = 'r', label = 'cumulative', marker='.',markersize=10)

recuperados.plot(ax = fig2ax1, color = 'b', label = 'daily', marker='x',markersize=10)

fig2ax1.set_xticks(recuperados.index.to_list())

fig2ax1.set_yticks((range(min(recuperados),max(recuperados)+2)))

fig2ax1.legend(loc = "best")
fig3 = plt.figure(figsize=(5,5))

fig3ax1 = fig3.add_subplot(1,1,1)

fig3ax1.set_title('Casos fallecidos')

fallecidos.cumsum().plot(ax = fig3ax1, color = 'r', label = 'acumulados', marker='.',markersize=10)

fallecidos.plot(ax = fig3ax1, color = 'b', label = 'diarios', marker='x',markersize=10)

fig3ax1.set_xticks(fallecidos.index.to_list())

fig2ax1.set_yticks(range(fallecidos.sum()+2))

fig3ax1.legend(loc = 'best')
# Confirmed cases

confirmados = pacientes[[ not pd.isna(i) for i in pacientes['confirmation_date']]]

ax = confirmados['sex'].value_counts().plot.pie(y='sex', legend = True, autopct='%2.0f%%', figsize = (5,5), title = 'Confirmed by sex')
recuperados = pacientes[[ not pd.isna(i) for i in pacientes['recovery_date']]]

ax = recuperados['sex'].value_counts().plot.pie(y='sex', legend = True, autopct='%.2f', figsize = (5,5), title = 'Recovered by sex')
fallecidos = pacientes[[ not pd.isna(i) for i in pacientes['deceased_date']]]

ax = fallecidos['sex'].value_counts().plot.pie(y='sex', legend = True, autopct='%.2f', figsize = (5,5), title = 'Deceased by sex')