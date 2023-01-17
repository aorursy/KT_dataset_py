import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline
poitivos_filepath = '../input/peru-covid19-august-2020/positivos_covid.csv'

fallecidos_filepath = '../input/peru-covid19-august-2020/fallecidos_covid.csv'







peru_pos = pd.read_csv(poitivos_filepath, 

                    encoding='latin-1', parse_dates=['FECHA_RESULTADO'])

peru_fall = pd.read_csv(fallecidos_filepath, 

                    encoding='latin-1', parse_dates=['FECHA_FALLECIMIENTO'])
peru_fall.head()
peru_fall.isna().sum()
peru_pos.isna().sum()
fig = plt.figure(figsize= (16, 8))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)



peru_pos.groupby("DEPARTAMENTO").DEPARTAMENTO.count().sort_values().plot.pie(cmap="tab20c", ax=ax1)

ax1.set_title("Postive Cases in Peru by Department")

peru_fall.groupby("DEPARTAMENTO").DEPARTAMENTO.count().sort_values().plot.pie(cmap="tab20c", ax=ax2)

ax2.set_title("Deaths in Peru by Department")
puno_pos = peru_pos[peru_pos.DEPARTAMENTO == "PUNO"]

puno_fall = peru_fall[peru_fall.DEPARTAMENTO == "PUNO"]
fig = plt.figure(figsize=(16, 8))



ax1 = fig.add_subplot(211)

ax2 = fig.add_subplot(212)





peru_pos.groupby("FECHA_RESULTADO").FECHA_RESULTADO.count().plot(kind="line", x = "FECHA_RESULTADO", ax=ax1, color = "darkolivegreen")

peru_fall.groupby("FECHA_FALLECIMIENTO").FECHA_FALLECIMIENTO.count().plot(kind="line", x = "FECHA_FALLECIMIENTO", ax=ax1, color= "darkorange")

ax1.set_title("Positive Cases and Deaths by Date")

ax1.set_facecolor("gainsboro")

ax1.set_xlabel('')



puno_pos.groupby("FECHA_RESULTADO").FECHA_RESULTADO.count().plot(kind="line", x = "FECHA_RESULTADO", ax=ax2, color = "darkolivegreen")

puno_fall.groupby("FECHA_FALLECIMIENTO").FECHA_FALLECIMIENTO.count().plot(kind="line", x = "FECHA_FALLECIMIENTO", ax=ax2, color= "darkorange")

ax2.set_title("Positive Cases and Deaths by Date in Puno")

ax2.set_facecolor("gainsboro")

ax2.set_xlabel('Date')
fig = plt.figure(figsize=(20, 20))

ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222)

ax3 = fig.add_subplot(223)

ax4 = fig.add_subplot(224)



peru_pos.EDAD.plot(kind='hist', y='EDAD', color = "darkolivegreen", bins=200, ax=ax1)

peru_fall.EDAD_DECLARADA.plot(kind='hist', y='EDAD', color= "darkorange", bins=200, ax=ax1)

ax1.set_xlim((0,100))

ax1.set_facecolor("gainsboro")

ax1.set_title("Peru Positive and Deaths by Age")



peru_fall.EDAD_DECLARADA.plot(kind='hist', y='EDAD', color= "darkorange", bins=200, ax=ax3)

ax3.set_xlim((0,100))

ax3.set_facecolor("gainsboro")

ax3.set_title("Peru Deaths Only by Age")



puno_pos.EDAD.plot(kind='hist', y='EDAD', color = "darkolivegreen", bins=200, ax=ax2)

puno_fall.EDAD_DECLARADA.plot(kind='hist', y='EDAD', color= "darkorange", bins=200, ax=ax2)

ax2.set_xlim((0,100))

ax2.set_facecolor("gainsboro")

ax2.set_title("Puno Positive and Deaths by Age")



puno_fall.EDAD_DECLARADA.plot(kind='hist', y='EDAD', color= "darkorange", bins=200, ax=ax4)

ax4.set_xlim((0,100))

ax4.set_facecolor("gainsboro")

ax4.set_title("Puno Deaths Only")
fig = plt.figure(figsize=(11, 11))

ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222)

ax3 = fig.add_subplot(223)

ax4 = fig.add_subplot(224)



peru_pos.groupby('SEXO').SEXO.count().plot.bar(ax=ax1, color='darkolivegreen')

ax1.set_facecolor("gainsboro")

ax1.set_title("Peru Positive Cases by Sex")



peru_fall.groupby('SEXO').SEXO.count().plot.bar(ax=ax2, color= "darkorange")

ax2.set_facecolor("gainsboro")

ax2.set_title("Peru Deaths by Sex")



puno_pos.groupby('SEXO').SEXO.count().plot.bar(ax=ax3, color='darkolivegreen')

ax3.set_facecolor("gainsboro")

ax3.set_title("Puno Positive Cases by Sex")



puno_fall.groupby('SEXO').SEXO.count().plot.bar(ax=ax4, color= "darkorange")

ax4.set_facecolor("gainsboro")

ax4.set_title("Puno Deaths by Sex")
posdata = peru_pos.groupby('SEXO').SEXO.count().rename(index={'FEMENINO': 'Positive F', 'MASCULINO': 'Positive M'}).rename_axis(['Sex'], axis=0)

dedata = peru_fall.groupby('SEXO').SEXO.count().rename(index={'FEMENINO': 'Deaths F', 'MASCULINO': 'Deaths M'}).rename_axis(['Sex'], axis=0)
fig = plt.figure(figsize=(16, 8))



ax1 = fig.add_subplot(211)

ax2 = fig.add_subplot(212)





peru_pos[peru_pos['FECHA_RESULTADO'] > '2020-07-01'].groupby("FECHA_RESULTADO").FECHA_RESULTADO.count().plot(kind="line", x = "FECHA_RESULTADO", ax=ax1, color = "darkolivegreen")

peru_fall[peru_fall['FECHA_FALLECIMIENTO'] > '2020-07-01'].groupby("FECHA_FALLECIMIENTO").FECHA_FALLECIMIENTO.count().plot(kind="line", x = "FECHA_FALLECIMIENTO", ax=ax1, color= "darkorange")

ax1.set_title("Positive Cases and Deaths by Date")

ax1.set_facecolor("gainsboro")

ax1.set_xlabel('')



puno_pos[puno_pos['FECHA_RESULTADO'] > '2020-07-01'].groupby("FECHA_RESULTADO").FECHA_RESULTADO.count().plot(kind="line", x = "FECHA_RESULTADO", ax=ax2, color = "darkolivegreen")

puno_fall[puno_fall['FECHA_FALLECIMIENTO'] > '2020-07-01'].groupby("FECHA_FALLECIMIENTO").FECHA_FALLECIMIENTO.count().plot(kind="line", x = "FECHA_FALLECIMIENTO", ax=ax2, color= "darkorange")

ax2.set_title("Positive Cases and Deaths by Date in Puno")

ax2.set_facecolor("gainsboro")

ax2.set_xlabel('Date')