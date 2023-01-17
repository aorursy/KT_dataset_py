import numpy as np

import pandas as pd

import datetime

import math

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dateParser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')



clima = pd.read_csv('/kaggle/input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv', 

                    parse_dates=['dt'],

                    date_parser=dateParser,

                    low_memory=False

                   )
clima.head()
clima.columns = ['fecha', 'temp', 'incertidumbre', 'pais']
clima.fecha.describe()
mexico = pd.DataFrame(clima[clima.pais == 'Mexico'])

mexico.head()
mexico['year'] = pd.DatetimeIndex(mexico['fecha']).year

mexico['mes'] = pd.DatetimeIndex(mexico['fecha']).month
mexico['decada'] = (pd.DatetimeIndex(mexico.fecha).year-1)/10

mexico.decada = mexico.decada.apply(lambda x: math.floor(x))*10
mexico.head()
plt.style.use('fivethirtyeight')
def temppormes(rango):

    plt.figure(figsize=(20, 8))

    for year in rango:

        ini = datetime.datetime.strptime('{}-01-01'.format(year), '%Y-%m-%d')

        fin = datetime.datetime.strptime('{}-12-31'.format(year), '%Y-%m-%d')



        rango = mexico[(mexico.fecha >= ini) & (mexico.fecha <= fin)]



        plt.plot(rango.mes, rango.temp, label=year)





    # plt.legend(loc='best')

    plt.title('Comparativa de temperaturas')

    plt.xlabel('Fecha')

    plt.ylabel('Temp °C')

print()
years = [1835, 1950, 2013]

temppormes(years)
temppormes(range(np.min(mexico.year), np.max(mexico.year)))
meant = mexico.groupby(['year'])['temp'].mean()
plt.figure(figsize=(20, 8))

plt.title('Temperatura promedio anual')



x = np.arange(0, len(meant))

xrange = np.arange(0, len(meant)+10, 10)



coef = np.polyfit(x, meant, 1)

poly1d_fn = np.poly1d(coef)



plt.plot(x, meant)

plt.plot(x, poly1d_fn(x), ls=":")



plt.xticks(xrange, np.arange(1835, 2020, 10), rotation=90)

plt.xlabel('Década')

plt.ylabel('Temp °C')

print()
meant = mexico.groupby(['decada'])['temp'].mean()
plt.figure(figsize=(20, 10))

plt.title('Temperatura promedio anual por década')

x = np.array(range(1, int(np.floor((2010-1820)/10)+1)))



coef = np.polyfit(x, meant, 1)

poly1d_fn = np.poly1d(coef)



plt.plot(x, meant, marker='o', mec='darkred', mfc='orange', mew=2)

plt.plot(x, poly1d_fn(x), ls=":")



plt.xticks(x, np.arange(1830, 2020, 10), rotation=90)

plt.xlabel('Década')

plt.ylabel('Temp °C')

print()
plt.figure(figsize=(20, 10))

g = sns.boxplot(x='decada', y='temp', data=mexico, color="teal", linewidth=2)

g.set_xticklabels(g.get_xticklabels(), rotation=90)



g.set_xlabel("Década")

g.set_ylabel("Temp°C")

g.set_title('Rango de Temperaturas por Década')

print()
line_color = ['red',    'blue',   'red',    'green', 'goldenrod',     'orange',

              'orchid', 'indigo', 'brown', 'black', 'cadetblue', 'darkslategrey'

             ]

nombre_meses = ['Enero', 'Febrero', 'Marzo',      'Abril',   'Mayo',      'Junio',

                'Julio', 'Agosto',  'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'

               ]
def mesporanos(meses, lw=4):

    for mes in meses:

        rango = mexico[mexico.mes == mes]

        x=np.arange(1, len(rango)+1)



        coef = np.polyfit(x, rango.temp, 1)

        poly1d_fn = np.poly1d(coef)



        plt.plot(x, rango.temp, c=line_color[mes-1], label=nombre_meses[mes-1], lw=lw)

        plt.plot(x, poly1d_fn(x), c=line_color[mes-1], ls=":", lw=lw)



        plt.xticks(np.arange(1, len(rango)+10, 10), np.arange(1835, 2020, 10), rotation=90)





    plt.legend(loc='best')

    plt.xlabel('Año')

    plt.ylabel('Temp °C')



    plt.tight_layout(pad=4)

    plt.suptitle('Comparativa de temperaturas')
plt.figure(figsize=(20, 12))

meses = [7]



mesporanos(meses)
plt.figure(figsize=(20, 12))

meses = [3, 6, 9, 12]



mesporanos(meses)
plt.figure(figsize=(15, 20))



charts = [[6, 7, 8],

          [5, 9],

          [4, 10],

          [3, 11],

          [2, 12]]



for i in range(len(charts)):



    plt.subplot(321+i)

    meses = charts[i]



    mesporanos(meses, lw=1)



plt.tight_layout(pad=4)

plt.suptitle('Comparativa de temperaturas')

print()