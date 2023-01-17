# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import datetime

import seaborn as sns

datos=pd.read_csv("../input/airplane-crash-data-since-1908/Airplane_Crashes_and_Fatalities_Since_1908_20190820105639.csv")
datos['year'] = pd.DatetimeIndex(datos['Date']).year

datos.head()
x=datos.loc[:,['year']]

y=datos.loc[:,['Fatalities']]
years_fatalities=datos.loc[datos['year'] & datos['Fatalities']]

fatalities_over_the_years=sns.lineplot(x=years_fatalities['year'],y=years_fatalities['Fatalities'])

fatalities_over_the_years.set_title("Muertes causadas por accidentes aéreos")

fatalities_over_the_years.set_xlabel('Años')

fatalities_over_the_years.set_ylabel('Número de muertes')
datos['Sobrevivientes'] = datos['Aboard']-datos['Fatalities']

numerosobrevivientes=datos['Sobrevivientes']

bin_edges=[0,1,2,3,4,5,6,7,8,9,10]

plt.hist(numerosobrevivientes, bins=bin_edges,color='r')

plt.xlabel('Número de sobrevivientes')

plt.ylabel('Número de accidentes')

plt.title('Distribución del número de sobrevivientes')