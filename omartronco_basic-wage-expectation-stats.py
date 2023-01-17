# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/2019_Spring.csv")

data
data.describe()
#print(data['Ending_college'].plot.hist())

#print(data['5_year_later'].plot.hist())

#print(data['10_years_later'].plot.hist())
sns.boxplot(data=data)
sns.violinplot(data=data)
data.plot.scatter(x='Ending_college', y='10_years_later')
data.std()/data.mean()
data.skew()
data.skew()/(data.std().pow(3))
data.kurtosis()
data.kurtosis()/(data.std().pow(4))
# Creamos un nuevo DataFrame con la columna ordenada

sorted0 = pd.Series.to_frame(data.sort_values(['Ending_college'], ascending=True)["Ending_college"])



# Creamos la columna en donde estará la función de media

# de pérdida en exceso para "Ending_college",

# MELF por sus siglas en inglés

sorted0["MELF0"] = 0



# Debido a que el índice mantuvo los valores originales

# reemplazamos el índice por un nuevo de 0 a n

sorted0.index = range(len(sorted0))



# A continuación con un bucle creamos cada elemento en MELF0

for i in range(0, len(sorted0)-1): # El -1 es porque el último dato no se puede calcular

    sorted0["MELF0"][i] = sorted0["Ending_college"][i+1:].mean() - sorted0["Ending_college"][i:i+1]



# Graficamos MELF0

sorted0["MELF0"].plot.line(title="MELF0")
# Para MELF5 

sorted5 = pd.Series.to_frame(data.sort_values(['5_year_later'], ascending=True)["5_year_later"])

sorted5["MELF5"] = 0

sorted5.index = range(len(sorted5))

for i in range(0, len(sorted5)-1):

    sorted5["MELF5"][i] = sorted5["5_year_later"][i+1:].mean() - sorted5["5_year_later"][i:i+1]

    

# Para MELF10 

sorted10 = pd.Series.to_frame(data.sort_values(['10_years_later'], ascending=True)["10_years_later"])

sorted10["MELF10"] = 0

sorted10.index = range(len(sorted10))

for i in range(0, len(sorted10)-1):

    sorted10["MELF10"][i] = sorted10["10_years_later"][i+1:].mean() - sorted10["10_years_later"][i:i+1]



# Graficamos las tres funciones

sorted0["MELF0"].plot.line(label="MELF0", title="MELF")

sorted5["MELF5"].plot.line(label="MELF5")

sorted10["MELF10"].plot.line(label="MELF10").legend()
print("Regresiones")

print("MELF0")

print(np.polyfit(sorted0.index, sorted0["MELF0"],1))

print("MELF5")

print(np.polyfit(sorted5.index, sorted5["MELF5"],1))

print("MELF10")

print(np.polyfit(sorted10.index, sorted10["MELF10"],1))
sorted0["MELF0"][0:len(sorted0)-1].plot.line(label="MELF0", title="MELF sin el último dato")

sorted5["MELF5"][0:len(sorted5)-1].plot.line(label="MELF5")

sorted10["MELF10"][0:len(sorted10)-1].plot.line(label="MELF10").legend()
print("Regresiones sin el último dato")

print("MELF0")

print(np.polyfit(sorted0.index[0:len(sorted0)-1], sorted0["MELF0"][0:len(sorted0)-1],1))

print("MELF5")

print(np.polyfit(sorted5.index[0:len(sorted5)-1], sorted5["MELF5"][0:len(sorted5)-1],1))

print("MELF10")

print(np.polyfit(sorted10.index[0:len(sorted10)-1], sorted10["MELF10"][0:len(sorted10)-1],1))
sorted0["MELF0"][1:len(sorted0)-1].plot.line(label="MELF0", title="MELF sin el último dato, y sin atípico bajo")

sorted5["MELF5"][1:len(sorted5)-1].plot.line(label="MELF5")

sorted10["MELF10"][1:len(sorted10)-1].plot.line(label="MELF10").legend()
print("Regresiones sin el último dato y sin el atípico bajo")

print("MELF0")

print(np.polyfit(sorted0.index[1:len(sorted0)-1], sorted0["MELF0"][1:len(sorted0)-1],1))

print("MELF5")

print(np.polyfit(sorted5.index[1:len(sorted5)-1], sorted5["MELF5"][1:len(sorted5)-1],1))

print("MELF10")

print(np.polyfit(sorted10.index[1:len(sorted10)-1], sorted10["MELF10"][1:len(sorted10)-1],1))
newsorted0 = sorted0.set_index('Ending_college')

newsorted5 = sorted5.set_index('5_year_later')

newsorted10 = sorted10.set_index('10_years_later')
newsorted0["MELF0"][1:len(newsorted0)-1].plot.line(label="MELF0", title="MELF sin el último dato, sin atípico bajo, y reescalado correctamente")

newsorted5["MELF5"][1:len(newsorted5)-1].plot.line(label="MELF5")

newsorted10["MELF10"][1:len(newsorted10)-1].plot.line(label="MELF10").legend()
print("Regresiones sin el último dato, sin el atípico bajo, y reescalado correctamente")

print("MELF0")

print(np.polyfit(newsorted0.index[1:len(newsorted0)-1], newsorted0["MELF0"][1:len(newsorted0)-1],1))

print("MELF5")

print(np.polyfit(newsorted5.index[1:len(newsorted5)-1], newsorted5["MELF5"][1:len(newsorted5)-1],1))

print("MELF10")

print(np.polyfit(newsorted10.index[1:len(newsorted10)-1], newsorted10["MELF10"][1:len(newsorted10)-1],1))
import scipy.stats
y = pd.Series(data['Ending_college']).values

dist = getattr(scipy.stats, 'gamma')

param = dist.fit(y)

mean, var, skew, kurt = dist.stats(param[0],param[1], param[2], moments='mvsk')

print("Parámetros:")

print(param)

print("Media")

print(mean)

print(data['Ending_college'].mean())

print("Varianza")

print(var)

print(data['Ending_college'].var())

print("Sesgo")

print(skew)

print(data['Ending_college'].skew())

print("Curtosis")

print(kurt)

print(data['Ending_college'].kurtosis())
y = pd.Series(data['5_year_later']).values

dist = getattr(scipy.stats, 'gamma')

param = dist.fit(y)

mean, var, skew, kurt = dist.stats(param[0],param[1], param[2], moments='mvsk')

print("Parámetros:")

print(param)

print("Media")

print(mean)

print(data['5_year_later'].mean())

print("Varianza")

print(var)

print(data['5_year_later'].var())

print("Sesgo")

print(skew)

print(data['5_year_later'].skew())

print("Curtosis")

print(kurt)

print(data['5_year_later'].kurtosis())
y = pd.Series(data['10_years_later']).values

dist = getattr(scipy.stats, 'gamma')

param = dist.fit(y)

mean, var, skew, kurt = dist.stats(param[0],param[1], param[2], moments='mvsk')

print("Parámetros:")

print(param)

print("Media")

print(mean)

print(data['10_years_later'].mean())

print("Varianza")

print(var)

print(data['10_years_later'].var())

print("Sesgo")

print(skew)

print(data['10_years_later'].skew())

print("Curtosis")

print(kurt)

print(data['10_years_later'].kurtosis())
y = pd.Series(data['10_years_later']).values

dist = getattr(scipy.stats, 'genpareto')

param = dist.fit(y)

mean, var, skew, kurt = dist.stats(param[0],param[1], param[2], moments='mvsk')

print("Parámetros:")

print(param)

print("Media")

print(mean)

print(data['10_years_later'].mean())

print("Varianza")

print(var)

print(data['10_years_later'].var())

print("Sesgo")

print(skew)

print(data['10_years_later'].skew())

print("Curtosis")

print(kurt)

print(data['10_years_later'].kurtosis())