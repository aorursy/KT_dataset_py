# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import numpy as np
import pandas as pd

daten=pd.read_excel('/kaggle/input/weightchangeduringquarantine/CambioPesoCuarentena.xls')


#Descripción de variables cuantitativas

daten[['PesoInicial','PesoFinal']].describe()


#Histograma para Peso Inicial

import numpy as np
import matplotlib.pyplot as plt

PesoInicial=daten.PesoInicial
PesoFinal=daten.PesoFinal

b=np.arange(35,100,10)

n, bins, patches1=plt.hist(PesoInicial, bins=b)

mids=[(bins[i+1]+bins[i])/2 for i in np.arange(len(bins)-1)]

for i in np.arange(len(mids)):
    plt.text(mids[i]-1,n[i]+0.5,round(n[i]))

plt.ylim([0,25])
plt.ylabel('Frecuencia')
plt.xlabel('Peso Inicial')
plt.title('Histograma Peso Incial')

plt.show()  



#Histograma para Peso Inicial

import numpy as np
import matplotlib.pyplot as plt


PesoFinal=daten.PesoFinal

b=np.arange(35,110,10)

n, bins, patches1=plt.hist(PesoFinal, bins=b)

mids=[(bins[i+1]+bins[i])/2 for i in np.arange(len(bins)-1)]

for i in np.arange(len(mids)):
    plt.text(mids[i]-1,n[i]+0.5,round(n[i]))

plt.ylim([0,25])
plt.ylabel('Frecuencia')
plt.xlabel('Peso Final')
plt.title('Histograma Peso Final')

plt.show()  

plt.boxplot([daten.PesoInicial.dropna(),daten.PesoFinal.dropna()], sym='*')
plt.show()
daten["PesoInicial_group"] = pd.cut(daten["PesoInicial"], bins=7)

Estatura_counts=(daten
  .groupby("PesoInicial_group")
  .agg(frequency=("PesoInicial", "count")))

Estatura_counts["cum_frequency"] = Estatura_counts["frequency"].cumsum()
Estatura_counts
daten["PesoFinal_group"] = pd.cut(daten["PesoFinal"], bins=7)

Estatura_counts=(daten
  .groupby("PesoFinal_group")
  .agg(frequency=("PesoFinal", "count")))

Estatura_counts["cum_frequency"] = Estatura_counts["frequency"].cumsum()
Estatura_counts
SyP=pd.crosstab(index=daten['Sexo'],
            columns=daten['Programa'], margins=True)
SyP
SyP.plot(kind='bar',alpha=0.8,rot=0)
plt.legend(loc=0)
plt.xlabel('Sexo')
plt.ylabel('Frecuencia')
plt.title('Diagrama de barras - Sexo y Programa')
plt.show

PyL=pd.crosstab(index=daten['Lugar'],
            columns=daten['Programa'], margins=True)
PyL
PyL.plot(kind='bar',alpha=0.8,rot=0)
plt.legend(loc=0)
plt.xlabel('Lugar')
plt.ylabel('Frecuencia')
plt.title('Diagrama de barras - Programa y Sexo')
plt.show
AyD=pd.crosstab(index=daten['ActFisAntes'],
            columns=daten['ActFisDurante'], margins=True)
AyD
T_CamHabAliment = daten.groupby('CamHabAliment')['Id'].nunique()
print(T_CamHabAliment)



n = T_CamHabAliment
Nomb =["No","Sí"]
plt.pie(n, labels=Nomb, autopct="%0.1f %%")
plt.axis("equal")
plt.show()
T_RazonCambioHabitoAlim = daten.groupby('RazonCambioHabitoaAlim')['Id'].nunique()
print(T_RazonCambioHabitoAlim)

n = T_RazonCambioHabitoAlim
Nomb =["Comía mucho menos durante la cuarentena que antes de ella","Comía mucho más durante la cuarentena que antes de ella"]
plt.pie(n, labels=Nomb, autopct="%0.1f %%")
plt.axis("equal")
plt.show()
total_afisd = daten.groupby('CabioHabitosSueño')['Id'].nunique()
print(total_afisd)

# grafico de barras
plt.title("")
plt.ylabel("Frecuencia")
total_afisd.plot(kind='bar',rot=0);
T_RazonCambioHabitoSueño = daten.groupby('RazonCambioHabitoSueño')['Id'].nunique()
print(T_RazonCambioHabitoSueño)

n = T_RazonCambioHabitoSueño
Nomb =["Durante la cuarentena dormía menos horas que antes de ella","Durante la cuarentena dormía muchas más horas que antes de ella"]
plt.pie(n, labels=Nomb, autopct="%0.1f %%")
plt.axis("equal")
plt.show()