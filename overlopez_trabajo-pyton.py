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
marcodat=pd.read_excel('/kaggle/input/weightchangeduringquarantine/CambioPesoCuarentena.xls');marcodat.head()


marcodat.info()
total_count = marcodat.groupby('Sexo')['Id'].nunique()
print(total_count)
print(marcodat['Sexo'].describe())

# grafico de barras
plt.ylabel("Frecuencia")
total_count.plot(kind='bar',rot=0);

import matplotlib.pyplot as plt
numero = [total_count.Femenino,total_count.Masculino]
nombres = ["Femenino","Masculino"]
colores = ["#AAF683","#EE6055","#60D394","#FFD97D","#FF9B85"]
desfase = (0.012,0.012)
plt.pie(numero, labels=nombres, autopct="%0.1f %%", colors=colores, explode=desfase)
plt.axis("equal")
plt.show()
total_prog = marcodat.groupby('Programa')['Id'].nunique()
print(total_prog)
print(marcodat['Programa'].describe())

# grafico de barras
plt.ylabel("Frecuencia")
total_prog.plot(kind='bar',rot=0);

#gradico de torta

numero = total_prog
nombres = ["Estadistica","Ing_Ambiental","Ing_sistemas"]
colores = ["#AAF683","#EE6055","#60D394","#FFD97D","#FF9B85"]
desfase = (0.012,0.012,0.012)
plt.pie(numero, labels=nombres, autopct="%0.1f %%", colors=colores, explode=desfase)
plt.axis("equal")
plt.show()
total_grup = marcodat.groupby('Grupo')['Id'].nunique()
print(total_grup)
print(marcodat['Grupo'].describe())

# grafico de barras
nombres =["Grupo_1","Grupo_2","Inferencia_est"]
plt.title("cantidad de personas en cada grupo")
plt.ylabel("Frecuencia")
plt.bar(nombres, total_grup);
#gradico de torta

numero = total_grup
nombres =["Grupo_1","Grupo_2","Inferencia_est"]
colores = ["#AAF683","#EE6055","#60D394","#FFD97D","#FF9B85"]
desfase = (0.012,0.012,0.012)
plt.pie(numero, labels=nombres, autopct="%0.1f %%", colors=colores, explode=desfase)
plt.axis("equal")
plt.show()
total_lugar = marcodat.groupby('Lugar')['Id'].nunique()
print(total_lugar)
print(marcodat['Lugar'].describe())

# grafico de barras
plt.title("cantidad de personas en cada lugar")
plt.ylabel("Frecuencia")
total_lugar.plot(kind='bar',rot=0);

numero = total_lugar
nombres =["Casco urbano","Zona rutral"]
colores = ["#AAF683","#EE6055","#60D394","#FFD97D","#FF9B85"]
desfase = (0.012,0.012)
plt.pie(numero, labels=nombres, autopct="%0.1f %%", colors=colores, explode=desfase)
plt.axis("equal")
plt.show()
total_afisa = marcodat.groupby('ActFisAntes')['Id'].nunique()
print(total_afisa)
print(marcodat['ActFisAntes'].describe())

# grafico de barras
plt.ylabel("Frecuencia")
plt.title("Total  por respuesta")
total_afisa.plot(kind='bar',rot=0);

numero = total_afisa
nombres =["No","Sí"]
colores = ["#AAF683","#EE6055","#60D394","#FFD97D","#FF9B85"]
desfase = (0.012,0.012)
plt.pie(numero, labels=nombres, autopct="%0.1f %%", colors=colores, explode=desfase)
plt.axis("equal")
plt.show()

total_afisd = marcodat.groupby('ActFisDurante')['Id'].nunique()
print(total_afisd)
print(marcodat['ActFisDurante'].describe())

# grafico de barras
plt.title("")
plt.ylabel("Frecuencia")
total_afisd.plot(kind='bar',rot=0);


numero = total_afisd
nombres =["No","Sí"]
colores = ["#AAF683","#EE6055","#60D394","#FFD97D","#FF9B85"]
desfase = (0.012,0.012)
plt.pie(numero, labels=nombres, autopct="%0.1f %%", colors=colores, explode=desfase)
plt.axis("equal")
plt.show()
total_chal = marcodat.groupby('CamHabAliment')['Id'].nunique()
print(total_chal)
print(marcodat['CamHabAliment'].describe())

#grafico circular
numero = total_chal
nombres =["No","Sí"]
colores = ["#AAF683","#EE6055","#60D394","#FFD97D","#FF9B85"]
desfase = (0.012,0.012)
plt.pie(numero, labels=nombres, autopct="%0.1f %%", colors=colores, explode=desfase)
plt.axis("equal")
plt.show()



total_rcha = marcodat.groupby('RazonCambioHabitoaAlim')['Id'].nunique()
print(total_rcha)
print(marcodat['RazonCambioHabitoaAlim'].describe())

#grafico circular
numero = total_rcha
nombres =["Comía mucho menos","Comía mucho más"]
colores = ["#AAF683","#EE6055","#60D394","#FFD97D","#FF9B85"]
desfase = (0.012,0.012)
plt.pie(numero, labels=nombres, autopct="%0.1f %%", colors=colores, explode=desfase)
plt.axis("equal")
plt.show()
total_chs = marcodat.groupby('CabioHabitosSueño')['Id'].nunique()
print(total_chs)
print(marcodat['CabioHabitosSueño'].describe())

#grafico circular
numero = total_chs
nombres =["No","Sí"]
colores = ["#AAF683","#EE6055","#60D394","#FFD97D","#FF9B85"]
desfase = (0.012,0.012)
plt.pie(numero, labels=nombres, autopct="%0.1f %%", colors=colores, explode=desfase)
plt.axis("equal")
plt.show()

total_rchs = marcodat.groupby('RazonCambioHabitoSueño')['Id'].nunique()
print(total_rchs)
print(marcodat['RazonCambioHabitoSueño'].describe())

# grafico de barras
nombres =["Dormía menos","Dormía más"]
plt.title("")
plt.ylabel("Frecuencia")
plt.bar(nombres, total_rchs);
Pesoinicial = marcodat['PesoInicial'].dropna()
Pesoinicial.describe()
#numero de clases
k = round(1+3.332*np.log(Pesoinicial.count()))
print(k)

#Tabla de frecuencia 
marcodat["Peso_inicial"] = pd.cut(marcodat["PesoInicial"], bins=14)

PesoInicial_counts2=(marcodat
  .groupby("Peso_inicial")
  .agg(frequency=("PesoInicial", "count"))) 

PesoInicial_counts2["Frec acumulada"] = PesoInicial_counts2["frequency"].cumsum()
PesoInicial_counts2

#Histograma

#numero de clases por medio de la formula de Sturges
k = round(1+3.332*np.log(Pesoinicial.count()))
print(k)

#Grafico con k=14 clases
n, bins, patches=plt.hist(marcodat.PesoInicial, bins=14)

mids=[(bins[i+1]+bins[i])/2 for i in np.arange(len(bins)-1)]
for i in np.arange(len(mids)):
    plt.text(mids[i]-1,n[i]+0.5,round(n[i]))
plt.ylim([0,10])
plt.show();
# Box plot
plt.boxplot(marcodat.PesoInicial.dropna() ,sym='*')
plt.ylabel("Peso")
plt.ylim([30,100])
plt.show()
Pesofinal = marcodat['PesoFinal'].dropna()
Pesofinal.describe()
#numero de clases
k = round(1+3.332*np.log(Pesofinal.count()))
print(k)

#Tabla de frecuencia 
marcodat["Peso_final"] = pd.cut(marcodat["PesoFinal"], bins=14)

PesoFinal_counts2=(marcodat
  .groupby("Peso_final")
  .agg(frequency=("PesoFinal", "count"))) 

PesoFinal_counts2["Frec acumulada"] = PesoFinal_counts2["frequency"].cumsum()
PesoFinal_counts2
#Histograma

#numero de clases por medio de la formula de Sturges
k = round(1+3.332*np.log(Pesofinal.count()))
print(k)

#Grafico con k=14 clases
n, bins, patches=plt.hist(marcodat.PesoFinal, bins=14)

mids=[(bins[i+1]+bins[i])/2 for i in np.arange(len(bins)-1)]
for i in np.arange(len(mids)):
    plt.text(mids[i]-1,n[i]+0.5,round(n[i]))
plt.ylim([0,13])
plt.show();
# Box plot
plt.boxplot(marcodat.PesoFinal.dropna() ,sym='*')
plt.ylabel("Peso")
plt.ylim([30,110])
plt.show()
plt.boxplot([Pesoinicial,Pesofinal], sym='*')
plt.xlabel("")
plt.ylabel("Peso")
plt.show()
# grafico de densidad para la Peso Inicial
import seaborn as sns
sns.distplot(marcodat['PesoInicial']);

# grafico de densidad para la Peso final
sns.distplot(marcodat['PesoFinal']);

# Estadisticas de resumen para peso inicial discriminando por sexo.

(marcodat
    .groupby(['Sexo'])
    .agg({
       'PesoInicial': ['describe','var']
       
   }))
# Estadisticas de resumen para peso inicial discriminando por sexo.

(marcodat
    .groupby(['Sexo'])
    .agg({
       'PesoFinal': ['describe','var']
       
   }))
## 
(marcodat
    .groupby(['Sexo',"ActFisDurante"])
    .agg({
       'PesoFinal': ['describe','var']
       
   }))
Pesodif = marcodat.PesoFinal-marcodat.PesoInicial
AumentoPeso= Pesodif[Pesodif>0]
DismPeso=Pesodif[Pesodif<0]
Pesoestable = Pesodif[Pesodif==0]
pd.DataFrame({"Aumento":AumentoPeso, "Disminoyo":DismPeso, "igual":Pesoestable}).describe()


#Diagrama de barras
totales = [AumentoPeso.count(),DismPeso.count(),Pesoestable.count()]
nombres =["Aumento","Disminuyo","Igual"]
plt.title("")
plt.xlabel("")
plt.ylabel("Frecuencia")
plt.bar(nombres, totales);
#Grafico de torta
numero = totales
nombres =["Aumento","Bajo","Igual"]
colores = ["#AAF683","#EE6055","#60D394","#FFD97D","#FF9B85"]
desfase = (0.012,0.012,0.012)
plt.pie(numero, labels=nombres, autopct="%0.1f %%", colors=colores, explode=desfase)
plt.axis("equal")
plt.show()

gansex = pd.DataFrame({"Sexo":marcodat.Sexo, "cambio_peso":Pesodif})
gansex.head()
ganhombres = gansex[gansex.Sexo=='Masculino']
ganmujer = gansex[gansex.Sexo=='Femenino'];ganmujer
ganH = ganhombres[ganhombres.cambio_peso>0]
ganM = ganmujer[ganmujer.cambio_peso>0]
x=[1,2]
y=[ganH.mean(),ganM.mean()]
n=[len(ganH.cambio_peso), len(ganM.cambio_peso)]
error=[1.96*ganH.cambio_peso.std(ddof=1),1.96*ganM.cambio_peso.std(ddof=1)]/np.sqrt(n)
plt.errorbar(x,y, yerr=error, fmt='o',capsize=5, capthick=3)
plt.show()