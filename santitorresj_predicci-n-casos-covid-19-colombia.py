from matplotlib.font_manager import FontProperties

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

import pandas as pd

import pylab as pl

import warnings

import datetime as dt

from fbprophet import Prophet

from fbprophet.plot import add_changepoints_to_plot

warnings.filterwarnings("ignore")



import seaborn as sns







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/cases1/Casos1.csv")

#print(df.head(20))
dist=df["Departamento o Distrito"]

cont = []

for i in range(0,len(dist)):

    cont.append(i+1)



df["Contador"]=cont

#print(df.head(len(cont)))
columns=df[["Fecha de diagnóstico", "Contador"]]

columns=columns.rename(columns={"Fecha de diagnóstico":"dd", "Contador":"y"})

dat=pd.to_datetime(columns["dd"],  dayfirst=True)

columns["ds"]=dat

columns=columns[["ds", "y"]]

df["Atención**"].value_counts()
plt.figure(figsize=(10, 8))

plt.title("Casos Coronavirus en Colombia por género", {'fontsize': 20})

sns.countplot(x="Sexo", hue="Atención**", data=df,palette=["g", "c", "y", "m", "r", "b"])

facet = sns.FacetGrid(df, hue="Sexo",aspect=4)

facet.map(sns.kdeplot,'Edad',shade= True)

facet.set(xlim=(0, df['Edad'].max()))

facet.add_legend()

 

plt.show()
n = Prophet()

n.fit(columns)

fut = n.make_future_dataframe(periods=20)

fut.tail()

forecast = n.predict(fut)

#print(forecast)

fig1 = n.plot(forecast, xlabel='Fecha hasta el 1 de Mayo',ylabel='Casos confirmados en Colombia (predicción)')

plt.title("Tendencia de casos en Colombia (predicción - línea azul)")