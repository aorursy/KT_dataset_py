# Librerías a utilizar

import numpy as np                # algebra lineal

import pandas as pd               # data frames

import seaborn as sns             # gráficos

import matplotlib.pyplot as plt   # gráficos

import scipy.stats                # estadísticas

from sklearn import preprocessing



import os

print(os.listdir("../input"))
dfOde = pd.read_csv("../input/dataodebrecht3/Odebrecht3.csv")



df = dfOde.copy()

df.drop_duplicates(subset=['Pais'],inplace=True)

print("___ LISTADO DE PAISES DISP. SOBORNADOS POR ODEBRECHT ___")

print (df.iloc[:10, :1])
dfOdeDepu = dfOde.loc[:, ["Pais", "Presupuesto Inicial USD", "Presupuesto Adicional USD"]]



#dfOdeDepu.groupby(level=0).mean()

print("_ INFORMACIÓN COLUMNAS _")

print(dfOde.info())

print("_ SUMA CONTRATACIONES TOTALES POR PAIS _")

dfOdeDepu['Presupuesto'] = dfOdeDepu.apply(lambda x: x['Presupuesto Inicial USD'] + x['Presupuesto Adicional USD'], axis=1)

dfOdeDepu = dfOdeDepu.loc[:, ("Pais", "Presupuesto")]



dfOdeDepu = dfOdeDepu.groupby('Pais', as_index=False).agg({"Presupuesto": "sum"})

print(dfOdeDepu)
# Graficar Ranking

dfGraph = dfOdeDepu.sort_values(['Presupuesto'],ascending=False).reset_index(drop=True)

plt.figure(figsize=(20,20))

sns.barplot(x=dfGraph["Presupuesto"],y=dfGraph["Pais"])

plt.xlabel("Tamaño Contratación",fontsize=25)

plt.ylabel("Países",fontsize=25)

plt.title("Ranking Odebrecht",fontsize=25)

plt.show()
dfSoborGan = pd.read_csv("../input/sobornosganancias/Sobornos-ganancias.csv")



print("_ SOBORNOS vs. GANANCIAS _")

print(dfSoborGan)
# Integrar la cuantía de los contratos con los sobornos y las ganancias

mergedOdePSG = pd.merge(dfOdeDepu, dfSoborGan, on=["Pais"],how='outer')

#mergedOdePSG = pd.merge(dfOdeDepu,

                 #dfSoborGan[['Soborno', 'Ganancias']],

                 #on='Pais')

print("_ INTEGRACIÓN DATOS VR. CONTRATOS, SOBORNOS, GANANCIAS _")

print(mergedOdePSG)
sns.lmplot(x="Soborno",y="Presupuesto",data=mergedOdePSG)
sns.lmplot(x="Soborno",y="Ganancias",data=mergedOdePSG)