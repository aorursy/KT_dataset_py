import numpy as np     # álgebra linear

import pandas as pd    # entrada de dados

import seaborn as sns  # plotagem de gráficos

import pylab as pl     # para mudar tamanho de figuras

import sys

import math

import csv



import os
entries = os.listdir("../input/jhucsseafter0201")



data = pd.read_csv("../input/jhucsseafter0201/" + entries[0])



soma = data.shape[0]



"""

for i in range(1, len(entries)):

    if entries[i][0:2] != "01" and entries[i][0:5] != "02-01": # Não seleciona os anteriores a 02/02

        dataaux = pd.read_csv("../input/jhucss-covid-19-data/" + entries[i])

        soma += dataaux.shape[0]

        data = pd.concat([data, dataaux], axis=0, sort=False)

"""



for i in range(1, len(entries)):

    dataaux = pd.read_csv("../input/jhucsseafter0201/" + entries[i])

    soma += dataaux.shape[0]

    data = pd.concat([data, dataaux], axis=0, sort=False)







data = data.sort_values(by=['Province/State', 'Last Update'], ascending=False)







# Modificando coluna Last Update para separar



new = data["Last Update"].str.split("T", n = 1, expand = True)



data["Day"] = new[0]

data["Hours"] = new[1]

data.drop(columns =["Last Update"], inplace = True)

    

print("tamanho:", data.shape, "soma:", soma)



data.head(30)
aux = set()



for index, row in data.iterrows():

    if type(row['Province/State']) is str:

        aux.add((row['Country/Region'], row['Province/State']))

        

aux2 = list(aux)



"""

with open('teste.csv','w') as out:

    csv_out=csv.writer(out)

    for row in aux2:

        csv_out.writerow(row)

"""