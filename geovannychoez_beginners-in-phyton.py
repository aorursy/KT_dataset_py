import pandas as pd #libreria pandas para utilizar dataframes
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv') # lectura del directorio y archivo en kaggle
data #ver los datos
#import statistics as stats  
sum(data.Deaths)
sum(data.Confirmed)
sum(data.Recovered)
#import numpy as np

import matplotlib.pyplot as plt
x=[sum(data.Confirmed),sum(data.Recovered),sum(data.Deaths)]
fig = plt.figure()

plt.bar(range(3), x, edgecolor='black')

etiquetas = ['Confirmados', 'Recuperados', 'Muertes']

plt.xticks(range(3), etiquetas)

plt.title("Casos de Covid19")

plt.show()
x