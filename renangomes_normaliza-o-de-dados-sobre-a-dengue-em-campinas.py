import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.preprocessing import *
dateParser = lambda date: pd.datetime.strptime(date, '%Y-%m-%d')
dataset = pd.DataFrame(pd.read_csv("../input/dengue-dataset.csv", index_col=[0], header=0, parse_dates=['data'], sep=","))
display(dataset.head(25))
dataset.interpolate(inplace=True)
display(dataset.head(25))
normalizado = dataset.copy()

constanteNormalizacao = dataset['casos-confirmados'].max()
normalizado['casos-confirmados'] = dataset['casos-confirmados'] / constanteNormalizacao
normalizado['chuva'] = dataset['chuva'] / dataset['chuva'].max()
normalizado['temperatura-media'] = dataset['temperatura-media'] / dataset['temperatura-media'].max()

print("Constante de Normalização para Saída (casos confirmados):",constanteNormalizacao)

display(normalizado.head())
plt.figure(figsize=(40,20))
sns.heatmap(normalizado,  cmap=sns.diverging_palette(15, 150, sep=1, as_cmap=True))
plt.show()
plt.close()