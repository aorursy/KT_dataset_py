import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_excel('ads.xlsx')
print(df)
df.info()
df.dtypes
df['TV'] = df['TV'].values.astype(np.float)
df['Sales'] = df['Sales'].values.astype(np.float)
df.head(5)
df.describe()
df.var()
def tipoDeCorrelacao(correlacao):
    if correlacao == 0: 
        print('Nao existe Correlacao')
    elif correlacao < 0: 
        print('Correlacao Linear Negativa Perfeita')
    else: 
        print('Correlacao Linear Positiva Perfeita')
    if correlacao < 0: 
        correlacao = - 1 * correlacao
    if 0.0 >= correlacao <= 0.19: 
        return 'Uma correlação bem fraca'
    elif correlacao <= 0.39:
        return 'Uma correlação fraca'
    elif correlacao <= 0.69:
        return 'Uma correlação moderada'
    elif correlacao <= 0.89: 
        return 'Uma correlação forte'
    else:
        return 'Uma correlação muito forte'
correlacao = df['TV'].corr(df['Sales']) 
correlacao
print(f'{correlacao:.9f}')
print(tipoDeCorrelacao(correlacao))
import scipy
import seaborn as sns
df.shape()
ax = sns.lmplot(x = 'TV', y = 'Sales', data = df)
ax.fig.set_size_inches(12,6)
ax.fig.suptitle('TV x Sales', fontsize = 16, y = 1.02)
ax.set_xlabels('Anuncio de TV', fontsize = 14)
ax.set_ylabels('Sales', fontsize = 14)

import sklearn
from sklearn import linear_model
model = sklearn.linear_model.LinearRegression()
X = np.c_[df['TV']]
y = np.c_[df['Sales']]
model.fit(X,y)
valor = float(input("Informe o valor a investir US$"))
X_new = [[valor]]
vendasEstimadas = float(model.predict(X_new))
print(f'O valor de retorno em vendas investido será de US${vendasEstimadas:.2f}')
from statistics import stdev, mean
def modeloVendas(vp): 
    inclinacao = correlacao * (stdev(df['Sales']) / stdev(df['TV']))
    interceptacao = mean(df['Sales']) - (inclinacao * mean(df['TV']))
    #print(inclinacao)
    #print(interceptacao)
    return interceptacao + (inclinacao * vp)
prever = float(input('Informe o valor a prever para retornar em vendas em US$'))
print(f'O valor de retorno em vendas investido será de US${modeloVendas(prever):.2f}')