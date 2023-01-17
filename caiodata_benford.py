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
dados = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv', encoding='latin1')
dados.head()
dados.number.hist()
dados.number.tail()
def primeiro_digito(num_str):

    return num_str[0]
# %%timeit



lista = []

for i in range(dados.number.shape[0]):

    lista.append(primeiro_digito(str(dados.number.iloc[i])))
# %%timeit



aqui = dados.number.astype(str).apply(primeiro_digito)
frequencia = dados.number.astype(str).apply(primeiro_digito)
# verificar a frequencia dos digitos

frequencia.value_counts().drop('0')
# total de amostras

total_amostras = frequencia.value_counts().drop('0').sum()

total_amostras
# verificar a frequencia dos digitos

frequencia.value_counts().drop('0').plot.bar()
# probabilidade de um digito é formado pela equacao abaixo

digito = 1

np.log10(digito + 1) - np.log10(digito)
# criando a funcao da formula acima

def benford(inicio, fim):

    return pd.Series([ np.log10(digito + 1) - np.log10(digito) \

            for digito in range(inicio, fim+1)])
frequencia = frequencia.value_counts().drop('0') / frequencia.value_counts().drop('0').sum()

frequencia
benford(1,9).plot.bar()
# esperado - observado
esperado = benford(1,9)
(esperado.values - frequencia.values)**2/esperado.values


esperado.iloc[0] - frequencia.iloc[0]