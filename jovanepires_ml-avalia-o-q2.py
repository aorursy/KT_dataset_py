# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



from warnings import filterwarnings

filterwarnings('ignore')



sns.set()

sns.set_palette("Set1")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/ex2data1.txt', header=None, names=['nota_1', 'nota_2', 'aprovado'])
# amostra dos dados

data.head()
# verificar nulos

fig, ax = plt.subplots(figsize=(20,5))

sns.heatmap(ax=ax, data=data.isnull(), yticklabels=False, cbar=False, cmap='viridis')

plt.title("Verificar itens nulos", fontsize=16)

plt.show()
# gráfico em duas dimensões, cada uma contendo uma nota, e diferenciando aprovação e reprovação pela cor dos dados no gráfico.

markers = {0: 'X', 1: 'o'}

sns.scatterplot(x='nota_1', y='nota_2', hue='aprovado', style="aprovado", markers=markers, s=100, data=data)

plt.title("Aprovados x Reprovados", fontsize=16)

plt.show()
# dividir conjunto de dados em treino e teste

X = data[['nota_1', 'nota_2']]

y = data['aprovado']



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)



scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# calcule a acurácia (score) obtida para o conjunto de teste.

model = LogisticRegression()

model.fit(X_train, y_train)

train = model.score(X_train, y_train)

test = model.score(X_test, y_test)



print("Score train:\t%0.2f" % train)

print("Score test:\t%0.2f" % test)