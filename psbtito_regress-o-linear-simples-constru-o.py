import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Carregamento dos dados 

data = pd.read_csv('../input/plano_saude.csv')



# Cálculo de correlação das variáveis.

data.corr()
# Separação das variáveis recursos e alvo.

X = data.iloc[:,0].values

Y = data.iloc[:,1].values



X = X.reshape(-1,1)
from sklearn.linear_model import LinearRegression



model = LinearRegression()

model.fit(X,Y)



# Visualização gráfica do modelo.

plt.scatter(X, Y)

plt.plot(X, model.predict(X), color = 'red')

plt.title ("Regressão simples")

plt.xlabel("Idade")

plt.ylabel("Custo")