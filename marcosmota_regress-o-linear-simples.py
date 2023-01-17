import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numpy import genfromtxt

%matplotlib inline

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

data = genfromtxt('/kaggle/input/1.01. Simple linear regression.csv', delimiter=',',skip_header=1)



# Realizar a transposta da matrix

XtX = data.transpose()

X_entrada = XtX[0]

y_saida =  XtX[1]



# Mostrar os graficos

plt.plot(X_entrada, y_saida, 'r*', label='Dados')

plt.legend(loc="upper left", bbox_to_anchor=(1,1))


# Matriz de 1s para servir como as previsões

ones = np.ones(len(X_entrada))



# Convertendo as variáveis para matrizes NumPy

X_matrix = np.matrix(np.column_stack((X_entrada, ones)))  # Matriz aumentada (explicada nas aulas)

y_matrix = np.transpose(np.matrix(y_saida))



# Encontrando a transposta das matrizes e aplicando o dot product

XtX = X_matrix.transpose().dot(X_matrix)

Xty = X_matrix.transpose().dot(y_matrix)



# Obtendo a matrix inversa, onde o resultado é os coeficientes da operação

w, b = np.linalg.inv(XtX).dot(Xty).tolist()



# Previsões

y_pred = w[0]*X_entrada + b[0]



# Plot

plt.plot(X_entrada, y_saida, 'r*', label='Dados')

plt.plot(X_entrada, y_pred, 'b', label='Linha de Regressão - Previsões')

plt.legend(loc="upper left", bbox_to_anchor=(1,1))