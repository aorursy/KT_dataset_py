# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Vector aleatorio

N = 499 #hay 499 pacientes en el conjunto de test

sol = np.random.rand(N,1)

#tamaño del vector
print("Size of the vector")
print(sol.shape)
print("--------")
#primeras 5 posiciones
print("first 5 solutions2")
print(sol[:5])
sol_1_0 = sol.copy() #ojo, si quiere crear un vector como copia del otro, siempre tengo que utilizar copy, si no, pasan cosas malas !!

sol_1_0[sol<=0.7] = 1
sol_1_0[sol>=.7] = 0

#vamos a ver las 5 primeras filas

print(sol_1_0[:5])

#Vamos a convertir en enteros el vector, para que al pasarlo a csv no haya ningún problema. Kaggle espera valores enteros, no float

sol_1_0 = sol_1_0.astype('int')

print("")
print('Conversión a entero')
print(sol_1_0[:5])
print(np.sum(sol_1_0)/N)
#Creamos un vector con los id

ids = list(range(0,499))

print(ids[:5])

dict_to_df = {'Id':ids,'Prediction':sol_1_0[:,0]}
#creamos el dataframe

sol_df = pd.DataFrame(dict_to_df)

print(sol_df.head())

sol_df.to_csv("my_sol.csv",index=False)
