 # El objetivo de este Kernel es enseñar cómo he vectorizado el tratamiento de datos (juntar los dos csv's principales). Esta 

# técnica permite cargar y preparar los datos de una manera muy rápida y eficiente. Por supuesto, es mejorable y se admiten

# comentarios al respecto :D



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Lo primero cargar los datasets (no los he tratado/limpiado/pulido)

df_pokemon = pd.read_csv('../input/pokemon.csv')

df_battles = pd.read_csv('../input/battles.csv')

df_test = pd.read_csv('../input/test.csv')



df_pokemon.shape, df_battles.shape, df_test.shape
# empezamos a vectorizar en el dominio de numpy

# dataframe a numpy array

pokemon_values = df_pokemon.values 

battles_values = df_battles.values 



df_pokemon.columns, df_battles.columns
# ids de los pokemons (son únicos)

ids_pokemon = pokemon_values[:,0]



# obtenemos valores únicos y los índices inversos para luego reconstruir el array original

# y hacemos los cálculos separando los pokemons que se pelean en cada batalla

ids_pok1, inv1 = np.unique(battles_values[:, 1], return_inverse=True)

ids_pok2, inv2 = np.unique(battles_values[:, 2], return_inverse=True)

resultados_batallas = battles_values[:, 3]



# buscamos donde estan las características de cada pokemon en las batallas

indices1 = np.intersect1d(ids_pok1, ids_pokemon, return_indices=True)

indices2 = np.intersect1d(ids_pok2, ids_pokemon, return_indices=True)



print(ids_pokemon.shape)



# asignamos las caracteristicas (todas las columnas menos el id)

vals_pok1 = pokemon_values[indices1[2], 1:]

vals_pok2 = pokemon_values[indices2[2], 1:]



# ahora tienen tamaño reducido (784), pero buscamos hacer broadcast para las 50000 filas

vals_pok1.shape, vals_pok2.shape



# parece que hay algunos pokemons que no tienen registros de peleas
# reconstruímos el array original

# (50000, 11) cada uno

pok1 = vals_pok1[inv1]

pok2 = vals_pok2[inv2]



pok1.shape, pok2.shape
# aquí juntamos las caracteristicas de los dos pokemons que se están enfrentando

juntar_carac = np.concatenate((pok1, pok2), axis=1)



juntar_carac.shape



# Y hasta aquí la vectorización, el resto es concatenar las columnas que se deseen, ya existentes o derivadas...