import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

from sklearn.ensemble import RandomForestClassifier



import os

INPUT_DIR = '../input/'

print(os.listdir(INPUT_DIR))
# CARGAR LOS DATASETES COPIANDO LA CARGA DE OTRO KERNEL

df_test = pd.read_csv(INPUT_DIR + 'test.csv')

df_pokemon = pd.read_csv(INPUT_DIR + 'pokemon.csv')

df_battles = pd.read_csv(INPUT_DIR + 'battles.csv')



pokemon_values = df_pokemon.values 

battles_values = df_battles.values 



ids_pokemon = pokemon_values[:,0]



ids_pok1, inv1 = np.unique(battles_values[:, 1], return_inverse=True)

ids_pok2, inv2 = np.unique(battles_values[:, 2], return_inverse=True)

resultados_batallas = battles_values[:, 3]



indices1 = np.intersect1d(ids_pok1, ids_pokemon, return_indices=True)

indices2 = np.intersect1d(ids_pok2, ids_pokemon, return_indices=True)



vals_pok1 = pokemon_values[indices1[2], 0:]

vals_pok2 = pokemon_values[indices2[2], 0:]



pok1 = vals_pok1[inv1]

pok2 = vals_pok2[inv2]



# Aquí juntamos las caracteristicas de los dos pokemons que se están enfrentando y le añadimos la columna de resultados

data = np.concatenate((pok1, pok2), axis=1)

data = np.insert(data, 24, values=resultados_batallas, axis=1)

data.shape
# UN POCO DE LIMPIEZA

# CAMBIAR EL TIPO DE POKEMON POR UN INT Y QUITAR LAS COLUMNAS DE LEGENDARIO Y GENERACIÓN

df = pd.DataFrame(data)

poke_type = df[3].unique()

poke_type = dict(zip(poke_type, range(len(poke_type))))

df[2] = [poke_type[item] for item in df[2]]

df[3] = [poke_type[item] for item in df[3]]

df[14] = [poke_type[item] for item in df[14]]

df[15] = [poke_type[item] for item in df[15]]



# REMOVE NAMES AND 

df = df.drop([1, 11, 13, 23], axis=1)

df



# FINAL HEADER CONFIGURATION:

# 0: ID    2: Type 1    3: Type 2    4: HP    5: Attack    6: Defens     7: Sp. Atk    8: Sp. Def    9: Speed   10:Generation

# 12: ID  14: Type 1   15: Type 2   16: HP   17: Attack   18: Defens    19: Sp. Atk   20: Sp. Def   21: Speed   22:Generation

# 24: WIN
# BUSCAR LOS INPUTS MÁS RELEVANTES

df = df.astype('int')

corr = df.corr()

corr = pd.DataFrame(corr[24])

corr
# SELECCIONO LAS COLUMNAS MÁS MOLONAS

features = [4, 5, 7, 8, 9, 16, 17, 19, 20, 21]#, 25]



X=df[features]

Y=np.ravel(df[24])

# forest_clf = RandomForestClassifier(n_jobs=-1, n_estimators=2000, class_weight="balanced", criterion='gini', bootstrap=True, min_samples_leaf=15, min_samples_split=20)

forest_clf = RandomForestClassifier(n_jobs=-1, n_estimators=2000, class_weight="balanced", criterion='gini', bootstrap=True)

forest_clf.fit(X, Y)



cvs = cross_val_score(forest_clf, X, Y, cv=3, scoring="accuracy")

np.mean(cvs)
# PUES YA LO TENEMOS, CARGAMOS EL DATASET GÜENO

battles_values = df_test.values

ids_pokemon = pokemon_values[:,0]



ids_pok1, inv1 = np.unique(battles_values[:, 1], return_inverse=True)

ids_pok2, inv2 = np.unique(battles_values[:, 2], return_inverse=True)



indices1 = np.intersect1d(ids_pok1, ids_pokemon, return_indices=True)

indices2 = np.intersect1d(ids_pok2, ids_pokemon, return_indices=True)



vals_pok1 = pokemon_values[indices1[2], 0:]

vals_pok2 = pokemon_values[indices2[2], 0:]



pok1 = vals_pok1[inv1]

pok2 = vals_pok2[inv2]

data = np.concatenate((pok1, pok2), axis=1)



# CLEAN DATA

df = pd.DataFrame(data)

poke_type = df[3].unique()

poke_type = dict(zip(poke_type, range(len(poke_type))))

df[2] = [poke_type[item] for item in df[2]]

df[3] = [poke_type[item] for item in df[3]]

df[14] = [poke_type[item] for item in df[14]]

df[15] = [poke_type[item] for item in df[15]]



# REMOVE NAMES AND 

df = df.drop([1, 11, 13, 23], axis=1)

df



# FINAL HEADER CONFIGURATION:

# 0: ID    2: Type 1    3: Type 2    4: HP    5: Attack    6: Defens     7: Sp. Atk    8: Sp. Def    9: Speed   10:Generation

# 12: ID  14: Type 1   15: Type 2   16: HP   17: Attack   18: Defens    19: Sp. Atk   20: Sp. Def   21: Speed   22:Generation

# 24: WIN
# Y GUARDAMOS EL RESULTADO (GRACIAS RIC ;)

predictions = forest_clf.predict(df[features])

sampleSubmission = pd.read_csv(INPUT_DIR + 'sampleSubmission.csv')

sampleSubmission['Winner'] = predictions

sampleSubmission.to_csv('my_sub.csv', index=False)

sampleSubmission.head()