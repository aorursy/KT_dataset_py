#Buenas! En este Kernel mostraré lo que hice para cargar los datos de los pokemons, tratar sus valores nulos, encoding, ...

#Espero que os sirva de ayuda/inspiración :D
#Primero cargamos librerías imprescindibles 

import numpy as np # <3

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Leemos el csv

df_pokemon = pd.read_csv('../input/pokemon.csv')



#Parece que la ultima fila mostrada tiene un valor nulo en su columna Type1...

df_pokemon.head()
#Como podemos ver a continuación existen más filas con "NaN"

df_pokemon[df_pokemon.isnull().any(axis=1)].head()
#Lo mejor es ser precavido y, en mi caso, sustituyo los "NaN" por "None" (se puede poner lo que uno quiera)

df_pokemon = df_pokemon.fillna({'Name': 'None', 'Type 1': 'None', 'Type 2': 'None'})

df_pokemon.head()
#cambiando nombre de variable (# queda feo para identificar un id)

df_pokemon = df_pokemon.rename(index=str, columns={"#": "id_pokemon"})

df_pokemon.head()
# la columna LEGENDARY tiene un tipo booleano: decido pasarlo a 0 en caso de False y a 1 en caso de True

df_pokemon['Legendary'] = np.where(df_pokemon['Legendary'] == True, 1, 0)

df_pokemon.head()
# Para facilitar un poco la vida al modelo, decido pasar las columnas con valores string a valores numéricos.

# sklearn tiene una función que permite realizar esta labor: LabelEncoder

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

from sklearn import preprocessing

# obtengo las columnas que me interesa codificar (son las únicas con strings en sus valores)

valores_type1 = df_pokemon['Type 1'].values

valores_type2 = df_pokemon['Type 2'].values

valores_name = df_pokemon['Name'].values



# y aplico el encoder. Tengo uno para cada columna, aunque se podrían juntar todo en uno. A gusto del consumidor.

le1 = preprocessing.LabelEncoder()

le2 = preprocessing.LabelEncoder()

lename = preprocessing.LabelEncoder()



# strings --> a --> números

encoding1 = le1.fit_transform(valores_type1)

encoding2 = le2.fit_transform(valores_type2)

encodingName = lename.fit_transform(valores_name)



# aquí asigno los nuevos valores

df_pokemon['Type 1'] = encoding1

df_pokemon['Type 2'] = encoding2

df_pokemon['Name'] = encodingName



df_pokemon.head()
# Y bueno, esta es la idea a seguir. 

# Puede parecer una tontería, pero al quitar/sustituir valores o codificar labels se puede

# mejorar unos puntos la precisión del modelo, así que creo que merece la pena realizar estas labores previas.

# Un saludo y gracias por leerme!