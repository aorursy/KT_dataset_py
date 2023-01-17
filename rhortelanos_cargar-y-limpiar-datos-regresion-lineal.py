import pandas as pd

import numpy as np
INPUT_DIR = '../input/'

df_test = pd.read_csv(INPUT_DIR + 'test.csv')

df_pokemon = pd.read_csv(INPUT_DIR + 'pokemon.csv')

df_battles = pd.read_csv(INPUT_DIR + 'battles.csv')
df_pokemon.head()
df_pokemon.dtypes
df_pokemon['Type 1'].head()
df_pokemon.describe(include='all')
df_pokemon[df_pokemon.Name.isna()]
df_pokemon.iloc[61].Name
df_pokemon.iloc[62, 1] = 'Primeape'
df_pokemon.describe(include='all')
df_pokemon['Type 2'] = df_pokemon['Type 2'].fillna('None')
df_pokemon.describe(include='all')
df_battles.describe(include='all')
useful_columns = ['Type 1', 'Type 2', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary']

df_pokemon = df_pokemon.set_index('#')

first_df = df_pokemon.loc[df_battles['First_pokemon'].values, useful_columns].add_prefix('poke1_')

second_df = df_pokemon.loc[df_battles['Second_pokemon'].values, useful_columns].add_prefix('poke2_')

df_train = first_df.reset_index().join(second_df.reset_index(), lsuffix='a' ).drop(columns=['#a', '#'])

df_train['target'] = df_battles['Winner']
df_train.head()
df_train.dtypes
for col in df_train.select_dtypes('object').columns:

    df_train[col] = df_train[col].astype('category')
df_train.dtypes
df_train.head()
for col in df_train.select_dtypes('category').columns:

    df_train[col] = df_train[col].cat.codes
df_train.dtypes
df_train.head()
from sklearn.linear_model import LinearRegression

from sklearn.metrics import classification_report
# aqui creamos nuestro modelo

lr = LinearRegression(normalize=True)
# aqui dejamos listos nuestros datos de entrenamiento, sinedo X_train los datos e y_train la variable objetivo

X_train = df_train[df_train.columns[:-1]]

y_train = df_train.target
# y aqui entrenamos

lr.fit(X_train, y_train )
# ale ya está, ya hemos definido nuestro modelo y entrenado en dos lineas. Increible verdad? Podemos ver que predice de la siguiente manera

lr.predict(X_train)
# Ahora vamos a sacar algunas metricas (sobre los mismos datos de entrenamiento, lo que nunca debemos hacer porque no sabremos si hemos hecho overfitting)

# Como la regresion lineal devuelve valores continuos y nosotros necesitamos valores 0 o 1. Vamos a redondear los valores usando numpy

print(classification_report(y_train,np.round(lr.predict(X_train))))
first_df_test = df_pokemon.loc[df_test['First_pokemon'].values, useful_columns].add_prefix('poke1_')

second_df_test = df_pokemon.loc[df_test['Second_pokemon'].values, useful_columns].add_prefix('poke2_')

df_test_ready = first_df_test.reset_index().join(second_df_test.reset_index(), lsuffix='a' ).drop(columns=['#a', '#'])
for col in df_test_ready.select_dtypes('object').columns:

    df_test_ready[col] = df_test_ready[col].astype('category').cat.codes
predictions = np.round(lr.predict(df_test_ready)).clip(0,1).astype(int)
predictions.min(), predictions.max()
sampleSubmission = pd.read_csv(INPUT_DIR + 'sampleSubmission.csv')
sampleSubmission['Winner'] = predictions
sampleSubmission.to_csv('my_sub.csv', index=False)
sampleSubmission.head()