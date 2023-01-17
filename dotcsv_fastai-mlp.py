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
from sklearn import preprocessing



battles = pd.read_csv('../input/battles.csv')

pokemon = pd.read_csv('../input/pokemon.csv')

test    = pd.read_csv('../input/test.csv')



# Descartamos el nombre de los pokemons. No los conozco. No me interesan.

pokemon = pokemon.drop(columns='Name', errors='ignore')



# Transformamos ambos tipos de pokemon a una representación dummie.

pokemon.loc[:, 'Type 1'] = pokemon.loc[:, 'Type 1'].fillna('None')

pokemon.loc[:, 'Type 2'] = pokemon.loc[:, 'Type 2'].fillna('None')



# Mapeamos columna Legendary a int [0,1]

pokemon.Legendary  = pokemon.Legendary.astype(int)



# Juntamos las dos tablas de pokemons.

data = pd.merge(battles, pokemon, left_on='First_pokemon' ,right_on='#')

data = pd.merge(data,    pokemon, left_on='Second_pokemon',right_on='#', suffixes=('_A','_B'))

data = data.sort_values(['battle_number'])

data = data.drop(columns=['#_A', '#_B'])

data = data.iloc[:, 3:]



data.head()
# Juntamos las dos tablas de pokemons.

tout = pd.merge(test, pokemon, left_on='First_pokemon' ,right_on='#')

tout = pd.merge(tout, pokemon, left_on='Second_pokemon',right_on='#', suffixes=('_A','_B'))

tout = tout.sort_values(['battle_number'])

tout = tout.drop(columns=['#_A', '#_B'])

tout = tout.iloc[:, 3:]



tout.head()
from fastai.tabular import *

from sklearn.model_selection import train_test_split



# OJO: CAMBIA ESTE VALOR A 0.0 PARA TU SUBMISSION FINAL, y 0.3 PARA VALIDAR.

TEST_SIZE = 0.01 # Valor desechable para contar con metrics del validation.



# Selección aleatoria de los indices de entrenamiento y prueba.

tr_idx, ts_idx = train_test_split(range(len(battles)), test_size=TEST_SIZE)



# Indicamos a la librería el listado de variables categóricas.

cat_names = ['Generation_A', 'Generation_B', 'Type 1_A', 'Type 1_B', 'Type 2_A', 'Type 2_B', 'Legendary_A', 'Legendary_B']



# Indicamos a la librería que categorice las variables categóricas

# y normalice las variables continuas.

procs = [Categorify, Normalize]



# Generamos el objeto que nos creará y suministrará los minibatches. Es importante fijar de antemano cuál será el conjunto de datos que

# querremos predecir una vez el modelo esté entrenado : test_df=tout. 

db = TabularDataBunch.from_df(path='.', df=data, dep_var='Winner', valid_idx=ts_idx, procs=procs, test_df=tout, cat_names=cat_names, bs=1024)



# Generamos un modelo sencillo con cuatro capas ocultas.

learn = tabular_learner(db, layers=[64, 32, 16, 8], metrics=accuracy)



# E inspeccionamos el valor del learning-rate óptimo.

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(50, 1e-1)
learn.recorder.plot_losses()

plt.title('Training Loss vs Steps')
# learn.lr_find()

# learn.recorder.plot()

# learn.fit_one_cycle(25, 1e-2)
# Obtenemos las predicciones para la tabla tout. 

preds, _ = learn.get_preds(ds_type=DatasetType.Test)

preds = np.argmax(preds, axis=1).numpy()



# Generamos el dataframe de subida.

submission = pd.DataFrame(test.iloc[:, 0])

submission['Winner'] = preds
# Comprobamos que la tabla tenga la forma que buscamos.

print(submission.head())



# y lo guardamos...

submission.to_csv('./submission.csv', index=False)
from IPython.display import HTML

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe

create_download_link(submission)