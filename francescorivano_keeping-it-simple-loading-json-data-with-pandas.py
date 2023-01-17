import pandas as pd

import json

import numpy as np



with open('../input/ds397__ammcomunale_bilancio_rendiconto_previsioni_triennali_2015-2019.json') as json_file:

    parsed_file = json.load(json_file)

    

# Had it been in the same folder and not on Kaggle, I would have used this instead:

#with open('../input/ds397__ammcomunale_bilancio_rendiconto_previsioni_triennali_2015-2019.json') as json_file:

#    parsed_file = json.load(json_file)
type(parsed_file)
type(parsed_file[5])
df = pd.DataFrame(parsed_file)

df.head()
for val, col in zip(df.iloc[0], df.columns):

    print("Column: {}, value: {}".format(col, val))
df.Cdc.unique()
df['Cdc'] = df['Cdc'].astype('str')

df = df.replace({'None':'0'})

df['Cdc'] = df['Cdc'].astype('int')



numerical_columns = ['ARTICOLO', 'CAPITOLO', 

                     'Centro di responsabilità', 

                     'NUMERO', 'PDC-Livello1', 

                     'PDC-Livello2', 'PDC-Livello3',

                     'PDC-Livello4', 'PDC-Missione',

                     'PDC-Programma']



stuff_with_commas = ['RENDICONTO 2015', 'RENDICONTO 2016',

                     'STANZIAMENTO 2017', 'STANZIAMENTO 2018',

                     'STANZIAMENTO 2019', 'STANZIAMENTO DI CASSA 2017']



for col in stuff_with_commas:

    df[col] = df.replace('.', '').replace(',', '.')



numerical_columns = numerical_columns + stuff_with_commas



for col in numerical_columns:

    df[col] = pd.to_numeric(df[col])
columns_with_text = ['DIR', 'Descrizione Centro di Responsabilità',

                     'Descrizione Direzione', 'Descrizione capitolo PEG',

                     'Descrizione centro di costo', 'TIPO']

for col in columns_with_text:

    print('Column: {}, unique values: {}'.format(col, df[col].unique().shape[0]))
df['TIPO'] = pd.Categorical(df['TIPO'])
df.dtypes
for col in df.columns:

    print('Column: {}, unique values: {}'.format(col, df[col].unique().shape[0]))