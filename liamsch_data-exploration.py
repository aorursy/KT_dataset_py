import pandas as pd

import numpy as np

import seaborn as sns



train_types = {'Agencia_ID':np.uint16, 'Ruta_SAK':np.uint16, 'Cliente_ID':np.uint32, 

               'Producto_ID':np.uint16, 'Demanda_uni_equil':np.uint32}



test_types = {'Agencia_ID':np.uint16, 'Ruta_SAK':np.uint16, 'Cliente_ID':np.uint32, 

              'Producto_ID':np.uint16, 'id':np.uint32}



df_train = pd.read_csv('../input/train.csv', usecols=train_types.keys(), dtype=train_types)

df_test = pd.read_csv('../input/test.csv',usecols=test_types.keys(), dtype=test_types)

df_client = pd.read_csv('../input/cliente_tabla.csv')

df_product = pd.read_csv('../input/producto_tabla.csv')

df_town = pd.read_csv('../input/town_state.csv')
print('Train Data\n', df_train.head(1), '\n')

print('Test Data\n', df_test.head(1), '\n')

print('Client Data\n', df_client.head(1), '\n')

print('Product Data\n', df_product.head(1), '\n')

print('Town Data\n', df_town.head(1), '\n')
sns.distplot(np.log1p(df_train['Demanda_uni_equil']), kde=False)
agencies_subset = np.zeros(len(df_train))

for i in range(4):

    this_agency = df_train['Agencia_ID'].unique()[i]

    agencies_subset += df_train['Agencia_ID'] == this_agency
print(agencies_subset)

#sns.distplot(df_train.loc[agencies_subset, 'Demanda_uni_equil'], kde=False)