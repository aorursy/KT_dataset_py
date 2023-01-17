import pandas as pd

import numpy as np

import seaborn as sns



train_types = {'Agencia_ID':np.uint16, 'Ruta_SAK':np.uint16, 'Cliente_ID':np.uint32, 

               'Producto_ID':np.uint16, 'Demanda_uni_equil':np.uint32}



test_types = {'Agencia_ID':np.uint16, 'Ruta_SAK':np.uint16, 'Cliente_ID':np.uint32, 

              'Producto_ID':np.uint16, 'id':np.uint32}



df_train = pd.read_csv('../input/train.csv', usecols=train_types.keys(), dtype=train_types)

df_test = pd.read_csv('../input/test.csv',usecols=test_types.keys(), dtype=test_types)
print('Train Data\n', df_train.head(1), '\n')

print('Test Data\n', df_test.head(1), '\n')