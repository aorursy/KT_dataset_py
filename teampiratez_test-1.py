# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc
print ('Reading train!')

train = pd.read_csv('../input/train.csv',

                    usecols=['Agencia_ID',

                                  'Ruta_SAK',

                                  'Cliente_ID',

                                  'Producto_ID',

                                  'Demanda_uni_equil'],

                    dtype={'Agencia_ID': 'uint16',

                                      'Ruta_SAK': 'uint16',

                                      'Cliente_ID': 'int32',

                                      'Producto_ID': 'uint16',

                                      'Demanda_uni_equil': 'float32'})

print ('Train read!')

print ('Estimating means features')

train['Demanda_uni_equil'] = train['Demanda_uni_equil'].apply(lambda x: 1.005*np.log1p(x + 0.01) - 0.005)

train['MeanP'] = train.groupby('Producto_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')

MP = train.loc[:, ['Producto_ID','MeanP']].drop_duplicates(subset=['Producto_ID'])

test = pd.read_csv('../input/test.csv',

                   usecols=['Agencia_ID',

                              'Ruta_SAK',

                              'Cliente_ID',

                              'Producto_ID',

                            'id'],

                   dtype={'Agencia_ID': 'uint16',

                                      'Ruta_SAK': 'uint16',

                                      'Cliente_ID': 'int32',

                                      'Producto_ID': 'uint16',

                                      'id': 'int32'})

print ('Test read!')
test = test.merge(MP,

                  how='left',

                  on=['Producto_ID'],

                  copy=False)

print ('P merged!')

test.loc[:, ['id', 'Producto_ID']].to_csv('submission.csv', index=False)