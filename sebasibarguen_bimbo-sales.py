import numpy as np

import pandas as pd



from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer



from sklearn.cross_validation import train_test_split
# Read train dataset

# 'Canal_ID',

df_train = pd.read_csv('../input/train.csv', usecols = ['Demanda_uni_equil', 'Semana', 'Producto_ID', 'Agencia_ID',  'Cliente_ID' ],

                       dtype  = {'Semana': 'int32',

                                 'Producto_ID':'int32',

                                 'Agencia_ID': 'int32',

                                 'Cliente_ID':'int32',

                                 'Demanda_uni_equil':'int32'})



print(df_train.head())