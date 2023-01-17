import pandas as pd
import numpy as np
train = pd.read_csv('../input/titanic/train.csv')
mdata = pd.DataFrame({'colunas': train.columns,
                    'tipos': train.dtypes,
                    'percentual_faltante': train.isna().sum() / train.shape[0]})
mdata
#Preenchendo Dados númericos com média ou mediana
train['Age'] = train['Age'].fillna(train['Age'].mode())
#Preenchendo dados categoricos com Unknown ou moda
train['Cabin'] = train['Cabin'].fillna('Unknown')
train['Cabin'].value_counts()
train.shape[0]