import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

        

df = pd.read_csv('http://informacao.mec.gov.br/bilibs/PDA/PRONATEC/CONCLUINTES_RF_EPCT_2016_CSV.csv', sep = ';', encoding='cp1252')

df.head()

df['SIGLA_UF_UNIDADE_ENSINO'].value_counts().plot(kind='bar').legend()
