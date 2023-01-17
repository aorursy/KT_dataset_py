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
df_reclamacoes_2012 = pd.read_csv('../input/reclamacoes-fundamentadas-sindec-2012.csv', error_bad_lines=False)
df_reclamacoes_2013 = pd.read_csv('../input/reclamacoes-fundamentadas-sindec-2013.csv', error_bad_lines=False)
df_reclamacoes_2014 = pd.read_csv('../input/reclamacoes-fundamentadas-sindec-2014.csv', error_bad_lines=False)
df_reclamacoes_2015 = pd.read_csv('../input/reclamacoes-fundamentadas-sindec-2015.csv', error_bad_lines=False)
df_reclamacoes_2016 = pd.read_csv('../input/reclamacoes-fundamentadas-sindec-2016.csv', error_bad_lines=False)
pd_columns = ['length']
pd_index   = ['2012', '2013', "2014", "2015", "2016"]
pd_data    = [len(df_reclamacoes_2012), len(df_reclamacoes_2013), len(df_reclamacoes_2014), len(df_reclamacoes_2015), len(df_reclamacoes_2016)]

pd.DataFrame(pd_data, index = pd_index, columns = pd_columns)
df_reclamacoes_2012.head()
df_reclamacoes_2012.hist(column='CodigoRegiao')
df_reclamacoes_2012.hist(column='CodigoAssunto')
df_reclamacoes_2012.hist(column='CodigoProblema')
df_reclamacoes_2012['Atendida'] = df_reclamacoes_2012['Atendida'].map({'S' : 1, 'N': 0})
df_reclamacoes_2012.hist(column='Atendida')
from collections import Counter
age_counter = Counter(df_reclamacoes_2012['FaixaEtariaConsumidor'])
df = pd.DataFrame.from_dict(age_counter, orient='index')
df.plot(kind='bar')