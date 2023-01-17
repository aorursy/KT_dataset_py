# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

  #  for filename in filenames:

    #    print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib_venn import venn2
df_alunos = pd.read_csv('../input/bd_alunos.csv', sep=';', encoding = 'ISO-8859-1', index_col= 0)

df_alunos.head()
df_dengue = pd.read_csv('../input/bd_dengue.csv', sep=';', encoding = 'ISO-8859-1', index_col= 0)

df_dengue.head()
df_onibus = pd.read_csv('../input/bd_onibus.csv', sep=';', encoding = 'ISO-8859-1', index_col= 0)

df_onibus.head()
df_alunos.isnull().any()
df_dengue.isnull().any()
df_onibus.isnull().any()
df_educacao = pd.merge(df_alunos, df_dengue, how='outer')

df_educacao.head()
namesColsEducacao = ['nome','nome_da_mae','nome_do_pai', 'sexo', 'data_nascimento', 'data_dengue']

df_educacao.columns = namesColsEducacao

df_educacao.head()
df_rel_educacao = df_educacao[df_educacao['data_dengue'].isnull() ]  

df_rel_educacao
df_rel_educacao = df_educacao[df_educacao['data_dengue'].isnull() ]  

df_rel_educacao
len(df_rel_educacao)
df_rel_saude = pd.merge(df_dengue, df_onibus,  how='outer', sort=False)

df_rel_saude.head()
df_rel_saude.rename(columns = {'Nome':'nome', 'Nome da Mae':'nome_mae', 'Nome do Pai':'nome_pai', 'Sexo':'sexo','Data de Nascimento':'data_nasc', 'Data da Dengue':'data_dengue', 'Ãnibus':'onibus'}, inplace = True) 
df_rel_saude.head()
mask_onibus = df_rel_saude['onibus'].isnull()

mask_dengue = df_rel_saude['data_dengue'].notnull()
df_rel_saude = df_rel_saude[mask_onibus & mask_dengue]

df_rel_saude
mask_onibus2 = df_rel_saude['onibus'].notnull()

mask_dengue2 = df_rel_saude['data_dengue'].isnull()
frames = [df_alunos, df_dengue]

df_rel_educacao_saude = pd.merge(df_alunos, df_dengue, on= 'Nome' , how='outer')

df_rel_educacao_saude.head()
df_rel_educacao_saude.drop(columns=['Nome da Mae_y', 'Nome do Pai_y', 'Sexo_y', 'Data de Nascimento_y'], inplace=True, axis=1)

df_rel_educacao_saude.head()
df_rel_educacao_saude.rename(columns = {'Nome':'nome', 'Nome da Mae_x':'nome_mae', 'Nome do Pai_x':'nome_pai', 'Sexo_x':'sexo','Data de Nascimento_x':'data_nasc', 'Data da Dengue':'data_dengue'}, inplace = True) 

df_rel_educacao_saude.head()
df_rel_educacao_saude = df_rel_educacao_saude[ df_rel_educacao_saude['data_dengue'].notnull() ]

df_rel_educacao_saude
df_rel_educacao_mobilidade = pd.merge(df_alunos, df_onibus, on= 'Nome' , how='outer')

df_rel_educacao_mobilidade.head()
df_rel_educacao_mobilidade.drop(columns=['Nome da Mae_y', 'Nome do Pai_y', 'Sexo_y', 'Data de Nascimento_y'], inplace=True, axis=1)

df_rel_educacao_mobilidade.head()
df_rel_educacao_mobilidade.rename(columns = {'Nome':'nome', 'Nome da Mae_x':'nome_mae', 'Nome do Pai_x':'nome_pai', 'Sexo_x':'sexo','Data de Nascimento_x':'data_nasc', 'Ãnibus':'onibus'}, inplace = True) 

df_rel_educacao_mobilidade.head()
df_rel_educacao_mobilidade.isnull().any()
df_rel_educacao_mobilidade = df_rel_educacao_mobilidade[df_rel_educacao_mobilidade['onibus'].notnull()]

df_rel_educacao_mobilidade