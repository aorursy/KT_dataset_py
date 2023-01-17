# Biblotecas de manipulação de dados

import numpy as np

import pandas as pd
path = '/kaggle/input/enade-microdados-2016-2017-2018/enade_combinado/'
df = pd.read_csv(path + 'dados_enade.csv', low_memory=False)
df.shape
data_dict = pd.read_csv(path + 'dicionario_categorias.csv')
data_dict.shape
df['DS_VT_ESC_OFG'].str.len().value_counts()
df['DS_VT_ESC_OCE'].str.len().value_counts()
pd.DataFrame([df['DS_VT_ESC_OFG'].str[i].value_counts() for i in range(8)])
pd.DataFrame([df['DS_VT_ESC_OCE'].str[i].value_counts() for i in range(27)])
df['PROVA_OBJ_RESPOSTAS_IGUAIS'] = False



for resp in ['A', 'B', 'C', 'D', 'E', '*', '.']:

    df['PROVA_OBJ_RESPOSTAS_IGUAIS'] = df['PROVA_OBJ_RESPOSTAS_IGUAIS'] | ((df['DS_VT_ESC_OFG'] == (resp * 8)) & (df['DS_VT_ESC_OCE'] == (resp * 27)))
df['PROVA_OBJ_RESPOSTAS_IGUAIS'].value_counts()
df.groupby(['PROVA_OBJ_RESPOSTAS_IGUAIS'])['QE_I08'].value_counts(normalize=True).unstack()
questionario_colunas = [c for c in df.columns if 'QE_I' in c]

socioeconomico_colunas = [c for c in questionario_colunas if int(c[-2:]) < 27]
df[socioeconomico_colunas].apply(pd.Series.value_counts).T
socioeconomico_colunas = [c for c in socioeconomico_colunas if c not in ['QE_I16']]
resp_dict = {}

df['QE_RESPOSTAS_IGUAIS'] = False

for resp in ['A', 'B']:

    

    df['QE_RESPOSTAS_IGUAIS'] = df['QE_RESPOSTAS_IGUAIS'] | ((df[socioeconomico_colunas] == resp).mean(axis=1) == 1)



    resp_dict[resp]= ((df[socioeconomico_colunas] == resp).mean(axis=1) == 1).sum()
resp_dict
df.groupby(['QE_RESPOSTAS_IGUAIS', 'QE_I08'])['NT_GER'].mean().unstack()