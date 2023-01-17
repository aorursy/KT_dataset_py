import numpy as np

import pandas as pd
data_path_2016 = '/kaggle/input/inep-microdados-enade-2016/microdados_enade2016/3.DADOS/MICRODADOS_ENADE_2016.txt'



df_2016 = pd.read_csv(data_path_2016, sep=';', encoding='latin1') # decimal=',' -> Existem inconsistências na padronização dos arquivos, em 2016 "." é utilizado como separador decimal, em 2017 e 2018 é utilizada ","
df_2016['ENADE_ANO'] = 2016
df_2016.shape
data_path_2017 = '/kaggle/input/inep-microdados-enade-2017/3.DADOS/MICRODADOS_ENADE_2017.txt'



df_2017 = pd.read_csv(data_path_2017, sep=';', encoding='latin1', decimal=',', low_memory=False)
df_2017['ENADE_ANO'] = 2017
df_2017.shape
data_path_2018 = '/kaggle/input/inep-microdados-enade-2018/2018/3.DADOS/microdados_enade_2018.txt'



df_2018 = pd.read_csv(data_path_2018, sep=';', encoding='latin1', decimal=',')
df_2018['ENADE_ANO'] = 2018
df_2018.shape
[c for c in df_2017.columns if c not in df_2018.columns]
[c for c in df_2018.columns if c not in df_2017.columns]
[c for c in df_2016.columns if c not in df_2018.columns]
df_2016['CO_TURNO_GRADUACAO'] = np.where(df_2016[['IN_MATUT', 'IN_VESPER', 'IN_NOTURNO',]].sum(axis=1) > 1, 3, df_2016[['IN_MATUT', 'IN_VESPER', 'IN_NOTURNO',]].idxmax(axis=1)) # 3 = Integral = mais de um periodo

 

df_2016['CO_TURNO_GRADUACAO'] = df_2016['CO_TURNO_GRADUACAO'].replace({'IN_MATUT':1, 'IN_VESPER': 2, 'IN_NOTURNO': 4}).astype(int)
df_2016.drop(columns=['TP_SEMESTRE','AMOSTRA', 'IN_MATUT', 'IN_VESPER', 'IN_NOTURNO', 'IN_GRAD', 'ID_STATUS'], inplace=True)
df_2016.rename(columns={'ANO_FIM_2G': 'ANO_FIM_EM'}, inplace=True)
[c for c in df_2016.columns if c not in df_2018.columns]
[c for c in df_2018.columns if c not in df_2016.columns]
df = pd.concat([df_2016, df_2017, df_2018]).reset_index(drop=True)
df.shape
df['TP_SEXO'].value_counts()
df = df.query("TP_SEXO != 'N'").copy()
df['CO_MODALIDADE'].value_counts()
df_2016['CO_MODALIDADE'].value_counts()
df_2017['CO_MODALIDADE'].value_counts()
df_2018['CO_MODALIDADE'].value_counts()
df['CO_MODALIDADE'] = df['CO_MODALIDADE'].replace({0: 2})
df.to_csv('dados_enade.csv', index=False)
data_dict_path_2018 = '/kaggle/input/inep-microdados-enade-2018/2018/1.LEIA-ME/Dicionário de Variáveis dos Microdados do Enade_Edição 2018.xlsx'



data_dict_2018 = pd.read_excel(data_dict_path_2018, sep=';', encoding='latin1', decimal=',', header=[1])



data_dict_2018['NOME'] = data_dict_2018['NOME'].fillna(method='ffill')
data_dict_path_2017 = '/kaggle/input/inep-microdados-enade-2017/1.LEIA-ME/Dicionário de variáveis dos Microdados do Enade_Edição 2017.xlsx'



data_dict_2017 = pd.read_excel(data_dict_path_2017, sep=';', encoding='latin1', decimal=',', header=[1])
data_dict_path_2016 = '/kaggle/input/inep-microdados-enade-2016/microdados_enade2016/1.LEIA-ME/Dicionário de variáveis dos Microdados do Enade_Ediç╞o 2016.xlsx'



data_dict_2016 = pd.read_excel(data_dict_path_2016, sep=';', encoding='latin1', decimal=',', header=[1])
def treat_data_dict_to_2018_stds(dd):

    dd.columns = [c.upper() for c in dd.columns]

    dd = pd.DataFrame(dd.iloc[:,-1].str.split('\n', expand=True).values, index=dd['NOME']).stack()

    dd = dd.reset_index().rename(columns = {0:'CATEGORIAS'})[['NOME', 'CATEGORIAS']]

    dd['NOME'] = dd['NOME'].fillna(method='ffill')

    return dd.query("NOME == 'CO_GRUPO'")
data_dict = pd.concat([treat_data_dict_to_2018_stds(data_dict_2016), treat_data_dict_to_2018_stds(data_dict_2017), data_dict_2018])
data_dict['CATEGORIAS'] = data_dict['CATEGORIAS'].str.strip().replace('', np.nan)
data_dict = data_dict[~data_dict['CATEGORIAS'].isnull()].copy()
def normalize(series):

    return series.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
data_dict['NOME'] = normalize(data_dict['NOME'])
data_dict['NOME'].fillna(method='ffill', inplace=True)
data_dict = data_dict[['NOME','CATEGORIAS']]
data_dict.to_csv('dicionario_categorias.csv', index=False)