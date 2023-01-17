# Biblotecas de manipulação de dados

import numpy as np

import pandas as pd



# Bibliotecas de visualização

import seaborn as sns

import matplotlib.pyplot as plt



# Testes estatísticos

from scipy.stats import ttest_ind
path = '/kaggle/input/enade-microdados-2016-2017-2018/enade_combinado/'
df = pd.read_csv(path + 'dados_enade.csv', low_memory=False)
df.shape
data_dict = pd.read_csv(path + 'dicionario_categorias.csv')
data_dict.shape
mapa_pg = pd.read_excel(path + 'dicionario_completo_2018.xlsx', 

                        header=[1])[['NOME', 'DESCRIÇÃO']].dropna().set_index('NOME').to_dict()['DESCRIÇÃO']
def mapear_coluna(df, data_dict, coluna):

    tmp = data_dict.query("NOME == @coluna")['CATEGORIAS'].str.split('=', expand=True).dropna()

    if tmp.shape[0] <= 1: #work-around for NU_IDADE

        return {i:i for i in sorted(df[coluna].unique())}

        

    tmp.iloc[:, 0] = tmp.iloc[:, 0].str.strip().astype(df[coluna].dtype)

    tmp.iloc[:, 1] = tmp.iloc[:, 1].str.strip().str.upper()

    return tmp.set_index(0).iloc[:,0].to_dict()
mapear_coluna(df, data_dict, 'QE_I08')
def groupby_1_var(df, groupby_var, key_var, agg_method=['mean', 'std', 'count']):

    tmp = df.groupby(groupby_var)[key_var].agg(agg_method).sort_index()

    tmp.index = pd.Series(tmp.index).replace(mapear_coluna(df, data_dict, groupby_var))

    tmp.index.name = mapa_pg[tmp.index.name]

    return tmp
def groupby_2_var(df, var1, var2, key_var, agg_method=['mean'], index_line=False, percentualize=False):

    tmp = df.groupby([var1, var2])[key_var].agg(agg_method).unstack()

    if percentualize:

        tmp = tmp.div(tmp.sum(axis=1), axis=0)

    if index_line:

        tmp = tmp.div(tmp.iloc[0])

        

    tmp.index = pd.Series(tmp.index).replace(mapear_coluna(df, data_dict, var1))

    tmp.index.name = mapa_pg[tmp.index.name]

    

    tmp.columns = tmp.columns.droplevel(0)

    tmp.columns = pd.Series(tmp.columns).replace(mapear_coluna(df, data_dict, var2))

    tmp.columns.name = mapa_pg[tmp.columns.name]

    

    return tmp
def stat_sig(df, var, key_var):

    lista_de_listas = []

    valores = sorted(df[var].dropna().unique().tolist())

    for i in valores:

        lista_de_listas.append([])

        for j in valores:

            t_res = ttest_ind(df.query(f'{var} == @i')[key_var], df.query(f'{var} == @j')[key_var])

            lista_de_listas[-1].append(t_res.pvalue)

            

    

    valores = pd.Series(valores).replace(mapear_coluna(df, data_dict, var))

    tmp = pd.DataFrame(lista_de_listas, index=valores, columns=valores)    

   

    return tmp
df['NT_GER'].isnull().mean()
df['NT_GER'].fillna(0, inplace=True)
groupby_1_var(df=df, groupby_var='QE_I08', key_var = 'NT_GER')
groupby_1_var(df=df, groupby_var='QE_I23', key_var = 'NT_GER')
s = stat_sig(df, 'QE_I23', 'NT_GER')

sns.heatmap(s)
groupby_2_var(df, var1='QE_I08', var2='QE_I23', key_var = 'NT_GER')
sns.heatmap(data=groupby_2_var(df, var1='QE_I08', var2='QE_I23', key_var = 'NT_GER'), 

            cmap=sns.diverging_palette(10, 133, as_cmap=True),

            )
tipos_de_nota = ['NT_GER', 'NT_OBJ_FG', 'NT_DIS_FG', 'NT_OBJ_CE', 'NT_DIS_CE']
notas_por_renda = groupby_1_var(df=df, groupby_var='QE_I08', key_var = tipos_de_nota, agg_method=['mean'])

notas_por_renda
notas_por_renda.div(notas_por_renda.iloc[0])
notas_por_estudo = groupby_1_var(df=df, groupby_var='QE_I23', key_var = tipos_de_nota, agg_method=['mean'])

notas_por_estudo
notas_por_estudo.div(notas_por_estudo.iloc[0])
groupby_2_var(df=df, var1='QE_I08', var2='QE_I23', key_var = 'NT_OBJ_FG')
sns.heatmap(data=groupby_2_var(df, var1='QE_I08', var2='QE_I23', key_var = 'NT_OBJ_FG'),

            cmap=sns.diverging_palette(10, 133, as_cmap=True),

            )
groupby_2_var(df=df, var1='QE_I08', var2='QE_I23', key_var = 'NT_DIS_CE')
sns.heatmap(data=groupby_2_var(df, var1='QE_I08', var2='QE_I23', key_var = 'NT_DIS_CE'),

            cmap=sns.diverging_palette(10, 133, as_cmap=True),

            )
gen_vs_nota = groupby_1_var(df=df, groupby_var='TP_SEXO', key_var = 'NT_GER')

gen_vs_nota
gen_vs_nota.div(gen_vs_nota.iloc[1])
df['IS_PROVA_NULA'] = (df.NT_GER == 0) | df.NT_GER.isnull()
groupby_1_var(df=df, groupby_var='TP_SEXO', key_var = 'IS_PROVA_NULA')
sexo_vs_renda = groupby_2_var(df=df, var1='TP_SEXO', var2='QE_I08', key_var = 'NT_GER', agg_method=['count'])

sexo_vs_renda
sexo_vs_renda.div(sexo_vs_renda.iloc[1])
notas_por_sexo = groupby_1_var(df=df, groupby_var='TP_SEXO', key_var = tipos_de_nota, agg_method=['mean'])

notas_por_sexo
notas_por_sexo.div(notas_por_sexo.iloc[0])
groupby_2_var(df=df, var1='TP_SEXO', var2='QE_I08', key_var = 'NT_OBJ_FG')
groupby_2_var(df=df, var1='TP_SEXO', var2='QE_I08', key_var = 'NT_DIS_CE')