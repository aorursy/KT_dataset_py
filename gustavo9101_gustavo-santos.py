# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling as ppf

import missingno as msno

import sys

import scikitplot as splt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Carregando as bases de treino e teste 



train = pd.read_csv('../input/train.csv', thousands=',')

test = pd.read_csv('../input/test.csv', thousands=',')



train.shape, test.shape
# Unindo as tabelas para tratar uma única vez

df = train.append(test, sort=True)

df.shape
df.sample(5).T
# Valores missing na tabela de teino

df.isnull().sum().to_frame('Qtd. Missing')
#verificando visualmente a distribuição dos valores missing

msno.matrix(df,figsize=(12,5))
#Valores missing no data frame DF

df.isnull().sum().to_frame('Qtd. Missing')
df.info()
# Verificando os tipos das colunas

print('========Contagem==========')

print(df.dtypes.value_counts())

print('==========================')

print('=======Percentual=========')

print(df.dtypes.value_counts(normalize=True).apply("{:.2%}".format))

print('==========================')
# Obtendo as colunas

df.columns
df['populacao'] = df['populacao'].str.replace(',','').str.replace('.','').apply(lambda x: x.split('(')[0]).astype(int)

df['comissionados_por_servidor'] = df['comissionados_por_servidor'].str.replace('#DIV/0!','').str.rstrip('%')

df['comissionados_por_servidor'] = df['comissionados_por_servidor'].convert_objects(convert_numeric=True)/100

df['exp_vida']=df['exp_vida'].round(2)

df['codigo_mun'] = df['codigo_mun'].astype(object)

df['ct_porte'] = df['porte'].astype('category').cat.codes

df['ct_regiao'] = df['regiao'].astype('category').cat.codes

df['ct_estado'] = df['estado'].astype('category').cat.codes
# Verificando os tipos das colunas

print('========Contagem==========')

print(df.dtypes.value_counts())

print('==========================')

print('=======Percentual=========')

print(df.dtypes.value_counts(normalize=True).apply("{:.2%}".format))

print('==========================')
f, axes=plt.subplots(1,2, figsize=(15,6))

plt.suptitle('Caracteristicas das colunas', ha='center', fontsize=14)

P=df.dtypes.value_counts().plot.pie(autopct='%1.2f%%',ax=axes[0], label='',title='Tipos Colunas - Distr Percentual', legend=True)

bplot = df.dtypes.value_counts().plot(kind='bar',ax=axes[1],rot=0)

for b in bplot.patches:

    bplot.annotate(format(b.get_height(),'.0f'), \

                   (b.get_x() + b.get_width() / 2., \

                   b.get_height()), \

                   ha = 'center',\

                   va = 'center',\

                   xytext = (0, 7),\

                   textcoords = 'offset points')    

plt.title('Tipos Colunas - Contagem')

plt.xlabel('')

plt.yticks([])

plt.ylabel('Frequência',labelpad=3)



sns.despine(left=True)
#verificando visualmente a distribuição dos valores missing

msno.matrix(df,figsize=(12,5))
#substituindo os valores nulos pela média (não incluindo nota_mat)

preencher=['comissionados_por_servidor', 'densidade_dem', 'exp_anos_estudo',

           'exp_vida', 'gasto_pc_educacao', 'gasto_pc_saude', 'hab_p_medico',

           'participacao_transf_receita', 'perc_pop_econ_ativa','servidores']



for c in preencher:

    df[c] = df.groupby(['estado', 'porte'])[c].transform(lambda x:x.fillna(x.mean()))
df.shape
# Verificando se as transformações deram certo

df.isnull().sum().to_frame('Qtd. Missing')
# Obtendo as colunas (exceto nota_mat) que apresentaram valores nulos

fltnota = ~df['nota_mat'].isnull()

temp=df[fltnota]

colunas=temp.columns[temp.isnull().any()].tolist()

colunas
temp.shape
# Verificando quem são os valores nulos

temp[temp.isnull().T.any().T]
# preenchendo dos valores pela média do estado

for d in colunas:

    df[d] = df.groupby(['estado'])[d].transform(lambda x:x.fillna(x.mean()))
df.shape
# Verificando se as transformações deram certo

df.isnull().sum().to_frame('Qtd. Missing').T
#verificando visualmente a distribuição dos valores missing após

msno.matrix(df,figsize=(12,5))
# Frequencia de Municípios por Porte

plt.figure(figsize=(7,6))

splot=sns.countplot(df['porte'], order=df['porte'].value_counts().index)



for p in splot.patches:

    splot.annotate(format(p.get_height(),'.0f'), \

                   (p.get_x() + p.get_width() / 2., \

                   p.get_height()), \

                   ha = 'center',\

                   va = 'center',\

                   xytext = (0, 7),\

                   textcoords = 'offset points')    

plt.title('Municípios por Porte')

plt.xlabel('Porte dos Municípios', labelpad=10)

plt.ylabel('Contagem',labelpad=10)

plt.yticks([])

sns.despine(left=True)
# Frequencia de Municípios por UF

plt.figure(figsize=(17,6))

s_uf=sns.countplot(df['estado'],order=df['estado'].value_counts().index)



for p in s_uf.patches:

    s_uf.annotate(format(p.get_height(),'.0f'), \

                   (p.get_x() + p.get_width() / 2., \

                   p.get_height()), \

                   ha = 'center',\

                   va = 'center',\

                   xytext = (0, 7),\

                   textcoords = 'offset points')    

plt.title('Municípios por Estado')

plt.xlabel('', labelpad=10)

plt.ylabel('Frequência',labelpad=10)

plt.yticks([])

sns.despine(left=True)
sns.set(rc={'figure.figsize':(16.7,8.27)})

sns.swarmplot(x='regiao', y='gasto_pc_educacao', data=df,hue='porte')

plt.title('Gastos com Educação')

plt.xlabel('',labelpad=10)

plt.ylabel('GASTOS PER CAPITA',labelpad=10)
#municípios com gastos acima de R$ 1.500,00

temp=df['gasto_pc_educacao']>=1500.0

maisq=df[temp]



#comparativo

temp.shape, maisq.shape, df.shape
f, axes=plt.subplots(1,2, figsize=(20,6))

# Frequencia de Municípios com gastos per capita na educação superiores a R$ 1.500,00, por Região

s_uf=sns.countplot(maisq['regiao'],order=maisq['regiao'].value_counts().index, ax=axes[0])

for p in s_uf.patches:

    s_uf.annotate(format(p.get_height(),'.0f'), \

                   (p.get_x() + p.get_width() / 2., \

                   p.get_height()), \

                   ha = 'center',\

                   va = 'center',\

                   xytext = (0, 7),\

                   textcoords = 'offset points')    

axes[0].set_title('Municípios por região')

axes[0].set_xlabel('', labelpad=10)

axes[0].set_ylabel('Frequência',labelpad=10)

axes[0].set_yticks([])

sns.despine()



flt = df['regiao']=='SUDESTE'

reg=df[flt]

#sns.set(rc={'figure.figsize':(16,6)})

sns.scatterplot(x="gasto_pc_educacao", 

                y="gasto_pc_saude", 

                hue="municipio", 

                size="municipio",

                data=reg.sort_values(by='gasto_pc_educacao', ascending=False)[0:10],

                sizes=(70,250),

                palette="dark",

                ax=axes[1])



axes[1].set_title('[10+] gastos per capita com educação x gastos per capita com saúde')

axes[1].set_xlabel('Gasto per capita educação',labelpad=10)

axes[1].set_ylabel('Gasto per capita saude',labelpad=10)
#Checando os data frames

flt.shape, reg.shape, df.shape
list(reg['porte'].value_counts().index),\

list(reg['porte'].value_counts())
# Frequencia de Municípios na região Sudeste

plt.figure(figsize=(9,6))

s_uf=sns.countplot(reg['porte'], palette= "Blues_d", order=reg['porte'].value_counts().index)



for p in s_uf.patches:

    s_uf.annotate(format(p.get_height(),'.0f'), \

                   (p.get_x() + p.get_width() / 2., \

                   p.get_height()), \

                   ha = 'center',\

                   va = 'center',\

                   xytext = (0, 7),\

                   textcoords = 'offset points')    

plt.title('Municípios por Estado')

plt.xlabel('', labelpad=10)

plt.ylabel('Frequência',labelpad=10)

plt.yticks([])

sns.despine()
# Importando as bibliotecsa necessárias

from sklearn.model_selection import train_test_split as tts

from sklearn.ensemble import RandomForestClassifier as rfc

import scikitplot as skplt



# Importando a biblioteca de métricas

from sklearn.metrics import accuracy_score as acs
# Guardando a base df

df_copy = df.copy()

# Separando a base de teste e treino



df = df_copy[~df_copy['nota_mat'].isnull()]

test = df_copy[df_copy['nota_mat'].isnull()]



df_copy.shape, df.shape,test.shape
# Criando a base de validação

train, valid = tts(df, random_state=13)
train.shape, valid.shape,df.shape, df_copy.shape, test.shape
train.dtypes
df.select_dtypes(include='object')[0:5]
# Remover a coluna nota_mat que não será usada para o treino

ntmat = ['nota_mat', 'codigo_mun','estado','municipio','porte','regiao']



# obtendo as colunas fets

feats = [c for c in train.columns if c not in ntmat]

feats
# Instanciando o modelo

rf= rfc(random_state=13, n_jobs=-1, n_estimators=200, min_samples_split=8)
# Treinando o modelo 

rf.fit(train[feats], train['nota_mat'])
# Fazendo as previsões

preds = rf.predict(valid[feats])
# verificando a acurácia do modelo

acs(valid['nota_mat'],preds)
# Avaliando as caracterísitcas mais importantes

pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh(figsize=(6,6))
train.shape
rf.fit(train[feats], train['nota_mat'])

test['nota_mat']=rf.predict(test[feats])

test.shape
skplt.estimators.plot_feature_importances(rf,

                                          feature_names=['anos_estudo_empreendedor', 'area', 'capital', 'comissionados', 'comissionados_por_servidor',

                                                         'densidade_dem', 'exp_anos_estudo', 'exp_vida', 'gasto_pc_educacao', 'gasto_pc_saude', 'hab_p_medico',

                                                         'jornada_trabalho', 'participacao_transf_receita', 'perc_pop_econ_ativa', 'pib', 'pib_pc', 'populacao',

                                                         'servidores', 'taxa_empreendedorismo', 'ct_porte', 'ct_regiao', 'ct_estado'],

                                          max_num_features=5,

                                          x_tick_rotation=45,

                                          title='Caracterísitcas importantes [5+]')

plt.show()
# Obtendo os valores mínimos da coluna exp_vida

df_copy['exp_vida'].min(), df_copy['exp_vida'].max()
# Criando uma categoria para exp_vida



# Copiando a base

df=df_copy.copy()

df['ds_exp_vida']=pd.qcut(df['exp_vida'],3, labels=["65.00 a 71.70", "71.80 a 74.53", "74.54+"])

df['ct_exp_vida']=pd.qcut(df['exp_vida'],3).cat.codes



print(df['ct_exp_vida'].sort_values().unique()),

print(df['ds_exp_vida'].sort_values().unique())

print(df['exp_vida'].sort_values().unique())
#Avaliando perc_pop_econ_ativa



print(' Avaliando perc_pop_econ_ativa ')

print('-------------------------------')

print(' Valor Mínimo ')

print(df['perc_pop_econ_ativa'].min()),

print('-------------------------------')

print(' Valor Máximo ')

print(df['perc_pop_econ_ativa'].max())

print('-------------------------------')

print(' Contagem de Valores Únicos ')

print(len(df['perc_pop_econ_ativa'].unique().tolist()))

print('-------------------------------')

print(' Testando Categorias Valores Únicos ')

pd.qcut(df['exp_vida'],3)
#Testando categorias para perc_pop_econ_ativa



print('-------------------------------')

print(' Testando Categorias para 3, 5 e 7 valores categoricos ')



print('-------------------------------')

print(pd.qcut(df['perc_pop_econ_ativa'],3).unique())

print('-------------------------------')

print(pd.qcut(df['perc_pop_econ_ativa'],5).sort_values().unique())

print('-------------------------------')

print(pd.qcut(df['perc_pop_econ_ativa'],7).sort_values().unique())

print('-------------------------------')
# Criando uma categoria para perc_pop_econ_ativa

df['ds_perc_pop_econ_ativa']=pd.qcut(df['perc_pop_econ_ativa'],5, 

                          labels=['0.289 - 0.779', '0.780-0.889', '0.890-0.989', '0.990-1.069', '1.070-1.640'])



df['ct_perc_pop_econ_ativa']=pd.qcut(df['perc_pop_econ_ativa'],5).cat.codes
#Avaliando jornada_trabalho



print(' Avaliando perc_pop_econ_ativa ')

print('-------------------------------')

print(' Valor Mínimo ')

print(df['jornada_trabalho'].min()),

print('-------------------------------')

print(' Valor Máximo ')

print(df['jornada_trabalho'].max())

print('-------------------------------')

print(' Lista de Valores Únicos ')

print(len(df['jornada_trabalho'].unique().tolist()))

print('-------------------------------')

print(df['jornada_trabalho'].sort_values().unique())
#Testando categorias para jornada_trabalho



print('-------------------------------')

print(' Testando Categorias para 3, 5 e 7 valores categoricos ')

print('                               ')

print('-------------------------------')

print(pd.qcut(df['jornada_trabalho'],3).unique())

print('                               ')

print('-------------------------------')

print(pd.qcut(df['jornada_trabalho'],5).sort_values().unique())

print('                               ')

print('-------------------------------')

print(pd.qcut(df['jornada_trabalho'],7).sort_values().unique())

print('                               ')

print('-------------------------------')
# Criando as categorias jornais de trabalho

df['ds_jornada_trabalho']=pd.qcut(df['perc_pop_econ_ativa'],7, labels=['0.000-35.982', '35.983-38.704','38.705-40.821',

                                                                       '40.822-42.471', '42.472-44.160','44.161-46.481', 

                                                                       '46.482-68.528'])

df['ct_jornada_trabalho']=pd.qcut(df['perc_pop_econ_ativa'],7)
# Avaliando expectativa de vida

sns.set(rc={'figure.figsize':(14.7,6.27)})

sns.swarmplot(x='regiao', 

              y='exp_vida', 

              data=df,

              hue='ds_exp_vida')



plt.title('Expectativa de Vida por Faixa de Idade')

plt.xlabel('',labelpad=10)

plt.ylabel('Expectativa de Vida',labelpad=10)
df['ds_exp_vida'].value_counts().plot.pie(autopct='%1.2f%%',label='',title='Distr Percentual Expectativa de Vida por faixa ')



sns.despine(left=True)
# Avaliando características perc_pop_econ_ativa

sns.set(rc={'figure.figsize':(14.7,6.27)})

sns.swarmplot(x='regiao', 

              y='perc_pop_econ_ativa', 

              data=df,hue='ds_perc_pop_econ_ativa')



plt.title('')

plt.xlabel('',labelpad=10)
test[['codigo_mun','nota_mat']].to_csv('modelo.csv', index=False)