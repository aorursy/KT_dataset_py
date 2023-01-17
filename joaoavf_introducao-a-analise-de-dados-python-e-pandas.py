x = 10
x
y = 5
x + y 
nome = 'José'
sobrenome = 'da Silva'
nome_completo = nome + ' ' + sobrenome
nome_completo
import pandas as pd
import seaborn as sns
igm = pd.read_csv('../input/igm_modificado.csv')
igm
igm.head()
igm.tail()
igm.sample(5)
igm.sample(5).T
igm[0:5].T
igm[-5:].T
igm[20:30].T
igm['porte']
igm[['municipio', 'indice_governanca']]
type(igm['porte'])
igm['porte'].value_counts()
%matplotlib inline
igm['porte'].value_counts().plot.bar()
ind_des = igm['indice_governanca']
ind_des.count()
ind_des.size
ind_des.isnull()
ind_des.isnull().sum()
ind_des.dropna()
ind_des.isnull().sum()
ind_des.dropna(inplace=True)
ind_des.isnull().sum()
ind_des.min()
ind_des.max()
ind_des.mean()
ind_des.std()
ind_des.describe()
igm.describe()
ind_des.hist()
sns.kdeplot(ind_des)
sns.distplot(ind_des.dropna())
igm[igm['regiao']=='NORDESTE']
igm['regiao']=='NORDESTE'
filtro = igm['regiao']=='NORDESTE'
igm[filtro].T
nordeste = igm[filtro]
nordeste.sample(5).T
nordeste['perc_pop_econ_ativa'].describe()
nordeste['perc_pop_econ_ativa'].hist(bins=50)
filtro_nordeste = ~nordeste['perc_pop_econ_ativa'].isnull()
nordeste[filtro_nordeste].sort_values(by='perc_pop_econ_ativa', ascending=False)[0:5].T
filtro_2 = ~igm['perc_pop_econ_ativa'].isnull()
igm[filtro & filtro_2].sort_values(by='perc_pop_econ_ativa', ascending=False)[0:5].T
igm.shape
nordeste.shape
igm['municipio'].shape
igm[['municipio']].shape
filtro.shape
filtro_nordeste.shape
filtro_2.shape
nordeste[filtro_2]
igm.info()
igm['area'].sample(5)
igm['area'].astype(float)
igm['area'].str.replace(',','').astype(float).sample(10)
igm['area'] = igm['area'].str.replace(',','').astype(float)
igm['populacao'].sample(10)
igm['populacao'] = igm['populacao'].str.replace(',','')
igm['populacao'] = igm['populacao'].astype(float)
igm['populacao'] = igm['populacao'].str.replace('.','')
valor_problema = '41.487(1)'
valor_problema.split('(')
valor_problema.split('(')[0]
igm['populacao'] = igm['populacao'].str.split('(')
igm['populacao'].sample(10)
igm['populacao'] = igm['populacao'].str[0]
igm['populacao'].sample(10)
igm['populacao'] = igm['populacao'].astype(float)
igm['populacao'] / igm['area']
igm['densidade_2'] = igm['populacao'] / igm['area']
igm[['municipio','populacao','area','densidade_dem', 'densidade_2']].sample(10)
igm['densidade_dem'] = igm['densidade_2']
igm.drop(columns='densidade_2', inplace=True)
igm['comissionados_por_servidor'] = igm['comissionados']/igm['servidores']
!pip install plotly
import plotly.offline as plotly
import plotly.graph_objs as go
plotly.init_notebook_mode(connected=True)
pyplot_data = [go.Histogram(x=igm['exp_vida'])]
plotly.iplot(pyplot_data)
pyplot_data_norm = [go.Histogram(x=igm['exp_vida'], histnorm='probability')]
plotly.iplot(pyplot_data_norm)
igm.to_csv('igm_virgula.csv', index=False, sep=';', decimal=',')
pd.read_csv('igm_virgula.csv',  sep=';').info()
pd.read_csv('igm_virgula.csv', sep=';', decimal=',').info()
pd.read_excel('../input/exemplo_1.xls')
pd.read_excel('../input/exemplo_1.xls', sheet_name='Municípios')
pd.read_excel('../input/exemplo_1.xls', sheet_name='Municípios', header=1)
pd.read_excel('../input/exemplo_1.xls', sheet_name='Municípios', header=1, skip_footer=14)
df = pd.read_excel('../input/exemplo_1.xls', sheet_name='Municípios', header=[1], skip_footer=14)
df.info()
df['codigo'] = df['COD. UF'] * 100000 + df['COD. MUNIC']
df.sample(5)
sns.distplot(igm[igm['regiao'] == 'NORDESTE']['indice_governanca'].dropna(), label='NORDESTE')
sns.distplot(igm[igm['regiao'] == 'SUDESTE']['indice_governanca'].dropna(), label='SUDESTE')
import matplotlib.pyplot as plt
sns.distplot(igm[igm['regiao'] == 'NORDESTE']['indice_governanca'].dropna(), label='NORDESTE')
sns.distplot(igm[igm['regiao'] == 'SUDESTE']['indice_governanca'].dropna(), label='SUDESTE')
plt.legend()
sns.distplot(igm[igm['regiao'] == 'NORDESTE']['indice_governanca'].dropna(), label='NORDESTE')
sns.distplot(igm[igm['regiao'] == 'SUDESTE']['indice_governanca'].dropna(), label='SUDESTE')
sns.distplot(igm[igm['regiao'] == 'CENTRO-OESTE']['indice_governanca'].dropna(), label='CENTRO-OESTE')
plt.legend()
sns.distplot(igm[igm['regiao'] == 'NORDESTE']['indice_governanca'].dropna(), label='NORDESTE')
sns.distplot(igm[igm['regiao'] == 'SUDESTE']['indice_governanca'].dropna(), label='SUDESTE')
sns.distplot(igm[igm['regiao'] == 'NORTE']['indice_governanca'].dropna(), label='NORTE')
sns.distplot(igm[igm['regiao'] == 'SUL']['indice_governanca'].dropna(), label='SUL')
sns.distplot(igm[igm['regiao'] == 'CENTRO-OESTE']['indice_governanca'].dropna(), label='CENTRO-OESTE')
plt.legend()
igm.nunique()
sns.boxplot(x="regiao", y="indice_governanca", data=igm)
sns.violinplot(x="regiao", y="indice_governanca", data=igm)
sns.violinplot(x="regiao", y="indice_governanca", data=igm, hue='capital')
sns.violinplot(x="regiao", y="indice_governanca", data=igm, hue='porte')
sns.factorplot(x="regiao", y="indice_governanca", data=igm, col='porte', kind='violin')
sns.countplot(x='regiao', data=igm)
sns.countplot(x='regiao', hue='porte', data=igm)
sns.factorplot(x="regiao", data=igm, col='porte', kind='count')
igm['regiao'].value_counts().plot.pie()
igm['regiao'].value_counts().index
pie_chart = go.Pie(labels=igm['regiao'].value_counts().index, values=igm['regiao'].value_counts())
plotly.iplot([pie_chart])
igm.isnull().sum()
igm['sem_igm'] = igm['indice_governanca'].isnull()
sns.violinplot(x="regiao", y="gasto_pc_educacao", data=igm, hue='sem_igm')
igm['sem_gasto_pc_saude'] = igm['gasto_pc_saude'].isnull()
sns.violinplot(x="regiao", y="nota_mat", data=igm, hue='sem_gasto_pc_saude')
pd.qcut(igm['taxa_empreendedorismo'], 3)
igm['cat_te'] = pd.qcut(igm['taxa_empreendedorismo'], 3)
sns.violinplot(x="regiao", y="exp_vida", data=igm, hue='cat_te')
sns.swarmplot(x="regiao", y="indice_governanca", hue='porte', data=igm)
# sns.set(rc={'figure.figsize':(16.7,8.27)})
sns.barplot(x="regiao", y="indice_governanca", hue='porte', data=igm)
sns.barplot(x="regiao", y="taxa_empreendedorismo",  data=igm)
sns.barplot(x="regiao", y="anos_estudo_empreendedor",  data=igm)
sns.barplot(x="regiao", y="pib_pc",  data=igm)
igm.groupby('regiao')['pib_pc'].mean()
sns.pairplot(x_vars=['gasto_pc_educacao'], y_vars=['nota_mat'], data=igm, hue="regiao", size=5)
sns.pairplot(x_vars=['gasto_pc_educacao'], y_vars=['nota_mat'], data=igm, hue="regiao", kind='reg', size=5)
igm['nome_len'] = igm['municipio'].str.len()
sns.pairplot(x_vars=['nome_len'], y_vars=['idhm'], data=igm, kind='reg', size=5)
temp_df = igm[['nota_mat','exp_vida', 'gasto_pc_saude', 'gasto_pc_educacao', 'regiao']]
sns.pairplot(temp_df, hue='regiao', kind='reg')
sns.pairplot(temp_df.dropna(), hue='regiao', kind='reg')
sns.pairplot(x_vars=['populacao'], y_vars=['indice_governanca'], data=igm, hue="regiao", kind='reg', size=5)
import numpy as np
igm['log_pop'] = np.log(igm['populacao'])
sns.pairplot(x_vars=['log_pop'], y_vars=['indice_governanca'], data=igm, hue="regiao", kind='reg', size=5)
format_dict = {'perc_pop_econ_ativa' :'{:.2%}'}
igm.sample(5).style.format(format_dict)
igm.info()
format_dict = {'perc_pop_econ_ativa' :'{:.0%}', 'taxa_empreendedorismo' :'{:.0%}'}
igm.sample(5).style.format(format_dict)
igf = pd.read_excel('../input/exemplo_2.xls', header=[0], skiprows=8, skip_footer=2)
igf
igf = igf[1:].copy()
igf.rename(columns={'Unnamed: 1':'Ranking Estadual'}, inplace=True)
igf.rename(columns={'Município':'mun'}, inplace=True)
df.rename(columns={'NOME DO MUNICÍPIO':'mun'}, inplace=True)
df.merge(igf)
df.merge(igf).shape
df.shape
igf.shape
df['Município'] = df['mun']
igf['mun'] = igf['mun'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
df['mun'] = df['mun'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
df.merge(igf).shape
igf['mun'] = igf['mun'].str.replace(' ','')
df['mun'] = df['mun'].str.replace(' ','')
df.merge(igf).shape
df_mun = df[~df.mun.isin(igf['mun'])]['mun'].values
igf_mun = igf[~igf.mun.isin(df['mun'])]['mun'].values
from difflib import SequenceMatcher
for municipio_1 in df_mun:
    
    
    print(score, municipios_proximos)
    if score > 0.8:
        idx = igf[igf['mun']==municipios_proximos[1]].index[0]
        igf.at[idx, 'mun'] = municipios_proximos[0]       
df.merge(igf).shape
df[~df.mun.isin(igf.mun)]
igf[~igf.mun.isin(df.mun)]
igf.at[2911,'mun'] = 'EmbudasArtes'
df[~df.mun.isin(igf.mun)]
igf[~igf.mun.isin(df.mun)]
igf.at[4702, 'mun'] = 'SaoValerio'
igf.at[4962, 'mun'] = 'SaoVicentedoSerido'
df[~df.mun.isin(igf.mun)]
igf[~igf.mun.isin(df.mun)]
df.merge(igf).shape
