import pandas as pd 
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
%pylab inline
%matplotlib inline
pd.options.display.float_format = '{:,.2f}'.format
pd.options.display.max_colwidth = 200

conexao = sqlite3.connect('../input/caged/amostra1pc.sqlite')

# Carrega os dados do CNPJ
df = pd.read_sql_query("SELECT * FROM caged", conexao)

df_desc_cnae = pd.read_sql_query("SELECT DISTINCT cod_secao, nm_secao FROM tab_cnae", conexao)
df.head()
df_desc_cnae.head()
# Descrição do Porte da Empresa
df['faixa_empregados_descr']= df['faixa_empregados'].replace(["-1","01","02","03","04","05","06","07","08","09","99","1","2","3","4","5","6","7","8","9"],
                                                                   ['IGNORADO',
                                                                    'ATE 4',
                                                                    'DE 5 A 9',
                                                                    'DE 10 A 19',
                                                                    'DE 20 A 49',
                                                                    'DE 50 A 99',
                                                                    'DE 100 A 249',
                                                                    'DE 250 A 499',
                                                                    'DE 500 A 999',
                                                                    '1000 OU MAIS',
                                                                   '1000 OU MAIS',
                                                                    'ATE 4',
                                                                    'DE 5 A 9',
                                                                    'DE 10 A 19',
                                                                    'DE 20 A 49',
                                                                    'DE 50 A 99',
                                                                    'DE 100 A 249',
                                                                    'DE 250 A 499',
                                                                    'DE 500 A 999',
                                                                    '1000 OU MAIS'])
# Join para incluir descrição do CNAE
df = pd.merge(df, df_desc_cnae,how='left', left_on=['secao'], right_on='cod_secao')
df['data_ref'] = (df['competencia'].str[:4])+(df['competencia'].str[5:].str[:2])
df['ano_ref'] = (df['competencia'].str[:4])
df['data_ref'] = df['data_ref'].astype(int)
df['ano_ref'] = df['ano_ref'].astype(int)
df.head()
df_caged = df.copy()
df_caged.shape
antes = df_caged[df_caged['data_ref']<202002]
depois = df_caged[df_caged['data_ref']>202001]
#Antes da pandemia
fluxo = antes[['fluxo','data_ref']].groupby(['data_ref']).sum()
fluxo = fluxo.reset_index()
fluxo_antes = fluxo[(fluxo['data_ref']>201901) & (fluxo['data_ref']<201907)]
fluxo_antes.head()
#Depois
fluxo = depois[['fluxo','data_ref']].groupby(['data_ref']).sum()
fluxo_depois = fluxo.reset_index()
fluxo_depois.head(10)
ax = fluxo_antes.plot(kind = "bar", x = 'data_ref',title = " Contratações e Demissões antes da Pandemia",figsize = (10,4))
ax.set_xlabel("Periodo")
ax = fluxo_depois.plot(kind = "bar", x = 'data_ref',title = " Contratações e Demissões após inicio da Pandemia",color = 'green',figsize = (10,4))

ax.set_xlabel("Periodo")
#Antes da pandemia
fluxo = antes[['fluxo','data_ref']].groupby(['data_ref']).sum()
fluxo = fluxo.reset_index()
fluxo_antes = fluxo[(fluxo['data_ref']>201401) & (fluxo['data_ref']<201801)]
fluxo_antes.head()
ax = fluxo_antes.plot(kind = "bar", x = 'data_ref',title = " Contratações e Demissões antes da Pandemia",figsize = (10,4))
ax.set_xlabel("Periodo")
#Antes
antes_filtro = antes[antes['data_ref']>201909]
fluxo_antes = antes_filtro[['fluxo','uf']].groupby(['uf']).sum()
fluxo_antes = fluxo_antes.reset_index()
print(fluxo_antes)
#Antes
antes_filtro = antes[(antes['data_ref']>201901) &( antes['data_ref']<201907)]
fluxo_antes = antes_filtro[['fluxo','uf']].groupby(['uf']).sum()
fluxo_antes = fluxo_antes.reset_index()
print(fluxo_antes)
#Depois
fluxo_depois = depois[['fluxo','uf']].groupby(['uf']).sum()
fluxo_depois = fluxo_depois.reset_index()
print(fluxo_depois)
ax = fluxo_antes.plot(kind = "bar", x = 'uf',title = " Contratações e Demissões 4 meses anteriores a Pandemia",figsize = (10,5))
ax.set_xlabel("Estados - UF")
ax = fluxo_depois.plot(kind = "bar", x = 'uf',title = " Contratações e Demissões 4 meses após inicio da Pandemia",color = 'green',figsize = (10,5))
ax.set_xlabel("Estados - UF")
# 2019
valor_antes_cnae = antes_filtro[['nm_secao','fluxo']].groupby(['nm_secao']).sum()
valor_antes_cnae = valor_antes_cnae.reset_index()
valor_antes_cnae.columns = ['Ramo de Atividade','Novos Empregos']
valor_antes_cnae.sort_values(by = 'Novos Empregos', ascending = False ).head(10)
# 2020
valor_depois_cnae = depois[['nm_secao','fluxo']].groupby(['nm_secao']).sum()
valor_depois_cnae = valor_depois_cnae.reset_index()
valor_depois_cnae.columns = ['Ramo de Atividade','Novos Empregos']
valor_depois_cnae.sort_values(by = 'Novos Empregos', ascending = False ).head(50)
valor_depois_mun = depois[['municipio','uf','fluxo']].groupby(['municipio','uf']).sum()
valor_depois_mun  = valor_depois_mun .reset_index()
valor_depois_mun.columns = ['Municipio','UF','Novos Empregos']
valor_depois_mun.sort_values(by = 'Novos Empregos', ascending = False ).head(20)
valor_depois_mun.sort_values(by = 'Novos Empregos', ascending = True ).head(20)
valor_depois_emp = depois[['faixa_empregados_descr','fluxo']].groupby(['faixa_empregados_descr']).sum()
valor_depois_emp  = valor_depois_emp.reset_index()
valor_depois_emp.columns = ['Tipo da Empresa','Novos Empregos']
valor_depois_emp.sort_values(by = 'Novos Empregos', ascending = True).head(10)
valor_depois_uf = depois[['faixa_empregados_descr','uf','fluxo']].groupby(['faixa_empregados_descr','uf']).sum()
valor_depois_uf  = valor_depois_uf.reset_index()
valor_depois_uf.columns = ['Tipo da Empresa','Estado UF','Novos Empregos']
valor_depois_uf.sort_values(by = 'Novos Empregos', ascending = True).head(10)
Crise_2008 = df_caged[(df_caged['data_ref']>200808) &( df_caged['data_ref']<200902)]


Crise_2020 = df_caged[(df_caged['data_ref']>202001) &( df_caged['data_ref']<202007)]
#2008
contrata = Crise_2008[['fluxo','data_ref']].groupby(['data_ref']).sum()
contrata = contrata.reset_index()
contrata.head(20)
sum(contrata.fluxo)
#2020
contrata1 = Crise_2020[['fluxo','data_ref']].groupby(['data_ref']).sum()
contrata1 = contrata1.reset_index()
contrata1.head(20)
sum(contrata1.fluxo)
ax = contrata.plot(kind = "bar", x = 'data_ref',title = " Contratações e Demissões na crise de 2008",figsize = (10,4))
ax.set_xlabel("Periodo")
ax = contrata1.plot(kind = "bar", x = 'data_ref',title = " Contratações e Demissões na crise de 2020",color = 'green',figsize = (10,4))
ax.set_xlabel("Periodo")



#2020
estado = Crise_2008[['fluxo','uf']].groupby(['uf']).sum()
estado = estado.reset_index()
print(estado)




#2020
estado2 = Crise_2020[['fluxo','uf']].groupby(['uf']).sum()
estado2 = estado2.reset_index()
print(estado2)
ax1 = estado.plot(kind = "bar", x = 'uf',title = " Contratações e Demissões no ano de 2008",color = 'red', figsize = (10,5))
ax1.set_xlabel("Estados - UF")
ax1 = estado2.plot(kind = "bar", x = 'uf',title = " Contratações e Demissões no ano 2020",color = 'b',figsize = (10,5))
ax1.set_xlabel("Estados - UF")



#2008
valor_cnae = Crise_2008[['nm_secao','fluxo']].groupby(['nm_secao']).sum()
valor_cnae = valor_cnae.reset_index()
valor_cnae.columns = ['Ramo de Atividade','Novos Empregos']
valor_cnae.sort_values(by = 'Novos Empregos', ascending = False ).head(50)
# 2020
valor_cnae1 = Crise_2020[['nm_secao','fluxo']].groupby(['nm_secao']).sum()
valor_cnae1 = valor_cnae1.reset_index()
valor_cnae1.columns = ['Ramo de Atividade','Novos Empregos']
valor_cnae1.sort_values(by = 'Novos Empregos', ascending = False ).head(50)
municipio = Crise_2008[['municipio','uf','fluxo']].groupby(['municipio','uf']).sum()
municipio  = municipio .reset_index()
municipio.columns = ['Municipio','UF','Novos Empregos']
municipio.sort_values(by = 'Novos Empregos', ascending = True ).head(50)
municipio1 = Crise_2020[['municipio','uf','fluxo']].groupby(['municipio','uf']).sum()
municipio1  = municipio1 .reset_index()
municipio1.columns = ['Municipio','UF','Novos Empregos']
municipio1.sort_values(by = 'Novos Empregos', ascending = True ).head(50)
tamanho = Crise_2008[['faixa_empregados_descr','fluxo']].groupby(['faixa_empregados_descr']).sum()
tamanho  = tamanho.reset_index()
tamanho.columns = ['Tipo da Empresa','Novos Empregos']
tamanho.sort_values(by = 'Novos Empregos', ascending = True).head(10)
tamanho1 = Crise_2020[['faixa_empregados_descr','fluxo']].groupby(['faixa_empregados_descr']).sum()
tamanho1  = tamanho1.reset_index()
tamanho1.columns = ['Tipo da Empresa','Novos Empregos']
tamanho1.sort_values(by = 'Novos Empregos', ascending = True).head(20)
ax1 = tamanho.plot(kind = "bar", x = 'Tipo da Empresa',title = " Contratações e Demissões no ano de 2008",color = 'red', figsize = (10,5))
ax1.set_xlabel("Tipo de empresa")
ax1 = tamanho1.plot(kind = "bar", x = 'Tipo da Empresa',title = " Contratações e Demissões no ano 2020",color = 'b',figsize = (10,5))
ax1.set_xlabel("Tipo de empresa")
