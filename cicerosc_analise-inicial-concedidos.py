import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno





import os
# Carga os dados

df = pd.read_csv('../input/concedidos-2018-12/concedidos201812.csv', encoding='latin', sep=';', header=0)
### Correção dos dados gerais: espaço, troca de ponto por virgula e conversão de textos para NULL

df['Competência concessão'] = df['Competência concessão'].astype(str)

df['Qt SM RMI']             = df['Qt SM RMI'].str.replace(',','.')

df['Qt SM RMI']             = df['Qt SM RMI'].apply(lambda x: float(x))

df['UF']                    = df['UF'].apply(lambda x: x.strip())

df['Espécie']               = df['Espécie'].apply(lambda x: x.strip())

df['CID']                   = df['CID'].apply(lambda x: x.strip())

df['CID']                   = df['CID'].apply(lambda x: x.split()[0] if x!='Em Branco' and x!='{ñ class}' and x!='Zerados' else None)

df['Forma Filiação']        = df['Forma Filiação'].apply(lambda x: x.strip())

df['Vínculo dependentes']   = df['Vínculo dependentes'].apply(lambda x: x.strip())

df['Vínculo dependentes']   = df['Vínculo dependentes'].apply(lambda x: x if x!='Não Informado'and x!= '{ñ class}' else None)

df['Clientela']             = df['Clientela'].apply(lambda x: x.strip())
### Correção da coluna municipios

df['Mun Resid'] = df['Mun Resid'].apply(lambda x: x.strip())

df['Mun Resid'] = df['Mun Resid'].apply(lambda x: '00000-XX' if x == '{ñ class}' or x == '00000-Zerada' else x)

df['Mun Resid'] = df['Mun Resid'].apply(lambda x: x[0:8])



### Mesmo com as correções acima, o Registro 173494 encontra-se com um traço a menos. Dessa forma, segue correção:

#df['Mun Resid'].iloc[173494] = "16224-PI"

df.at[173494,'Mun Resid'] = "16224-PI"



### Criando as colunas codigo municipio e sigla municipio

df['CodMuni']    = df['Mun Resid'].apply(lambda x: x.split('-')[0])

df['SiglaUF']    = df['Mun Resid'].apply(lambda x: x.split('-')[1])



### Convertendo os valores não encontrados para Null

df['Mun Resid'] = df['Mun Resid'].apply(lambda x: None if x == '00000-XX' else x)

df['CodMuni']   = df['CodMuni'].apply(lambda x: None if x == '00000' else x)

df['SiglaUF']   = df['SiglaUF'].apply(lambda x: None if x == 'XX' else x)
### Correção da coluna Dt nascimento e criar a coluna idade

df['Dt Nascimento'] =  pd.to_datetime(df['Dt Nascimento'],format='%d/%m/%Y')

dataIdade =  pd.to_datetime('31/12/2018', format='%d/%m/%Y')

df['idade'] = df['Dt Nascimento'].apply(lambda x: dataIdade.year - x.year)
### Criando colunas novas

df['CID letra']  = df['CID'].apply(lambda x: x[0:1] if x!= None else x)

df['CID titulo'] = df['CID'].apply(lambda x: x[0:3] if x!= None else x)

### Observando as linhas iniciais do banco de dados

df.head(5)
### Observando quais variáveis possuem None

msno.matrix(df,figsize=(12,5))
print(df.info())

nameCol = df.columns

for i in nameCol:

    print(i,"\n  ",df[i].unique())
df.groupby('Espécie')['Espécie'].count().to_frame()
plt.figure(figsize=(17,12))

df.groupby('Espécie')['Espécie'].count().plot.barh()
df.groupby('Despacho')['Despacho'].count().to_frame()
# Quantidade de concessão por Despacho

plt.figure(figsize=(17,10))

df.groupby('Despacho')['Despacho'].count().plot.barh()
df.groupby('Clientela')['Clientela'].count().to_frame()
plt.figure(figsize=(17,8))

df.groupby('Clientela')['Clientela'].count().plot.pie()
df.groupby('Vínculo dependentes')['Vínculo dependentes'].count().to_frame()
plt.figure(figsize=(17,8))

df.groupby('Vínculo dependentes')['Vínculo dependentes'].count().plot.barh()
df.groupby('Forma Filiação')['Forma Filiação'].count().to_frame()
plt.figure(figsize=(17,8))

df.groupby('Forma Filiação')['Forma Filiação'].count().plot.barh()
df.groupby('UF')['UF'].count().to_frame()
plt.figure(figsize=(17,12))

df.groupby('UF')['UF'].count().plot.barh()
df.groupby('CID letra')['CID letra'].count().to_frame()
plt.figure(figsize=(17,12))

df.groupby('CID letra')['CID letra'].count().plot.barh()
df.describe()
### por possuir outlier, o hisograma a segir foi construido para a variavel RMI < 10

plt.figure(figsize=(17,8))

plt.hist(df[df['Qt SM RMI']<10]['Qt SM RMI'])
plt.figure(figsize=(17,8))

sns.distplot(df['idade'])
plt.figure(figsize=(17,8))

sns.boxplot(data=df,x="idade",orient="h")
plt.figure(figsize=(25,15))

sns.boxplot(df['UF'],df['Qt SM RMI'])

plt.xticks(rotation=90)
plt.figure(figsize=(25,15))

df_slice = df[df['Qt SM RMI']<5]

sns.boxplot(df_slice['UF'],df_slice['Qt SM RMI'])

plt.xticks(rotation=90)
### Tabela em percentual

pd.crosstab(df['CID letra'], df['SiglaUF']).apply(lambda r: round(r/r.sum()*100,2), axis=0)
df.groupby('Espécie')['idade'].mean().to_frame()
plt.figure(figsize=(25,15))

df.plot.scatter(x='idade',y='Qt SM RMI')
# Salvar o novo banco de dados

df.to_csv("concediddos201812_c.csv")