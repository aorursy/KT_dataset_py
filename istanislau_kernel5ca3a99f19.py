import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import numpy as np

import seaborn as sns
dados = pd.read_csv('/input/suicdios-por-armas-de-fogo-no-brasil/dados_ipea.csv', sep=';')
dados.shape
dados.info()
dados.head()
dados['valor'].describe().round(2)
AC = dados.query('nome== "AC"').reset_index().drop(columns=['index'])

AL = dados.query('nome== "AL"').reset_index().drop(columns=['index'])

AM = dados.query('nome== "AM"').reset_index().drop(columns=['index'])

AP = dados.query('nome== "AP"').reset_index().drop(columns=['index'])

BA = dados.query('nome== "BA"').reset_index().drop(columns=['index'])

CE = dados.query('nome== "CE"').reset_index().drop(columns=['index'])

DF = dados.query('nome== "DF"').reset_index().drop(columns=['index'])

ES = dados.query('nome== "ES"').reset_index().drop(columns=['index'])

GO = dados.query('nome== "GO"').reset_index().drop(columns=['index'])

MA = dados.query('nome== "MA"').reset_index().drop(columns=['index'])

MG = dados.query('nome== "MG"').reset_index().drop(columns=['index'])

MS = dados.query('nome== "MS"').reset_index().drop(columns=['index'])

MT = dados.query('nome== "MT"').reset_index().drop(columns=['index'])

PA = dados.query('nome== "PA"').reset_index().drop(columns=['index'])

PB = dados.query('nome== "PB"').reset_index().drop(columns=['index'])

PE = dados.query('nome== "PE"').reset_index().drop(columns=['index'])

PI = dados.query('nome== "PI"').reset_index().drop(columns=['index'])

PR = dados.query('nome== "PR"').reset_index().drop(columns=['index'])

RJ = dados.query('nome== "RJ"').reset_index().drop(columns=['index'])

RN = dados.query('nome== "RN"').reset_index().drop(columns=['index'])

RO = dados.query('nome== "RO"').reset_index().drop(columns=['index'])

RR = dados.query('nome== "RR"').reset_index().drop(columns=['index'])

RS = dados.query('nome== "RS"').reset_index().drop(columns=['index'])

SC = dados.query('nome== "SC"').reset_index().drop(columns=['index'])

SE = dados.query('nome== "SE"').reset_index().drop(columns=['index'])

SP = dados.query('nome== "SP"').reset_index().drop(columns=['index'])

TO = dados.query('nome== "TO"').reset_index().drop(columns=['index'])
estados = dados['nome'].unique()

selecao = dados['nome'].isin(estados)

dados = dados[selecao]

dados['nome'].drop_duplicates()

dados_estados = dados.groupby('nome')

dados_estados['valor'].describe().round(2).rename(columns = {'count':'Contagem','mean':'Média', 'std':'Desvio padrão',

                                                          'min':'Mínimo','50%':'Média','max':'Máximo'})
casos_estado = pd.DataFrame(dados_estados['valor'].sum())

casos_estado=casos_estado.reset_index()

dados_estados['valor'].sum()
anos = dados['período'].unique()

selecao = dados['período'].isin(anos)

dados_anos = dados.groupby('período')

dados_anos['valor'].describe().round(2).rename(columns = {'count':'Contagem','mean':'Média', 'std':'Desvio padrão',

                                                          'min':'Mínimo','50%':'Média','max':'Máximo'})

casos_ano = pd.DataFrame(dados_anos['valor'].sum())

dados_anos['valor'].sum()
dados['valor'].sum()
dados.corr().round(4)
grafico = sns.boxplot(y='valor', x='nome', data=dados, orient='v', width=0.5)

grafico.figure.set_size_inches(19, 10)

grafico.set_title('Casos por estados', fontsize=20)

grafico.set_ylabel('Casos', fontsize=16)

grafico.set_xlabel('Estados', fontsize=16)

grafico
grafico = sns.distplot(dados['valor'])

grafico.figure.set_size_inches(15, 8)

grafico.set_title('Distribuição de casos', fontsize=20)

grafico.set_ylabel('Casos', fontsize=16)

grafico.set_xlabel('Nº de casos nos esdado', fontsize=16)

grafico
plt.figure(figsize=(18,5))

sns.set(style="whitegrid")

ax = sns.barplot(y='valor', x="nome", data=casos_estado, palette="Blues_d")

ax
fig, grafico = plt.subplots(figsize=(18,6))

grafico.set_title('Casos de suicidio', fontsize=20)

grafico.set_ylabel('Nº de casos', fontsize=16)

grafico.set_xlabel('Anos', fontsize=16)

grafico = casos_ano['valor'].plot(fontsize=14)
AC['aumento'] = AC['valor'].shift(-1) - AC['valor'] 

AL['aumento'] = AL['valor'].shift(-1) - AL['valor'] 

AM['aumento'] = AM['valor'].shift(-1) - AM['valor'] 

AP['aumento'] = AP['valor'].shift(-1) - AP['valor'] 

BA['aumento'] = BA['valor'].shift(-1) - BA['valor'] 

CE['aumento'] = CE['valor'].shift(-1) - CE['valor'] 

DF['aumento'] = DF['valor'].shift(-1) - DF['valor'] 

ES['aumento'] = ES['valor'].shift(-1) - ES['valor'] 

GO['aumento'] = GO['valor'].shift(-1) - GO['valor']  

MA['aumento'] = MA['valor'].shift(-1) - MA['valor'] 

MG['aumento'] = MG['valor'].shift(-1) - MG['valor'] 

MS['aumento'] = MS['valor'].shift(-1) - MS['valor'] 

MT['aumento'] = MT['valor'].shift(-1) - MT['valor'] 

PA['aumento'] = PA['valor'].shift(-1) - PA['valor'] 

PB['aumento'] = PB['valor'].shift(-1) - PB['valor'] 

PE['aumento'] = PE['valor'].shift(-1) - PE['valor']  

PI['aumento'] = PI['valor'].shift(-1) - PI['valor'] 

PR['aumento'] = PR['valor'].shift(-1) - PR['valor'] 

RJ['aumento'] = RJ['valor'].shift(-1) - RJ['valor'] 

RN['aumento'] = RN['valor'].shift(-1) - RN['valor'] 

RO['aumento'] = RO['valor'].shift(-1) - RO['valor'] 

RR['aumento'] = RR['valor'].shift(-1) - RR['valor'] 

RS['aumento'] = RS['valor'].shift(-1) - RS['valor']  

SC['aumento'] = SC['valor'].shift(-1) - SC['valor'] 

SE['aumento'] = SE['valor'].shift(-1) - SE['valor']  

SP['aumento'] = SP['valor'].shift(-1) - SP['valor'] 

TO['aumento'] = TO['valor'].shift(-1) - TO['valor'] 
AC['aceleracao'] = AC['aumento'].shift(-1) - AC['aumento'] 

AL['aceleracao'] = AL['aumento'].shift(-1) - AL['aumento'] 

AM['aceleracao'] = AM['aumento'].shift(-1) - AM['aumento'] 

AP['aceleracao'] = AP['aumento'].shift(-1) - AP['aumento'] 

BA['aceleracao'] = BA['aumento'].shift(-1) - BA['aumento'] 

CE['aceleracao'] = CE['aumento'].shift(-1) - CE['aumento'] 

DF['aceleracao'] = DF['aumento'].shift(-1) - DF['aumento'] 

ES['aceleracao'] = ES['aumento'].shift(-1) - ES['aumento'] 

GO['aceleracao'] = GO['aumento'].shift(-1) - GO['aumento']  

MA['aceleracao'] = MA['aumento'].shift(-1) - MA['aumento'] 

MG['aceleracao'] = MG['aumento'].shift(-1) - MG['aumento'] 

MS['aceleracao'] = MS['aumento'].shift(-1) - MS['aumento'] 

MT['aceleracao'] = MT['aumento'].shift(-1) - MT['aumento'] 

PA['aceleracao'] = PA['aumento'].shift(-1) - PA['aumento'] 

PB['aceleracao'] = PB['aumento'].shift(-1) - PB['aumento'] 

PE['aceleracao'] = PE['aumento'].shift(-1) - PE['aumento']  

PI['aceleracao'] = PI['aumento'].shift(-1) - PI['aumento'] 

PR['aceleracao'] = PR['aumento'].shift(-1) - PR['aumento'] 

RJ['aceleracao'] = RJ['aumento'].shift(-1) - RJ['aumento'] 

RN['aceleracao'] = RN['aumento'].shift(-1) - RN['aumento'] 

RO['aceleracao'] = RO['aumento'].shift(-1) - RO['aumento'] 

RR['aceleracao'] = RR['aumento'].shift(-1) - RR['aumento'] 

RS['aceleracao'] = RS['aumento'].shift(-1) - RS['aumento']  

SC['aceleracao'] = SC['aumento'].shift(-1) - SC['aumento'] 

SE['aceleracao'] = SE['aumento'].shift(-1) - SE['aumento']  

SP['aceleracao'] = SP['aumento'].shift(-1) - SP['aumento'] 

TO['aceleracao'] = TO['aumento'].shift(-1) - TO['aumento'] 
AC= AC.fillna(0)

AL= AL.fillna(0)

AM= AM.fillna(0)

AP= AP.fillna(0)

BA= BA.fillna(0)

CE= CE.fillna(0)

DF= DF.fillna(0)

ES= ES.fillna(0)

GO= GO.fillna(0)

MG= MG.fillna(0)

MS= MS.fillna(0)

MT= MT.fillna(0)

PA= PA.fillna(0)

PB= PB.fillna(0)

PE= PE.fillna(0)

PI= PI.fillna(0)

PR= PR.fillna(0)

RJ= RJ.fillna(0)

RN= RN.fillna(0)

RO= RO.fillna(0)

RR= RR.fillna(0)

RS= RS.fillna(0)

SC= SC.fillna(0)

SE= SE.fillna(0)

SP= SP.fillna(0)

TO= TO.fillna(0)
grafico = sns.pairplot(AC, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(AL, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(AP, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(AM, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(BA, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(CE, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(DF, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(ES, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(GO, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(MA, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(MT, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(MS, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(MG, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(PA, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(PB, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(PR, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(PE, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(PI, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(RJ, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(RN, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(RS, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(RO, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(RR, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(SC, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(SP, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(SE, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico
grafico = sns.pairplot(AM, y_vars='período', x_vars=['valor', 'aumento', 'aceleracao'], kind='reg', height=6)

grafico.fig.suptitle('Dispersão período e nº de casos, aumento e aceleração', fontsize=20, y=1.1)

grafico