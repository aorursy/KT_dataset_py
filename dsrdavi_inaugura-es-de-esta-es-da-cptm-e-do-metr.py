import pandas as pd

import numpy as np



dados = pd.read_csv('../input/inauguracao-de-estacoes-do-metrosp-e-da-cptm/inauguracoes.csv')



dados
#desmembrando a data de inauguração em três colunas



dados['Dia'] = pd.to_datetime(dados.Inauguração).dt.day

dados['Mês'] = pd.to_datetime(dados.Inauguração).dt.month

dados['Ano'] = pd.to_datetime(dados.Inauguração).dt.year



dados.head()
dados[dados.Linha == 7].sort_values(by = ['Ano', 'Mês', 'Dia', 'Nome'], ascending = True)
dados.groupby('Linha').agg({'Idade' : np.average}).sort_values(by = ['Idade'], ascending = False)
dados.groupby('Construção').agg({'Idade' : np.average}).sort_values(by = ['Idade'], ascending = False)
dados.groupby('Ano')['Nome'].count()
import matplotlib.pyplot as plt

import random



fig = plt.figure(figsize = (20,12))

ax = fig.add_subplot(1, 1, 1) 

ax.set_xlabel('Ano', fontsize = 15)

ax.set_ylabel('Linha', fontsize = 15)

ax.set_title('Inaugurações de estações da CPTM e do Metrô', fontsize = 20)



legenda = dados.Linha.sort_values().apply(lambda l: 'Linha ' + str(l)).unique().tolist()

linhas = dados.Linha.sort_values().unique().tolist()



n_cores = 13



cores = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n_cores)]



anos = pd.to_datetime(dados.loc[:, 'Inauguração']).dt.year



plt.xticks(np.arange(min(anos), max(anos) + 5, 5.0))

plt.yticks(np.arange(1, 16, 1.0))



ax.legend(legenda)

ax.grid()



for l, cor in zip(linhas, cores):

    indices = dados['Linha'] == l

    ax.scatter(pd.to_datetime(dados.loc[indices, 'Inauguração']).dt.year,

               dados.loc[indices, 'Linha'],

               c = cor,

               s = 40)
fig = plt.figure(figsize = (20,6))



ax = fig.add_subplot(1, 1, 1) 



ax.set_xlabel('Ano', fontsize = 15)

ax.set_ylabel('Estações inauguradas', fontsize = 15)

ax.set_title('Inaugurações de estações da CPTM e do Metrô', fontsize = 20)



ax.grid()



anos = dados.Ano.unique()

inauguracoes = dados.groupby('Ano')['Nome'].count().tolist()



plt.xticks(np.arange(min(anos), max(anos) + 4, 4.0))

plt.yticks(np.arange(min(inauguracoes), max(inauguracoes) + 1, 1.0))



ax.plot(anos, inauguracoes)
fig = plt.figure(figsize = (20,6))



ax = fig.add_subplot(1, 1, 1) 



ax.set_xlabel('Ano', fontsize = 15)

ax.set_ylabel('Estações inauguradas', fontsize = 15)

ax.set_title('Inaugurações de estações da CPTM e do Metrô', fontsize = 20)



anos = dados.Ano.unique()

inauguracoes = dados.groupby('Ano')['Nome'].count().tolist()



plt.xticks(np.arange(min(anos), max(anos) + 4, 4.0))

plt.yticks(np.arange(min(inauguracoes), max(inauguracoes) + 1, 1.0))



ax.grid()



plt.bar(anos, inauguracoes, color = cores)
fig = plt.figure(figsize = (20,6))



ax = fig.add_subplot(1, 1, 1) 



ax.set_xlabel('Ano', fontsize = 15)

ax.set_ylabel('Estações inauguradas', fontsize = 15)

ax.set_title('Inaugurações de estações da CPTM', fontsize = 20)



anos = dados[dados.Construção == 'CPTM'].Ano.unique()

inauguracoes = dados[dados.Construção == 'CPTM'].groupby('Ano')['Nome'].count().tolist()



plt.xticks(np.arange(min(anos), max(anos) + 4, 4.0))

plt.yticks(np.arange(min(inauguracoes), max(inauguracoes) + 1, 1.0))



ax.grid()



plt.bar(anos, inauguracoes, color = cores)
fig = plt.figure(figsize = (20,6))



ax = fig.add_subplot(1, 1, 1) 



ax.grid()

ax.set_xlabel('Ano', fontsize = 15)

ax.set_ylabel('Estações inauguradas', fontsize = 15)

ax.set_title('Inaugurações de estações do Metrô', fontsize = 20)



anos = dados[dados.Construção == 'Metrô'].Ano.unique()

inauguracoes = dados[dados.Construção == 'Metrô'].groupby('Ano')['Nome'].count().tolist()



plt.xticks(np.arange(min(anos), max(anos) + 2, 2.0))

plt.yticks(np.arange(min(inauguracoes), max(inauguracoes) + 1, 1.0))



plt.bar(anos, inauguracoes, color = cores)
inauguracoes = dados.loc[dados.Construção == 'Metrô', 'Ano']

bins = dados[dados.Construção == 'Metrô'].Ano.unique()



plt.hist(inauguracoes, bins)
#inaugurações em anos eleitorais (pós-ditadura, todos os anos pares)



anos_eleitorais = list(range(1988, 2020, 2))



anos_eleitorais
dados['AnoEleitoral'] = dados.apply(lambda row: 'Sim' if row.Ano in anos_eleitorais else 'Não', 1)

dados
estacoes_eleitorais = dados[dados.AnoEleitoral == 'Sim'].groupby('Construção').agg({'Construção' : 'count'})

estacoes_eleitorais.columns = ['Estações']



estacoes_eleitorais = estacoes_eleitorais.reset_index()

estacoes_eleitorais
plt.bar(estacoes_eleitorais.Construção, estacoes_eleitorais.Estações, label = 'Inaugurações', color = 'rb')

plt.title('Estações inauguradas em ano eleitoral desde 1988')

plt.ylabel('Estações inauguradas')
estacoes_eleitorais = dados[dados.Ano >= 1988].groupby(['Construção', 'AnoEleitoral']).agg({'AnoEleitoral' : 'count'})

estacoes_eleitorais.columns = ['Estações']



estacoes_eleitorais = estacoes_eleitorais.reset_index()

estacoes_eleitorais
labels = ['Ano Comum', 'Ano Eleitoral']



dados_cptm = list(estacoes_eleitorais.loc[estacoes_eleitorais.Construção == 'CPTM', 'Estações'])

dados_metro = list(estacoes_eleitorais.loc[estacoes_eleitorais.Construção == 'Metrô', 'Estações'])



x = np.arange(len(labels))

width = 0.35



fig, ax = plt.subplots()



bar_cptm = plt.bar(x - (width / 2), dados_cptm, width, label = 'CPTM', color = 'r')

bar_metro = plt.bar(x + (width / 2), dados_metro, width, label = 'Metrô', color = 'b')



ax.set_title('Inaugurações de estações a partir de 1988')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.set_ylabel('Estações inauguradas')

ax.legend()
anos = dados[(dados.AnoEleitoral == 'Sim') & (dados.Construção == 'Metrô')].Ano.unique()

inauguracoes = dados[(dados.AnoEleitoral == 'Sim') & (dados.Construção == 'Metrô')].groupby('Ano')['Nome'].count().tolist()



plt.figure(figsize = (20,6))



plt.bar(anos, inauguracoes)



plt.title('Inaugurações de estações do Metrô em anos eleitorais', fontsize = 20)

plt.xlabel('Ano', fontsize = 15)

plt.ylabel('Estações inauguradas', fontsize = 15)

plt.xticks(np.arange(min(anos), max(anos) + 2, 2.0))

plt.yticks(np.arange(0, max(inauguracoes) + 1, 1.0))

plt.grid()
anos = dados[(dados.AnoEleitoral == 'Sim') & (dados.Construção == 'CPTM')].Ano.unique()

inauguracoes = dados[(dados.AnoEleitoral == 'Sim') & (dados.Construção == 'CPTM')].groupby('Ano')['Nome'].count().tolist()



plt.figure(figsize = (20, 6))



plt.bar(anos, inauguracoes, color = 'r')



plt.title('Inaugurações de estações da CPTM em anos eleitorais', fontsize = 20)

plt.xlabel('Ano', fontsize = 15)

plt.ylabel('Estações inauguradas', fontsize = 15)

plt.xticks(np.arange(min(anos), max(anos) + 2, 2.0))

plt.yticks(np.arange(0, max(inauguracoes) + 1, 1.0))

plt.grid()
dados_cptm = []

dados_metro = []



for a in anos_eleitorais:

    dados_cptm.append(len(dados[(dados.Construção == 'CPTM') & (dados.Ano == a)]))

    dados_metro.append(len(dados[(dados.Construção == 'Metrô') & (dados.Ano == a)]))



x = np.arange(len(anos_eleitorais))

width = 0.25



fig, ax = plt.subplots(figsize = (15, 5))



bar_cptm = plt.bar(x - (width / 2), dados_cptm, width, label = 'CPTM', color = 'r')

bar_metro = plt.bar(x + (width / 2), dados_metro, width, label = 'Metrô', color = 'b')



ax.set_title('Inaugurações de estações em anos eleitorais')

ax.set_xticks(x)

ax.set_xticklabels(anos_eleitorais)

ax.set_ylabel('Estações inauguradas')

ax.set_yticks(np.arange(min(dados_metro), max(dados_metro) + 1, 1.0))

ax.legend()

ax.grid()