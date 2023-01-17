#para mostrar os gráficos dentro do notebook

%matplotlib inline 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (15,5)
#carregando arquivo

base = pd.read_csv("../input/gas-prices-in-brazil/2004-2019.tsv", sep='\t')
#mostra tamanho de linhas e colunas

print(base.shape)

#mostrando os 10 primeiros registros

print(base.head(10))

#mostra a lista de colunas e quantos registros unicos tem em cada uma, para ter ideia do dominio dessas informações

print(base.nunique())
#identificando os tipos de cada campo e sem tem algum registro nulo.

print(base.info())
#algumas análises sobre a base

print(base.describe())
#alterando o nome dos campos para ficar mais fácil de usar 

base.columns=['INDICE', 'DATA_INICIAL', 'DATA_FINAL', 'REGIAO', 'ESTADO', 'PRODUTO', 'NUM_POSTOS_PESQUISADOS',

            'UNIDADE_MEDIDA', 'MED_REVENDA', 'DP_REVENDA', 'MIN_REVENDA', 'MAX_REVENDA', 'MGM_REVENDA', 'COEFV_REVENDA',

            'MED_DISTRIBUICAO', 'DP_DISTRIBUICAO', 'MIN_DISTRIBUICAO', 'MAX_DISTRIBUICAO', 'COEFV_DISTRIBUICAO', 'MES', 'ANO']

print(base.info())
#altera campo objet para float

base['MGM_REVENDA'] = base['MGM_REVENDA'].apply(pd.to_numeric, errors='coerce')

base['MED_DISTRIBUICAO'] = base['MED_DISTRIBUICAO'].apply(pd.to_numeric, errors='coerce')

base['DP_DISTRIBUICAO'] = base['DP_DISTRIBUICAO'].apply(pd.to_numeric, errors='coerce')

base['MIN_DISTRIBUICAO'] = base['MIN_DISTRIBUICAO'].apply(pd.to_numeric, errors='coerce')

base['MAX_DISTRIBUICAO'] = base['MAX_DISTRIBUICAO'].apply(pd.to_numeric, errors='coerce')

base['COEFV_DISTRIBUICAO'] = base['COEFV_DISTRIBUICAO'].apply(pd.to_numeric, errors='coerce')

print(base.info())
base.groupby(['ANO', 'PRODUTO']).MED_REVENDA.mean().unstack().plot()

#utilizado o unstack() para que o agrupamento fosse mostrado em forma de tabela, senão ele combinaria os campos e não faria o plot corretamente

#aqui também não foi possível incluir o campo ESTADO para analisar junto
base.groupby(['PRODUTO', 'ESTADO']).MED_REVENDA.mean().unstack().plot(kind='bar')
base[(base.PRODUTO != 'GLP')].groupby(['ANO', 'PRODUTO']).MED_REVENDA.mean().unstack().plot()
base[(base.PRODUTO != 'GLP')].groupby(['PRODUTO', 'ESTADO']).MED_REVENDA.mean().unstack().plot(kind='bar', legend=None)
base[(base.PRODUTO != 'GLP')].groupby(['PRODUTO', 'REGIAO']).MED_REVENDA.mean().unstack().plot(kind='bar')
#contando registros NULOS por ano, mes, estado e produto

cont_nulos_MED_DISTRIBUICAO = pd.DataFrame(base[(base.MED_DISTRIBUICAO.isnull())]

                                           .groupby(['ANO', 'MES', 'ESTADO', 'PRODUTO']).

                                           PRODUTO.count().reset_index(name="QTD_NULOS"))
cont_nulos_MED_DISTRIBUICAO.groupby(['PRODUTO']).QTD_NULOS.sum().plot(kind='bar')
# Contando o total de registros não-nulos

cont_MED_DISTRIBUICAO = pd.DataFrame(base.groupby(['ANO', 'MES', 'ESTADO', 'PRODUTO'])

                                     .MED_DISTRIBUICAO.count().dropna().reset_index(name="QTD"))
cont_MED_DISTRIBUICAO.groupby(['PRODUTO']).QTD.sum().plot(kind='bar')
merge_med_distr = pd.merge(cont_MED_DISTRIBUICAO, cont_nulos_MED_DISTRIBUICAO, 

                           on=['ANO', 'MES', 'ESTADO', 'PRODUTO'] , how='left')
dfsum = merge_med_distr[(merge_med_distr.PRODUTO == 'GNV')].groupby(['ANO'])['QTD', 'QTD_NULOS'].sum().plot(kind='bar', stacked=True)
dfsum = merge_med_distr[(merge_med_distr.PRODUTO == 'GNV')].groupby(['ESTADO'])['QTD', 'QTD_NULOS'].sum().plot(kind='bar', stacked=True)
merge_med_distr.groupby(['ANO'])['QTD', 'QTD_NULOS'].sum().plot(kind='bar', stacked=True)
dfsum=merge_med_distr.groupby(['ANO'])['QTD', 'QTD_NULOS'].sum()

dfsum['PERC_NULOS']=dfsum['QTD_NULOS']/(dfsum['QTD']+dfsum['QTD_NULOS'])

print(dfsum)
dfsum=merge_med_distr.groupby(['ESTADO'])['QTD', 'QTD_NULOS'].sum()

dfsum['PERC_NULOS']=dfsum['QTD_NULOS']/(dfsum['QTD']+dfsum['QTD_NULOS'])

print(dfsum)
base_agrup = pd.DataFrame(base.groupby(['ANO', 'MES', 'ESTADO','PRODUTO'])['MED_DISTRIBUICAO'].mean().unstack().unstack().unstack())
base_agrup.info()
base_agrup.mean()
base_agrup.count()
base_transform = base_agrup.transform(lambda x: x.fillna(x.mean()))
base_transform.mean()
base_transform.count()
base_transform.groupby(['ANO']).mean()
base_agrup.groupby(['ANO']).mean()
base_semna = base.dropna(subset=['MED_DISTRIBUICAO'])
base_semna.info()
base_f = base_semna.drop(['INDICE', 'DATA_INICIAL', 'DATA_FINAL', 'NUM_POSTOS_PESQUISADOS', 'UNIDADE_MEDIDA', 

                         'DP_REVENDA', 'MIN_REVENDA', 'MAX_REVENDA', 'COEFV_REVENDA',

                        'DP_DISTRIBUICAO', 'MIN_DISTRIBUICAO', 'MAX_DISTRIBUICAO', 'COEFV_DISTRIBUICAO'], axis=1)
base_f['REVsobDIS'] = ((base_f.MED_REVENDA/base_f.MED_DISTRIBUICAO)-1)*100

plt.plot(base_f[(base_f.PRODUTO == 'GLP')].groupby(['ANO']).MED_REVENDA.mean())

plt.plot(base_f[(base_f.PRODUTO == 'GLP')].groupby(['ANO']).MED_DISTRIBUICAO.mean())

plt.plot(base_f[(base_f.PRODUTO == 'GLP')].groupby(['ANO']).REVsobDIS.mean())

plt.legend()

plt.title('Média de preço de revenda e distribuição do GLP, e a margem percentual da revenda "RevSobDis"')

plt.xlabel('ANO')

plt.ylabel('Média em R$, REVsobDIS em %')

plt.show()
# plt.plot(base_f[(base_f.PRODUTO == 'GLP')].groupby(['ANO', 'REGIAO']).MED_REVENDA.mean().unstack())

# plt.plot(base_f[(base_f.PRODUTO == 'GLP')].groupby(['ANO', 'REGIAO']).MED_DISTRIBUICAO.mean().unstack())

plt.plot(base_f[(base_f.PRODUTO == 'GLP')].groupby(['ANO', 'REGIAO']).REVsobDIS.mean().unstack())

plt.legend(base_f.REGIAO.unique())

plt.title('Média da margem percentual do valor de revenda sobre o valor de distribuição do GLP por Região')

plt.xlabel('ANO')

plt.ylabel('Média em %')

plt.show()
plt.plot(base_f[(base_f.PRODUTO == 'GLP')].groupby(['ANO', 'REGIAO']).MED_REVENDA.mean().unstack())

plt.plot(base_f[(base_f.PRODUTO == 'GLP')].groupby(['ANO', 'REGIAO']).MED_DISTRIBUICAO.mean().unstack())

plt.legend(base_f.REGIAO.unique())

plt.title('Média do preço de revenda e de distribuição do GLP por Região')

plt.xlabel('ANO')

plt.ylabel('Média em R$')

plt.show()
plt.plot(base_f.query('PRODUTO == "GLP" & REGIAO == "NORTE"').groupby(['ANO', 'ESTADO']).MED_REVENDA.mean().unstack())

plt.legend(base_f[(base_f.REGIAO == "NORTE")].ESTADO.unique())

plt.title('Média de preço de revenda do GLP por Estado, dos Estados da região Norte')

plt.xlabel('ANO')

plt.ylabel('Média em R$')

plt.ylim(20, 90)

plt.show()
# plt.plot(base_f.query('PRODUTO == "GLP" & REGIAO == "NORTE"').groupby(['ANO', 'ESTADO']).MED_REVENDA.mean().unstack())

plt.plot(base_f.query('PRODUTO == "GLP" & REGIAO == "NORTE"').groupby(['ANO', 'ESTADO']).MED_DISTRIBUICAO.mean().unstack())

plt.legend(base_f[(base_f.REGIAO == "NORTE")].ESTADO.unique())



plt.title('Média de preço de distribuição do GLP por Estado, dos Estados da região Norte')

plt.xlabel('ANO')

plt.ylabel('Média em R$')

plt.ylim(20, 90)

plt.show()
plt.plot(base_f.query('PRODUTO == "GLP" & REGIAO == "NORTE"').groupby(['ANO', 'ESTADO']).REVsobDIS.mean().unstack())

plt.legend(base_f[(base_f.REGIAO == "NORTE")].ESTADO.unique())

plt.title('Média da margem percentual do valor de revenda sobre o valor de distribuição do GLP por Estado da Região Norte')

plt.xlabel('ANO')

plt.ylabel('Média em %')

plt.show()
plt.plot(base_f.query('PRODUTO == "GLP" & REGIAO == "SUDESTE"').groupby(['ANO', 'ESTADO']).MED_REVENDA.mean().unstack())

plt.legend(base_f[(base_f.REGIAO == "SUDESTE")].ESTADO.unique())

plt.title('Média de preço de revenda do GLP por Estado, dos Estados da região Sudeste')

plt.xlabel('ANO')

plt.ylabel('Média em R$')

plt.ylim(20, 90)

plt.show()
plt.plot(base_f.query('PRODUTO == "GLP" & REGIAO == "SUDESTE"').groupby(['ANO', 'ESTADO']).MED_DISTRIBUICAO.mean().unstack())

plt.legend(base_f[(base_f.REGIAO == "SUDESTE")].ESTADO.unique())

plt.title('Média de preço de distribuição do GLP por Estado, dos Estados da região Sudeste')

plt.xlabel('ANO')

plt.ylabel('Média em R$')

plt.ylim(20, 90)

plt.show()
plt.plot(base_f.query('PRODUTO == "GLP" & REGIAO == "SUDESTE"').groupby(['ANO', 'ESTADO']).REVsobDIS.mean().unstack())

plt.legend(base_f[(base_f.REGIAO == "SUDESTE")].ESTADO.unique())

plt.title('Média da margem percentual do valor de revenda sobre o valor de distribuição do GLP por Estado da Região Sudeste')

plt.xlabel('ANO')

plt.ylabel('Média em %')

plt.show()