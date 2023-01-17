# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#imprime a área que está sendo utilizada

print(os.getcwd())
#importando a base de dados - amostra de base de clientes pf de banco descaracterizada 

base = pd.read_csv('../input/BaseEstudo_Python.csv', sep=';', decimal=',')
#descrição do tipo de cada uma das variáveis do banco

base.dtypes
import matplotlib.pyplot as plt

import numpy as np

from matplotlib.patches import Polygon

import pandas as pd

import re

#from bokeh.charts import Histogram, show

import seaborn as sns

import numpy as np

import datetime as dt
# Descrição das variáveis

# Estatísticas básicas



# Verificar mínimos e máximos para garantir se estão dentro dos limites esperados

# Verificar intervalo de variação da medida

# Verificar possíveis outliers

base.describe()
Segmento_status = sns.countplot(base['cSegmento'])



# Percebe-se que a quantidade de segmentos é pequena e que a grande maioria dos clientes está concentrada em um segmento
# Análise sobre a Carteira

Carteira_status = sns.countplot(base['cCarteira'])



# Percebe-se que o número de carteiras é alto e que a grande maioria dos clientes encontra-se em uma carteira
# Análise sobre o gênero do cliente

Genero_status = sns.countplot(base['cGenero'])



# Percebe-se um certo equilíbrio de clientes em relação ao gênero
# Análise sobre a situação cadastral do cliente

SituacaoCad_status = sns.countplot(base['cSituacao_Cadastral'])



# Percebe-se uma divisão de clientes em relação à situação cadastral
# Análise sobre UF de residência do cliente

UF_status = sns.countplot(base['vResidencia_UF'])



# Percebe-se uma distribuição quase uniforme de clientes para a maioria das UFs
# Análise sobre indicativo de servidor público para o cliente

ServPub_status = sns.countplot(base['bNch_Servidor_Publico'])



# Percebe-se uma concentração de clientes em relação ao indicativo de servidor público
# Analisando as UF por rendas

base.groupby(['vResidencia_UF']).agg({'iPV_Referencia' : 'count', 'mConta_Corrente_Saldo' : 'mean', 'mRenda_Presumida' : 'mean'}).sort_values(by=['mRenda_Presumida'], ascending=False)
# Analisando as rendas em relação ao saldo em conta corrente

figsize=(10,6)

plt.subplots(figsize=figsize)

wh_ = base.loc[:,['mConta_Corrente_Saldo','mRenda_Presumida']].values

plt.plot(wh_[:,0], 



         wh_[:,1],



         'r.'



         )

plt.xlabel('Saldo de CC')

plt.ylabel('Renda Presumida')

plt.title('Saldo de CC X Renda Presumida')

plt.grid(True)

plt.show()

plt.tight_layout()
#verificando valores nulos para tipo de pessoa

base[base['iTipo_Pessoa'].isnull()].head()
#verificando valores nulos para UF

base[base['vResidencia_UF'].isnull()].head()
# Contagem de valores para UF

base.vResidencia_UF.value_counts()
#Definindo o gráfico para tipo de pessoa

base['iTipo_Pessoa'].value_counts().head(10).plot(kind='bar', figsize=(11,5), grid=False, rot=0, color='green')

#melhorando o gráfico

plt.title('Clientes por Tipo de Pessoa')

plt.xlabel('Tipo de Pessoa')

plt.ylabel('Quantidade de clientes')

plt.show()
# Analisando informações cruzadas: Tipo Pessoa x Gênero

dfPessoaGenero = pd.crosstab(base['iTipo_Pessoa'],base['cGenero'])

dfPessoaGenero.head()

dfPessoaGenero['Total'] = dfPessoaGenero.sum(axis=1)

dfPessoaGenero.head()
# Analisando informações cruzadas: Tipo Pessoa x UF

dfPessoaUF = pd.crosstab(base['iTipo_Pessoa'],base['vResidencia_UF'])

dfPessoaUF.head()

dfPessoaUF['Total'] = dfPessoaUF.sum(axis=1)

dfPessoaUF.head()
# Analisando de forma númerica:

print(base.count())



# Uma vez que os dados relativos tem o total de 10.000 clientes, é possível efetuar a estimativa de missing values.
# Verificar outliers para Renda Presumida

sns.boxplot(data=base,x="mRenda_Presumida",orient="v")
# Verificar outliers para Renda cadastrada

sns.boxplot(data=base,x="mRenda_SICLI",orient="v")
# Foram utilizados diferentes boxplots para mensurar os outliers.



# Lembrando: o boxplot se divide entre máximo, mínimo, e os chamados quartiles ou quartis, e dentro da parte preenchida se encontra a mediana dos dados, os outliers por sua vez se encontram acima do máximo ou em alguns casos abaixo do mínimo.



# Para tomar a decisão de retirar ou não os outliers, podemos simplesmente identificá-los a partir dos boxplots e partir para sua retirada ou tentar verificar seu real impacto nos dados.



# Dependendo do que será feito nas etapas seguintes é algo extremamente necessário esse tratamento, por exemplo, em casos onde seja feito o uso do k-means para identificar grupos de clientes (O k-means é um algoritmo de clusterização que é sensível aos outliers, ou seja o impacto causado por esses dados fora da curva, podem gerar impactos negativos e análises erradas).



# Aqui verificou-se se os dados apresentavam alguma normalidade e caso não apresentassem qual seria o impacto da não retirada dos outliers, para isso foram feitos histogramas e o calculo do skewness, que apresenta o quão distante os dados estão de uma normal, nesse sentido quanto mais o valor estiver próximo a 0 de skewness melhor.



Renda_Presumida = sns.distplot(base['mRenda_Presumida'])

plt.title("Distribuição da Renda Presumida")

plt.show(Renda_Presumida)
# No caso da Renda Presemida o skewness mesmo com outliers apresentava o valor de 0.096, que pode ser considerado bom, no entanto outras variáveis não se pareciam nada com uma normal, no entanto decidiu-se por não se retirar os outliers, e isso será mostrado nos tópicos que serão apresentados posteriormente.

# calculando o skewness: skewness = 3(média – mediana) / desvio padrão

# Mas a questão que fica é de como remover os outliers, para tal existem diferentes formas, como a remoção a partir do desvio padrão, a padronização via score Z ou algo mais incisivo que seria a remoção utilizando o cálculo do inter quartil (interquartile)

# Retirar utilizando o desvio padrão:

baseSemOutliers=base[np.abs(base["mRenda_Presumida"]-base["mRenda_Presumida"].mean())<=(3*base["mRenda_Presumida"].std())]

baseSemOutliers
# Ajustar tipos de campos datas

import datetime as dt

base['dData_Nascimento'] = pd.to_datetime(base['dData_Nascimento']).dt.date

base['dInicio_Relacionamento'] = pd.to_datetime(base['dInicio_Relacionamento']).dt.date

base.head(10)
# Verificando ajustes

base.dtypes
# A remoção apresentada anteriormente faz efeito apenas na variável mRenda_Presumida, nesse sentido seria necessário a sua aplicação em todas outras variáveis, o problema identificado é que as linhas com outliers não são deletadas mas tem seus valores alterados para NaN, e isso traz a necessidade de um novo tratamento após a remoção desse outliers.



# Criando– novas variáveis

# Nos dados são apresentadas variáveis que se dividem entre categorias, intervalares e aquelas que apresentam datas, parte dessas variáveis precisam ser transformadas, nesse sentido foram criadas variáveis que apresenta valores mais simples de serem tratados.



# PossuiConsigado;

base['PossuiConsignado'] = np.where((base['bConsignado_Utilizado'] == 1) | \

    (base['bConsignado_INSS'] == 1) | (base['bConsignado_Publico'] == 1) | (base['bConsignado_Outros'] == 1), 1, 0)



# PossuiSeguro;

base['PossuiSeguro'] = np.where((base['bSeguro_Vida'] == 1) | \

    (base['bSeguro_Auto'] == 1) | (base['bSeguro_Residencial'] == 1) | (base['bSeguro_Prestamista'] == 1), 1, 0)



# TotalInvest – Soma de todos os valores de investimentos do cliente;

base['TotalInvest'] = base['mInvest_Fundos_Saldo'] + base['mInvest_CDB_RDB_Saldo'] + base['mInvest_LCI_Saldo'] + base['mInvest_LCA_Saldo']



# TotalSeguro - Soma de todos os valores de seguros do cliente;

base['TotalSeguro'] = base['mSeguro_Vida_Saldo'] + base['mSeguro_Auto_Saldo'] + base['mSeguro_Resid_Saldo'] + base['mSeguro_Prest_Saldo'] + base['mSeguro_Odont_Saldo'] + base['mOutros_Seguros_Saldo']



# TotalConsorcio - Soma de todos os valores de consórcio do cliente;

base['TotalConsorcio'] = base['mConsorcio_Imob_Saldo'] + base['mConsorcio_Veic_Saldo']



# TotalConsignado - Soma de todos os valores de consignado do cliente;

base['TotalConsignado'] = base['mConsignado_INSS_Saldo'] + base['mConsignado_Publico_Saldo'] + base['mConsignado_Outros_Saldo']



base.head(10)
# Checando Coerência



# Nessa etapa foi efetuada a checagem de coerência das variáveis criadas, como por exemplo, o TotalConsignado e PossuiConsignado, uma vez que não teria muito sentido obter totais em seguro se o cliente não tiver seguro.



gastoConsigando = base['TotalConsignado'] > 0

semConsignado = base['PossuiConsignado'] ==0

aux = base['TotalConsignado'].mean()

base['TotalConsignado'] = np.where(gastoConsigando & semConsignado, aux, base['TotalConsignado'])

base.head(10)
# Caso tal incoerência seja identificada é necessário o reajuste das variáveis e o recálculo de forma a que isso não ocorra.



# Segmentação

# A segmentação aqui serve como uma forma de “quebrar” os dados em sub-grupos homogêneos, gerando grupos de variáveis que se correlacionam, tornando mais fácil compreender os clientes, para tal foram utilizados três segmentos:



# Perfil Cliente

# As variáveis foram divididas de acordo com os clientes e suas necessidades, levando em consideração a quantia gasta em produtos, de acordo com o dataset, existem seis variáveis que podem ser utilizadas nessa segmentação.

# iTipo_Pessoa; Idade; IdadeRelacionamento; cGenero; mRenda_SICLI; mRenda_Presumida; vResidencia_UF



# Consumo de Produtos

# Com as variáveis dessa segmentação é possível compreender os clientes em termos de receita gerada, perfil e relacionamento referente aos clientes, algumas variáveis utilizadas remetem as que foram geradas anteriormente.

# TotalInvest; TotalSeguro; TotalConsorcio; TotalConsignado;



# Correlação entre as variáveis

# Uma vez feita a segmentação das variáveis, podemos efetuar uma verificação de correlação mais concisa entre as variáveis de cada segmento, isso ajuda a obter uma melhor percepção do real impacto dessas variáveis nos dados, compreendendo assim quais estão positivamente relacionadas e quais são aquelas negativamente relacionadas.



#CORRELATION MATRICES

import seaborn as sns

corr = base[['iTipo_Pessoa','cGenero','mRenda_SICLI','mRenda_Presumida','vResidencia_UF','bConta_Corr_Mov','bCROT_Utilizado','bCDC_Utilizado','bPerfil_Digital','bPoupanca_Mov','bCobranca','vRating_Cliente','bFiltro_Dirigente']]

corr = base[['TotalInvest','TotalSeguro','TotalConsorcio','TotalConsignado']]

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)]=True

f, ax = plt.subplots(figsize=(11,9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap =cmap, vmax=1, vmin=-1, center=0, square=True, linewidth=.5,cbar_kws={"shrink":.5})

f.savefig('myimage.png', format='png', dpi=1200)

corr = base.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220,10,as_cmap=True), square=True, ax=ax)



# O código acima foi utilizado em ambas matrizes de correlação, nessas matrizes a cor vermelha representa correlações positivas enquanto o azul representa correlação negativa.
#quantidade total de carteiras

#a função nunique retorna a quantidade de valores únicos em um objeto

base.cCarteira.nunique() 
# quantidade total de clientes por carteira

base['cCarteira'].value_counts()
#media de Renda Presumida por Carteira

mediaPorCarteira = base.groupby('cCarteira').mRenda_Presumida.count()

mediaPorCarteira
# Visualizando dados

base.bCROT_Utilizado.plot(kind='hist', bins=20)

base.bCDC_Utilizado.plot(kind='hist', bins=20)
# Analisando os segmentos de clientes

base['cSegmento'].value_counts()
#histograma melhorado 

#from bokeh.charts import Histogram, show

# entre os melhores Segmento

melhoresSegs = base[(base.cSegmento == 'GV') | (base.cSegmento == 'GC') | (base.cSegmento == 'GR') | (base.cSegmento == 'Cl') | (base.cSegmento == 'EE') | (base.cSegmento == 'EF') | (base.cSegmento == 'ES') | (base.cSegmento == 'CI')]

#box plot para analisar saldos dos melhores segmentos

sns.set(style="whitegrid", color_codes=True)

sns.boxplot(x="cSegmento", y="mConta_Corrente_Saldo", hue="cSegmento", data=melhoresSegs, palette="PRGn")

sns.despine(offset=100, trim=True)
# Segmentos de melhores clientes com renda

melhoresRendas = base[base['mRenda_Presumida']> 10000]

grouped = melhoresRendas.groupby('cSegmento')

qtPorSeg = grouped.count()['cCarteira'].sort_values(ascending = False)

 

ax = sns.countplot(x = 'cSegmento', data = melhoresRendas, order = qtPorSeg.index)

ax.set_xticklabels(labels = qtPorSeg.index, rotation='vertical')

ax.set_ylabel('Numero de clientes')

ax.set_xlabel('Segmento')

ax.set_title('Segmentos com os melhores clientes')
#criando um novo dataset com os atributos que se deseja avaliar a correlação

analise = base.loc[:,['iTipo_Pessoa','cGenero','mRenda_SICLI','mRenda_Presumida','vResidencia_UF','bConta_Corr_Mov','bCROT_Utilizado','bCDC_Utilizado','bPerfil_Digital','bPoupanca_Mov','bCobranca','vRating_Cliente','bFiltro_Dirigente']]

 

#fazendo a matriz de correlação

corr = analise.corr()

 

#plotando a matriz de correlação

ax = sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values,

linewidths=0.25, vmax=1.0, square=True, cmap = 'PuBu', linecolor='black', annot=False)



# Lembrando que precisamos ignorar a diagonal principal, pois se trata da relação de uma variável com si mesma. 

# A partir da imagem podemos comprovar diversas relações intuitivas, como Perfil Difital e Movimentação de Conta Corrente,  Movimentação de Conta Corrente e Crot utilizado.
#Reduzindo o número de colunas

base2 = base.drop(columns=['cCarteira','iPV_Referencia','bNch_Construcao_Civil','iModelo_Retencao','mFaturamento_Anual','mFaturamento_Anual','mFaturamento_Anual_Nao_Comprovado','dFaturamento_Apuracao','iFatur_Anual_Ano_Ref','iResidencia_Cidade_IBGE','vResidencia_UF','bGiroCaixaFacil_Disp','bGiroCaixaFacil_Util','mGiroCaixaFacil_Util_Sd','dGiroCaixaFacil_Venc_Aval','mGiroCaixaFacil_Lim_Aval','bCredito_Empresa','mVolume_Credito_Empresa','bCobranca','dAval_Tomador_PJ','iPV_Ag_Digital','iNatureza_Juridica','mCredito_Conta_Corrente_3Meses','fEngajamento_PJ'])
# Criando uma base apenas com pessoa física - iTipo_Pessoa = 1 e que sem óbito 

base3=base2[base2.iTipo_Pessoa==1]

base3=base3[base3.bFiltro_Obito==0]
#retirando variável tipo pessoa já que não há necessidade

base3 = base3.drop(columns=['iTipo_Pessoa','bFiltro_Obito'])
#Correlação entre as variáveis da base

cor = base3.corr()

cor

import seaborn as sns; sns.set()

ax = sns.heatmap(cor)
#A partir da avaliação da correlação, considerando como variável targe a utilização do CROT, as seguintes variáveis foram consideradas

#pouco relevantes para o estudo.

base3 = base3.drop(columns=['bCred_Salario_001','bCred_Salario_013','bCred_Salario_023','bCred_Salario_037','bCred_Salario_outras','iIC_Total','iIC_Qualificado','mVolume_Negocios','bAdesao_SMS','mPoupanca_Saldo','mInvest_Fundos_Saldo','bInvest_CDB_RDB','mInvest_CDB_RDB_Saldo','bInvest_LCI','mInvest_LCI_Saldo','bInvest_LCA','mInvest_LCA_Saldo','bSeguro_Vida','mSeguro_Vida_Saldo','bSeguro_Auto','mSeguro_Auto_Saldo','bSeguro_Residencial','mSeguro_Resid_Saldo','bSeguro_Prestamista','mSeguro_Prest_Saldo','bPrevidencia','mPrevidencia_Saldo','bCapitalizacao','mCapitalizacao_Saldo','bConsorcio_Imobiliario','mConsorcio_Imob_Saldo','bConsorcio_Veiculos','mConsorcio_Veic_Saldo','bSeguro_Odontologico','mSeguro_Odont_Saldo','bOutros_Seguros','mOutros_Seguros_Saldo','vRating_Cliente','bFiltro_Gestor_Patrim','bCliente_Ag_Digital','vGerente_Ag_Digital','bConta_Caixa_Facil','mConta_Caixa_Facil_Saldo','bCaixa_Seguradora','bPrejuizo_SIAPC','bCadastro_Qualificado','bPerfil_Digital','bIBC_Utilizacao_60d','bMobile_Utilizacao_60d'])
# correlação

cor2 = base3.corr()

cor2

import seaborn as sns; sns.set()

ax = sns.heatmap(cor2)
#Esse método retorna o valor de simetria de cada coluna do dataset.

#Um valor zero indica uma distribuição simétrica, um valor maior que zero ou menor indica uma distribuição assimétrica. 

base3.skew()
# correlação

cor3 = base3[['bCROT_Contratado','mRenda_Presumida','bNch_Credito_Salario',

                'bNch_Servidor_Publico', 'mMargem_Contribuicao', 'mEndivid_BACEN',

             'iConta_Corr_Mov_Qtde_Mes','bPoupanca','bPoupanca_Mov','iPoupanca_Mov_Qtde_Mes',

             'bPoupanca_Integ','mPoupanca_Integ_Saldo','bFiltro_Renegociacao','bDebito_Automatico',

             'bCartao_Ativado']].corr()

import seaborn as sns; sns.set()

ax = sns.heatmap(cor3)

# Observe as correlações que surgem entre as variáveis, Servidor Público x Mov Poupança, Renegociação x Cartão Ativado, dentre outras.
# visualizando dados

sns.boxplot(data=base, x='bCROT_Contratado', y ='mRenda_Presumida')
# analisando dados - substituindo valores nan por categorias

base3.fillna(value={'bCROT_Contratado' : 2 }, inplace=True)

base3.fillna(value={'cSegmento' : 'NA' }, inplace=True)

base3.fillna(value={'cGenero' : 'NA' }, inplace=True)

base3.bCROT_Contratado.value_counts()

base3.cSegmento.value_counts()

base3.cGenero.value_counts()
# visualizando dados

import matplotlib.pyplot as plt

y= base3['bCROT_Contratado']

x= base3['cSegmento']

bar_color = 'yellow'

plt.bar(x,y, color=bar_color)

plt.show()
# analisando dados

base3.groupby('bCROT_Contratado').mean()
#excluir as linhas sem informaçao deCROT

base3.drop(base3[base3.bCROT_Contratado == 2].index, inplace=True)
# excluindo outras variáveis que apresentaram muitos missings

base3.drop(columns=['mVolume_Habitacao_Inad','mConsignado_Outros_Saldo','bConsignado_Outros','bNch_Bolsa_Familia','bEndivid_BACEN_Principal','bFiltro_Dirigente','bCROT_Utilizado'])
# analisando Segmento

import matplotlib.pyplot as plt

y= base3['bCROT_Contratado']

x= base3['cSegmento']

bar_color = 'yellow'

plt.bar(x,y, color=bar_color)

plt.show()
# analisando Gênero

import matplotlib.pyplot as plt

y= base3['bCROT_Contratado']

x= base3['cGenero']

bar_color = 'red'

plt.bar(x,y, color=bar_color)

plt.show()
# definindo dummie para segmento

cat_vars= ['cSegmento','cGenero']

for var in cat_vars:

    cat_list='var'+'_'+var

    cat_list = pd.get_dummies(base3[var], prefix=var)

    data1=base3.join(cat_list)

    base3=data1



cat_vars= ['cSegmento','cGenero']

data_vars=base3.columns.values.tolist()

to_keep=[i for i in data_vars if i not in cat_vars]



# definindo base final

data_final = base3
# ajustando base final

base3=base3[to_keep]

data_final.columns.values
# verificando ajustes

base3.head()
# calculando tempo de relacionamento do cliente

import datetime

hoje=datetime.date.today()

hoje = pd.to_datetime(hoje)

base3['dInicio_Relacionamento'] = pd.to_datetime(base3['dInicio_Relacionamento'])

base3['TempoRelacionamento'] = hoje - base3['dInicio_Relacionamento']

base3['TempoRelacionamento'] = pd.to_numeric(base3['TempoRelacionamento'])

base3['TempoRelacionamento'] = base3['TempoRelacionamento']/365

base3.head()
# Analusando o tempo de relacionamento dos clientes

base3.TempoRelacionamento.mean()
# redefinindo base final

base3 = base3.drop(columns=['dData_Nascimento','dInicio_Relacionamento','iOcupacao','cPerfil_API','bCROT_Utilizado','cSituacao_Cadastral','dConta_Corr_Ult_Mov','ID'])

data_final=base3

data_final.head()
# aplicando funções de regressão

from sklearn import preprocessing

import matplotlib.pyplot as plt 

plt.rc("font", size=14)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import seaborn as sns

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)
base3.dtypes
# visualizando dados

sns.countplot(x='bCROT_Contratado', data=base3, palette='hls')

plt.show()
# analisando amostra

count_no_sub = len(base3[base3['bCROT_Contratado']==0])

count_sub = len(base3[base3['bCROT_Contratado']==1])

pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)

print("percentage of no subscription is", pct_of_no_sub*100)

pct_of_sub = count_sub/(count_no_sub+count_sub)

print("percentage of subscription", pct_of_sub*100)
# analisando crot

base3.groupby('bCROT_Contratado').mean()

base3.head()
# redefinindo data final

data_final = data_final.drop(columns=['mVolume_Habitacao_Inad','bEndivid_BACEN_Principal'])
# verificar nulos

data_final.isnull().sum()
# ajustando base final

data_final.fillna(0,inplace=True)

data_final.isnull().sum()
# verificando base

X = data_final.loc[:, data_final.columns != 'bCROT_Contratado']

y = data_final.loc[:, data_final.columns == 'bCROT_Contratado']

from imblearn.over_sampling import SMOTE



os = SMOTE(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

columns = X_train.columns



os_data_X,os_data_y=os.fit_sample(X_train, y_train)

os_data_X = pd.DataFrame(data=os_data_X,columns=columns )

os_data_y= pd.DataFrame(data=os_data_y,columns=['bCROT_Contratado'])

 #we can Check the numbers of our data

print("length of oversampled data is ",len(os_data_X))

print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['bCROT_Contratado']==0]))

print("Number of subscription",len(os_data_y[os_data_y['bCROT_Contratado']==1]))

print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['bCROT_Contratado']==0])/len(os_data_X))

print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['bCROT_Contratado']==1])/len(os_data_X))
# configurando modelo

data_final_vars=data_final.columns.values.tolist()

y=['bCROT_Contratado']

X=[i for i in data_final_vars if i not in y]



from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(solver='lbfgs')
# seleção de variáveis

rfe = RFE(logreg, 20)

rfe = rfe.fit(os_data_X, os_data_y.values.ravel())

print(rfe.support_)

print(rfe.ranking_)
#Colocando as variáveis selecionadas para o modelo



cols=['bNch_Credito_Salario', 'mRenda_Presumida', 'bResidencia_Propria', 'mMargem_Contribuicao', 'mMargem_Cont_Ano_Anterior', 'mVolume_Credito', 'mVolume_Habitacao', 'mEndivid_BACEN', 'mEndivid_BACEN_90d', 'bConta_Corrente', 'bConta_Corr_Mov', 'iConta_Corr_Mov_Qtde_Mes', 'mConta_Corrente_Saldo', 'bCartao_Emitido', 'bCartao_Desbloqueado', 'bCDC_Disponivel', 'bCDC_Utilizado', 'bConsignado_Utilizado', 'mConsignado_Saldo', 'bPoupanca', 'bPoupanca_Mov', 'iPoupanca_Mov_Qtde_Mes', 'bPoupanca_Integ', 'mPoupanca_Integ_Saldo', 'bFiltro_Renegociacao', 'iMargem_Contribuicao_Quantidade_Meses', 'bDebito_Automatico', 'bCartao_Ativado', 'mVolume_Credito_Inad', 'cSegmento_CL', 'cSegmento_GC', 'cSegmento_GR', 'cSegmento_GV', 'cGenero_F', 'cGenero_M', 'TempoRelacionamento'] 

X=os_data_X[cols]

y=os_data_y['bCROT_Contratado']
import statsmodels.api as sm

logit_model=sm.Logit(y,X)

result=logit_model.fit()

print(result.summary2())
from sklearn.linear_model import LogisticRegression

from sklearn import metrics



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
# Apesar do tratamento de dados, observa-se que o modelo ainda necessita de novos ajustes e exclusão de variáveis pouco significativas para o modelo.

#Existem métodos de seleção de variáveis automáticos que poderiam ser utilizados como Stepwise, Foward ou Backward.

#Ainda assim, no modelo gerado pode-se observar que as variáveis Crédito Salario, o tipo de Segmento, filtro de renegociação e possuir cartão desbloqueados podem influenciar bastante na tendencia de uma pessoa contratar ou não o CROT
# Débora R Arnaud L Formiga

# Monique Brant Rocha

# Juliano Augusto Pereira Mateus

# Roger Cristiano Brok