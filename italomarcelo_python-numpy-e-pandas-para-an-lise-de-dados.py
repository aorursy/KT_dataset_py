import numpy as np
# Para criar um array com um numpy é muito simples:

meuarray = np.array([0, 1, 2, 3])

print(meuarray)
# Quer somar os valores dentro deste array, 0 + 1 + 2 + 3 = 6, 

# então aí vai:

print(meuarray.sum())
# Quer encontrar o máximo ou o mínimo, não mudou nada..

print(f'Max: {meuarray.max()}, Min: {meuarray.min()}')

# Já sei, você quer saber a média dos valores

# Muito difícil..

print(meuarray.mean())
# Vamos criar um array igual ao anterior, [0, 1, 2, 3]

meuarray2 = np.arange(4)

print(meuarray2)
# Não, 

# eu quero criar um array, com 4 posições mas somente com o valor 1

print(np.ones(4))
# Ou com o valor 0

print(np.zeros(4))
# Ou com valores randômicos (aleatórios)

print(np.random.random(4))
# mas Italo, a demanda exige números randômicos porém, inteiros

print(np.random.randint(4))

# ooops..  ele retornou somente um número, randômico, de 0 a 3.. 

# mas a nossa demanda exige que seja um array com 4 posições e 

# com números inteiro. Complicou? Acho q nao… 

# Com numpy, utilizaremos a função randint(início, fim, qtde)

print(np.random.randint(0, 10, 4))

# E somente para concluir esses comandos bacanas, lembra do arange

print(np.arange(4))

# Então, outro comando numpy muito utilizado é o linspace. 

print(np.linspace(0, 3, 4))

# Ela é bem semelhante a função linspace, retorna valores espaçados 

# em um intervalo, mas de forma uniforme. Entao porque nao usar a 

# arange? Bem, a linspace é muito utilizada quando estamos plotando 

# dados, tanto na visualização dos valores ou quando estamos montando 

# os eixos desta plotagem

print(np.linspace(0, 1, 4))
# vamos importar a biblioteca pandas no python

import pandas as pd

# e criar a Série semana, eu faço simples assim:

semana = pd.Series(['Domingo', 'Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado'])



print(semana)
# ou eu poderia criar assim

dados = ['Domingo', 'Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado']

semana = pd.Series(dados)



print(semana)
# algumas formas de imprimir os dados ou atributos do df

# se quiser imprimir somente os valores

print(semana.values)



# se quiser imprimir somente os índices

print(semana.index)



# se quiser… aff, já está repetitivo… mas se quiser imprimir o dia Segunda, 

print(semana.iloc[1])



# e lembre-se, a Série é um array, então eu poderia imprimir assim também

print(semana[1])

# Agora criaremos um DataFrame

df = pd.DataFrame()

df['coluna 1'] = [1, 2, 3, 4]

df['coluna 2'] = ['a','b', 'c','d']



# vamos exibir nosso dataframe

print(df)
# veja, temos índices que vão de 0 a 3, mas também podemos nomear os índices. Como? Assim por exemplo:

df.index = ['linha0', 'linha1', 'linha2', 'linha3']

print(df)
# uma forma muito comum seria criar esse dataframe utilizando um dicionario 'dict'

data = {

	'coluna 1':  [1, 2, 3, 4],

	'coluna 2':  ['a','b', 'c','d']

}

indice = ['linha0', 'linha1', 'linha2', 'linha3']

df = pd.DataFrame(data=data, index=indice)



print(df)
# importando dados  de um arquivo csv

dfCSV = pd.read_csv('/kaggle/input/minidatasetsexemplos/meuArquivo.csv')

print(dfCSV)



# importando dados  de um arquivo json

dfJSON = pd.read_json('/kaggle/input/minidatasetsexemplos/meuArquivo.json')

print(dfJSON)



# recebendo dados de um arquivo excel

dfXLSX = pd.read_excel('/kaggle/input/minidatasetsexemplos/meuArquivo.xlsx')

print(dfXLSX)

# Vamos voltar ao nosso simples df

print(df, '\n\n')

# e explorar um pouco mais os nossos dados

# se eu quero imprimir a coluna 1, 

print(df['coluna 1'])
# se eu quero imprimir a coluna 2, 

print(df['coluna 2'])
# se eu quero a linha 1

print(df.loc['linha1'])
# mas vcs lembram que antes de renomear nossos índices, 

# eles tinham a sua sequência numérica? 

# Entao, nesse caso, é só usar o iloc

print(df.iloc[1])
# e para imprimir somente o valor da [coluna 1]x[linha1].. então

print(df['coluna 1']['linha1'])

# vamos importar o meu dataset meuArquivo.csv

df = pd.read_csv('/kaggle/input/minidatasetsexemplos/meuArquivo.csv')
# ou eu poderia fazer uma cópia do dfCSV ao qual já importamos

df = dfCSV.copy()
# vamos visualizar os 5 primeiros registros de nosso dataset

print(df.head())
# ou os 5 últimos registros

print(df.tail())
# Italo, pode aumentar para 10 registros? Sim, somente add o número que deseja visualizar, no head ou no tail

print(df.head(10))
# E se você quiser imprimir ou pegar aleatóriamente algumas linhas

# do seu dataset, a função sample irá lhe ajudar

# e facilitar bastante... 

# Então se eu quero 8 linhas aleatórias do dataset df, basta isso.



df.sample(8)

# execute mais de uma vez pra você notar que realmente os valores mudam
# Temos também a função info que retorna informações básicas sobre a estrutura do dataframe e dos dados existentes

print(df.info())
# outra função bastante utilizada é a shape, que retorna somente o número de colunas e linhas do dataset

print(df.shape)
# Continuando a análise, uma função bastante utilizada é a função isnull

df.isnull()
df.isnull().sum()
# e se houver valores nulos? Bem, aí vem a análise.

# somente pra efetuar um teste, vamos criar uma cópia do df

teste = df.copy()
# vamos exibir o shape

teste.shape
# agora, vou utilizar um recurso de adicionar dados no dataframe

# e no dataset teste, vou adicionar novamente todos os dados do dataset df, 

# ou seja vou duplicar valores

teste  = teste.append(df)
teste.shape
# agora sim, mais uma rica função: remover dados duplicados

teste = teste.drop_duplicates()
teste.shape
# assim, podemos fazer a mesma coisa para dados nulos

# vamos rever essa informação

teste.isnull().sum()
# veja, não existe nenhuma coluna com valor nulo…

# então vamos gerar um valor, somente pra análise

teste.head()
# vamos exibir os 5 primeiros produtos no dataframe teste

teste['PRODUTO'].head()
# vou alterar o valor do primeiro produto para nulo (None)

teste['PRODUTO'][0] = None

teste['PRODUTO'].head()
# ulalaaa, o nosso primeiro produto está sem valor, então vamos analisar a info do nosso dataset teste

teste.isnull().sum()
# e agora vemos que a variável (ou coluna) PRODUTO possui um valor nulo

# vamos imprimir o shape deste dataset

print(teste.shape)
teste.dropna()

print(teste.shape) 
teste = teste.dropna()
# vamos ver o sumário de registros nulos

print(teste.isnull().sum())

# e vermos o shape

print(teste.shape)

# outra dica, imaginando que eu queira ter uma variável, todos os produtos que foram vendidos



produtos = teste['PRODUTO']

print(produtos)
produtos = teste['PRODUTO'].unique()

print(produtos)
# somente pra curiosidade: qual é o tipo da variável produtos?

# ?

# ?

type(produtos)

# array numpy…. que mundo maravilhoso.. 
# do dataset inteiro

print(teste.describe())
# ou somente de uma variável, seja ela numérica

print(teste['QTDE'].describe())

# ou categórica

print(teste['LOJA'].describe())
#somar os valores

print(teste['QTDE'].sum())
# contar a quantidade de registros

print(teste['QTDE'].count())
# retornar o valor mínimo ou o máximo

print(teste['QTDE'].min())

print(teste['QTDE'].max())
# quer a média 

print(teste.mean())
# ou a mediana

print(teste.median())
# quer saber quais são 10 os produtos mais vendidos, 

# utilizaremos a função value_counts combinada com a função head. 

# É isso aí amigão..

print(teste['PRODUTO'].value_counts().head(10))
# se eu quero fazer um slice, ou seja um fatiamento, no dataset

# eu simplesmente atribuo as colunas desejadas (variáveis) 



loja_vendedores = teste[['LOJA', 'VENDEDOR']]
# vamos ver o resultado disso

print(loja_vendedores)
# mas eu quero somente as lojas CHAMADAS GFFF

condicao = (loja_vendedores['LOJA'] == 'GFFF')

print(condicao)



# Ou seja, ele retornou a variável (coluna) LOJA, e onde o valor é igual GFFF, 

# ele retorna True, senão false

# Para exibir somente as lojas GFFF, vamos utilizarusar essa técnica

print(loja_vendedores[condicao])
# criamos a segunda condição

condicao2 = (loja_vendedores['VENDEDOR'] == 'DEEDE')
# e agora iremos exibir os dados com as duas condições

# eu quero a condição E a condicao2

print(loja_vendedores[condicao & condicao2])
# Quero somente imprimir um dataframe com todos os produtos ´A´

teste.query('PRODUTO == "A"')
# Mas somente os produtos vendidos na loja GGFF

teste.query('PRODUTO == "A" and LOJA == "GGFG"')
# exemplo, imprimir todas as vendas da categoria de produto BB

teste.query('`CATEGORIA PRODUTO` == "BB"')
# Por exemplo, eu quero saber por categoria do produto, 

# quantos produtos e qual o montante em valor 

# Se eu somente colocar as 3 variáveis envolvidas nesta proposta

# veja o resultado:

teste[['CATEGORIA PRODUTO', 'QTDE','VALOR']].sum()
# É aí que o groupby cai como uma luva

# primeiro, eu agrupei pela variável a ser agrupada, CATEGORIA PRODUTO

# segundo, coloquei as minhas variáveis target, ou seja, que terei o resultado

# terceiro, a operação

teste.groupby('CATEGORIA PRODUTO')[['QTDE','VALOR']].sum()
# mas desta lista, eu só quero o top 5 e os 3 piores

# como o Pandas é o amigão

print('5 Melhores') 

teste.groupby('CATEGORIA PRODUTO')[['QTDE','VALOR']].sum().head()
print('3 Piores')

teste.groupby('CATEGORIA PRODUTO')[['QTDE','VALOR']].sum().tail(3)
teste.corr()
# Primeiro, 

# vamos relembrar nossas variaveis no dataset teste

teste.columns
# ou

teste.info()
# Objetivo: exibir mês a mês o valor total de vendas

# como eu quero o total mes a mes, entao terei que agrupar

totalVendas = teste.groupby('MES')['TOTAL'].sum()

print(totalVendas)
# Obs, não é importar uma variável receber o valor

# poderiamos plotar diretamente, como na linha comentada abaixo

# teste.groupby('MES')['TOTAL'].sum()

# faça o teste..



# Mas vou exibir o grafico através da variável totalVendas. 

totalVendas.plot()
# mas e se quisermos em barras horizontais

totalVendas.plot(kind='barh', title='Gráfico em Barras')
# ou somente barras verticais

totalVendas.plot(kind='bar', title='Gráfico em Colunas')
# ou em pizza

totalVendas.plot(kind='pie', title='Gráfico em Pizza')
# ou boxplot

totalVendas.plot(kind='box', title='Gráfico Boxplot')
# e por último, vamos fazer um gráfico que mostre mes a mes

# como se comportou as vendas no TOTAL, e com possíveis vendas sem desconto

totalXbruto = teste.groupby('MES')[['QTDE','VALOR','DESCONTO','TOTAL']].sum()

print(totalXbruto)
# para entendermos, a variável total, traz o resultado

# qtde * valor - o desconto aplicado

# entao para termos o bruto, poderiamos usar somente o qtde * valor

# ou add o desconto em total

# vamos pelo mais fácil, vou criar uma coluna no nosso dataset e fazer o cálculo



totalXbruto['VALOR BRUTO'] = (totalXbruto['QTDE'] * totalXbruto['VALOR'])

print(totalXbruto)
# pronto, nossos dados estão preparados, agora é somente exibir

totalXbruto.plot(kind='line', title='Gráfico em Linhas', y=['TOTAL', 'VALOR BRUTO'])