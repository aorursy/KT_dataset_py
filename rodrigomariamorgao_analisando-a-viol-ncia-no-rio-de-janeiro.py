#importar os pacotes necessários

import pandas as pd
# local de hospedagem do arquivo

DATA_PATH = "../input/violencia_rio.csv"



#importar o arquivo violencia_rio.csv para um dataframe

df = pd.read_csv(DATA_PATH)



# listar as 5 primeiras entradas do dataset

df.head()
# identificar o volume de dados do DataFrame

print("Variáveis:\t {}\n".format(df.shape[1]))

print("Entradas:\t {}".format(df.shape[0]))
# verificar os tipos de entradas do dataset

display(df.dtypes)
# verificar a porcentagem dos dados faltantes, ordenados pela ordem decrescente

nullseries = (df.isnull().sum() / df.shape[0]).sort_values(ascending=False,)

print(nullseries[nullseries > 0])
# imprimir o resumo do dataframe

df.describe()
# informa a média dos roubos dos veículos

roubo_veiculos = df.roubo_veiculo.mean()

print(roubo_veiculos)
# informa a média dos furtos dos veículos

furto_veiculos = df.furto_veiculos.mean()

print(furto_veiculos)
# informa a média de recuperação dos veículos

recuperacao_veiculos = df.recuperacao_veiculos.mean()

print(recuperacao_veiculos)
# porcentagem da média dos carros recuperados

recuperacao_veiculos/(roubo_veiculos+furto_veiculos) * 100
# informa o valor mínimo de homicídios dolosos

df.hom_doloso.min()
# informa o valor máximo de homicídios dolosos

df.hom_doloso.max()
# plotar um histograma dos homicídios dolosos

df.hom_doloso.plot(kind="hist",figsize=(5,5), title="histograma da categoria homicídio doloso");
# plotar um gráfico de linhas dos roubos em coletivos

df.roubo_em_coletivo.plot(title="roubos em coletivos",figsize=(8,5));