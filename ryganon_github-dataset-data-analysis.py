import numpy as np # álgebra linear

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # biblioteca para gráficos/plots

import matplotlib.pyplot as plt # plots 



from google.cloud import bigquery # google big query / python

from bq_helper import BigQueryHelper



# opções para o notebook

%matplotlib inline

plt.style.use('ggplot')
# criando um assistente para as queries que serão executadas.

# os dois parâmetros indicam o nome do projeto eo nome da base, respectivamente.

%time

bq_assistant = BigQueryHelper("bigquery-public-data", "github_repos")
QUERY_LIC = """

        SELECT license, COUNT(*) AS count

        FROM `bigquery-public-data.github_repos.licenses`

        GROUP BY license

        ORDER BY COUNT(*) DESC

        """
# antes de executar, vamos verificar os custos da query

# resultado: aproximadamente 20mb

%time

bq_assistant.estimate_query_size(QUERY_LIC)
# executando a query e alocando o resultado em um

# dataframe do Pandas

%time

df_lic = bq_assistant.query_to_pandas_safe(QUERY_LIC)
# verificando o tamanho do objeto dataframe que acabou de ser criado

print('Tamanho do dataframe: {} Bytes'.format(int(df_lic.memory_usage(index=True, deep=True).sum())))
# verificando quantos registros existem no dataframe

print("Total de {} (linhas x colunas)".format(df_lic.shape))
# exibindo tudo na tela

print(df_lic)
# criando um gráfico para visualizar a representação das licenças



# configurando o tamanho da figura (12x9)

f, g = plt.subplots(figsize=(10, 7))

# configurando os eixos x e y do gráfico; adicionando os dados (df)

g = sns.barplot(x="license", y="count", data=df_lic, palette="Blues_d")

# ajustando os rótulos dos eixos

g.set_xlabel("Licenças")

g.set_ylabel("Total")

# ajustando a posição dos rótulos do eixo x

g.set_xticklabels(g.get_xticklabels(), rotation=30)

# finalizando

plt.title("Distribuição das licenças nos repositórios do Github")

plt.show(g)