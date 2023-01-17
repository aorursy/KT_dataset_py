import pandas as pd

import requests

import matplotlib.pyplot as plt
# Requisição dos dados através da API da Câmara dos Deputados



r = requests.get('https://dadosabertos.camara.leg.br/api/v2/deputados')

r



# Caso queira ver o conteúdo da requisição, execute a linha abaixo 

#response.content
# Decodificação dos dados da resposta (que estão em JSON)



j = r.json()
# Transformação da resposta em um dataframe



df = pd.DataFrame.from_dict(j['dados'])

df
# Agrupamento das UFs e contagem de deputados federais por UF



qnt_estados = df.groupby(['siglaUf'])[['id']].count().sort_values('id', ascending=False).copy().reset_index()

qnt_estados
# Agrupamento dos partidos e contagem de deputados federais por partido



qnt_partidos = df.groupby(['siglaPartido'])[['id']].count().sort_values('id').copy().reset_index()

qnt_partidos
# Definição do estilo da imagem e dos gráficos

plt.style.use('ggplot')



# Criação da imagem inteira (fig) e dos dois gráficos (ax)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 20))



# Inserção os dados nos gráficos

ax[0].bar(qnt_estados['siglaUf'], qnt_estados['id'], color='#66c7e8')

ax[1].barh(qnt_partidos['siglaPartido'], qnt_partidos['id'], color='#b2e866')



# Inserção dos títulos

ax[0].set_title('Quantidade de Deputados Federais por UF', color='#808080', fontsize=20, pad=10)

ax[1].set_title('Quantidade de Deputados Federais por Partido', color='#808080', fontsize=20, pad=10)



# Mudança das cores de fundo dos gráficos

ax[0].set_facecolor('#f2f2f2')

ax[1].set_facecolor('#f2f2f2')



# Mostrar os gráficos :)

plt.show()