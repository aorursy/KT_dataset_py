# Importando bibliotecas 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Abertura do arquivo csv (precos de veiculos - posição de Junho de 2018) 

arq_veiculo = pd.read_csv('/kaggle/input/brazilian-vehicle-prices-june-2018-fipe/fipe_Jun2018.csv', sep= ',')

# Apresentação dos primeiros dados

arq_veiculo.head(5)
# (1) Mudança na nomeclatura dos campos 

# Mudança nos nomes dos campos: <unnamed> <brand> <vehicle> <year_model> <fuel>        <price_reference>  <price> 

#                                <Codigo> <Marca> <Veiculo> <Ano_Modelo> <Combustivel> <Mes_Referencia>   <Preco> 

arq_veiculo = arq_veiculo.rename (columns= {'Unnamed: 0': 'Codigo', 'brand': 'Marca', 'vehicle': 'Veiculo', 'year_model': 'Ano_Modelo', 'fuel': 'Combustivel', 'price_reference': 'Mes_Referencia', 'price': 'Preco'})

# Lista a estrutura dos campos

arq_veiculo.info()

# O arquivo tem 7 colunas e 21.797 linhas preenchidas
# (2) Ajustes no conteúdo dos dados inconsistentes 

# Ajuste no conteúdo do campo "Preco" 

for i in range(0, 21797):

    arq_veiculo['Preco'][i] = arq_veiculo['Preco'][i].replace("R$", "").lstrip()  

    arq_veiculo['Preco'][i] = arq_veiculo['Preco'][i].replace(".", "")

    arq_veiculo['Preco'][i] = arq_veiculo['Preco'][i].replace(",", ".") 

# Conversão do campo "Preco" para o tipo de dado numérico 

arq_veiculo['Preco'] = arq_veiculo['Preco'].astype('float') 

# Eliminação dos registros que possuem ano do modelo incorreto (preenchimento = 32000) 

arq_veiculo = arq_veiculo[arq_veiculo['Ano_Modelo'] != 32000].copy() 

# Apresentação da estrutura do arquivo após ajustes realizados (Restou 20.902 registros)

arq_veiculo.info()
# Apresentação dos últimos registros 

arq_veiculo.tail(3) 
# (3) Seleção dos modelos mais recentes (2018: 760 veículos e 2019: 160 veículos)

arq_veiculo_atual = arq_veiculo[arq_veiculo['Ano_Modelo'] > 2017].copy() 

# Contagem dos véiculos de ultimos modelos

arq_veiculo_atual['Ano_Modelo'].value_counts()
# Mostra cinco registros selecionados aleatoriamente dos veiculos de modelos mais recentes

arq_veiculo_atual.sample(5)
# (4) Mostra o modelo mais barato

arq_veiculo_atual[arq_veiculo_atual['Preco'] == arq_veiculo_atual['Preco'].min()] 
# (5) Mostra o modelo mais caro

arq_veiculo_atual[arq_veiculo_atual['Preco'] == arq_veiculo_atual['Preco'].max()]
# (6) Contagem de veículos por Marca

arq_veiculo_atual['Marca'].value_counts().plot.bar()

plt.xlabel('Contagem de modelos por marca')
# (7) Média de preço por marca

arq_veiculo_atual.groupby('Marca')['Preco'].mean().plot(kind='bar',legend='Reverse')

plt.xlabel('Média de preço por marca', )
# (8) Contagem dos modelos por tipo de combustível 

arq_veiculo_atual['Combustivel'].value_counts().plot(kind='pie',cmap='Paired')

plt.xlabel('Contagem dos modelos por tipo de combustível', )