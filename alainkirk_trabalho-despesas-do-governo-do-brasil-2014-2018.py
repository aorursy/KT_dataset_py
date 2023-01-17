# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Esta célula é um resumo da célula inicial, trazendo apenas os comandos

# de importação das bibliotecas e obtenção e exibição do diretório do arquivo.



import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Criando o DataFrame

desp = pd.read_csv('/kaggle/input/brazil-gov-expenses/Brazil_Gov_Exp_All - Formated.csv')
# Criando novamente os dads, informando o delimitador de campos.

desp = pd.read_csv('/kaggle/input/brazil-gov-expenses/Brazil_Gov_Exp_All - Formated.csv',

                   sep=';')
# Importando novamente o DataFrame, informando também a codificação ISO-8859-1

desp = pd.read_csv('/kaggle/input/brazil-gov-expenses/Brazil_Gov_Exp_All - Formated.csv',

                   sep=';',

                   encoding='ISO-8859-1')
# Agora, verificação do tamanho da base:

print('Tamanho da base: ',desp.shape)
# Listagem do nome das variáveis, linha a linha

for col in desp.columns: 

    print(col)
# Verificação dos tipos de dados das variáveis

desp.info()
# Listagem dos valores únicos coluna Sup_Gov_agency_Name

desp.groupby('Sup_Gov_agency_Name')['Sup_Gov_agency_Name'].count()
# Listagem dos valores únicos coluna Gov_agency_Name

desp.groupby('Gov_agency_Name')['Gov_agency_Name'].count()
# Listagem das 5 primeiras observações

desp.head(5)
# Convertendo as colunas categóricas de inteiro para string

desp['Year'] = desp['Year'].apply(str)

desp['Gov_agency_Cod'] = desp['Gov_agency_Cod'].apply(str)

desp['Sup_Gov_agency_Cod'] = desp['Sup_Gov_agency_Cod'].apply(str)



# Verificação dos tipos de dados das variáveis

desp.info()
# Substituição dos caracteres . por '' e ',' por '.'

desp['Planned'] = desp['Planned'].str.replace('.','')

desp['Planned'] = desp['Planned'].str.replace(',','.')



desp['Released'] = desp['Released'].str.replace('.','')

desp['Released'] = desp['Released'].str.replace(',','.')



desp['Paid_out'] = desp['Paid_out'].str.replace('.','')

desp['Paid_out'] = desp['Paid_out'].str.replace(',','.')



# Listagem das 5 primeiras observações

desp.head(5)
# Convertendo os campos 'object' em 'numérico'

desp['Planned'] = desp['Planned'].astype(float)

desp['Released'] = desp['Released'].astype(float)

desp['Paid_out'] = desp['Paid_out'].astype(float)



# Verificação dos tipos de dados das variáveis

desp.info()
desp.head()
# Código para exibição do formato float do pandas para 2 casas decimais

pd.set_option('display.float_format', '{:.2f}'.format)
# Exibição dos primeiros registros do dataframe.

desp.head(5)
# Despesa por ano

desp.groupby('Year')['Planned'].sum()
# Despesa por ano no gráfico:

desp.groupby('Year')['Planned'].sum().plot()
# Análise das variáveis numéricas

desp.describe()
# Exibição da despesa por ano no gráfico

desp.groupby(desp['Year']).sum().plot()
# Criação da coluna Dif_Plan_Rel

# desp['Dif_Plan_Rel'] = desp['Planned'] - desp['Released']

# desp.describe()
# Importando seaborn para gráficos

import seaborn as sns



# Formatar o gráfico

import matplotlib.pyplot as plt
# Definindo o tamanho do gráfico

plt.figure(figsize=(15,5))

sns.boxplot(desp['Year'],desp['Paid_out'])



# Inserir o label

plt.title('Despesa por ano')



# Exibindo o gráfico

plt.show()
# Eixo y é o valor pago e eixo X são os anos

sns.barplot(y='Paid_out',x='Year',data=desp)
# Definndo index

desp.set_index('Year',inplace=True)
# Definição da largura

largura = 0.25





# Configurando o tamanho do gráfico

plt.figure(figsize=(15,5))



# Definindo a posição das barras

X = np.arange(5)



# Definindo a exibição da barra

plt.bar(X + 0.00, desp['Planned'].sum(), label = 'Planejado', color='#6A5ACD', width = 0.25)

plt.bar(X + 0.25, desp['Released'].sum(), label = 'Liberado', color='#00BFFF', width = 0.25)



# Inserindo o ano

plt.xticks([r + largura for r in range(5)], ['2014','2015','2016','2017','2018'])



# Ativando a legenda

plt.legend()

plt.show()
# Gráfico de valores pagos por Solicitante

plt.figure(figsize=(15,5))

sns.stripplot(x='Sup_Gov_agency_Name',y='Planned', data=desp)

plt.xticks(rotation=90)
# Listagem distinta das colunas

desp["Sup_Gov_agency_Name"].value_counts()
# Listagem distinta das colunas

desp["Sup_Gov_agency_Name"].value_counts()
# Plotando a correlação



# Aumentando a area do grafico

f, ax = plt.subplots(figsize=(15,6))

sns.heatmap(desp.corr(), annot=True, fmt='.2f', linecolor='black',ax=ax, lw=.7)
# Ordenando o dataframe pelo valor

desp_50 = desp.sort_values(by='Paid_out',ascending = False)



plt.figure(figsize=(15,5))



# Somente os 50 primeiros

sns.barplot(x='Sup_Gov_agency_Name',y='Paid_out',data=desp_7[:50])

plt.xticks(rotation=90)                                     

plt.show()