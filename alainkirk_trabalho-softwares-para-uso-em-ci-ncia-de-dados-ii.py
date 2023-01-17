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
# Observe que são exibidos 2 arquivos CSV com o mesmo nome.

# Vamos verificar importando-os para um dataframe diferente, final "a" e "b":



brflights_a = pd.read_csv('/kaggle/input/flights-in-brazil/BrFlights2/BrFlights2.csv')

brflights_a = pd.read_csv('/kaggle/input/flights-in-brazil/BrFlights2.csv')
# Observe que o erro de decodificação para utf-8 foi exibido.

# Nesse caso faremos a tentativa de decodificar para outro formato, ISO-8859-1:



brflights_a = pd.read_csv('/kaggle/input/flights-in-brazil/BrFlights2/BrFlights2.csv', encoding='ISO-8859-1')

brflights_b = pd.read_csv('/kaggle/input/flights-in-brazil/BrFlights2.csv', encoding='ISO-8859-1')
# Neste momento não apresentou erro.

# Vamos verificar as 5 primeiras observações de cada dataframe,

# começando com o dataframe "a":



brflights_a.head(5)
brflights_b.head(5)
# Não houve diferença nas 5 primeiras observações das bases "a" e "b".

# Ambas as bases possuem 21 colunas, conforme na descrição do arquivo.

# Ainda assim, vamos verificar o número total de observações.



print('Dataframe A: ',brflights_a.shape)

print('Dataframe B: ',brflights_b.shape)
# Listagem do nome das variáveis, linha a linha

for col in brflights_a.columns: 

    print(col)

    

# Poderia ter feito apenas

# brflights_a.columns
# Verificação dos tipos de dados das variáveis

brflights_a.info()
# Listagem das primeiras observações

brflights_a.head()
# Conversão das variáveis de Data/Hora para DateTime, ao invés de serem do tipo string:



brflights_a['Partida.Prevista'] = pd.to_datetime(brflights_a['Partida.Prevista'])

brflights_a['Partida.Real'] = pd.to_datetime(brflights_a['Partida.Real'])

brflights_a['Chegada.Prevista'] = pd.to_datetime(brflights_a['Chegada.Prevista'])

brflights_a['Chegada.Real'] = pd.to_datetime(brflights_a['Chegada.Real'])
# Verificação dos dados, se tem campos em branco, etc...

brflights_a.info()
# Listagem das primeiras observações

brflights_a.head()
# Quantidade de vôos por ano de partida:

# Para extrair um segmento da data, se faz necessário importar a biblioteca datetime

import datetime as dt
# Verificação da funcionalidade de extração do ano de uma variável DateTime:

brflights_a['Partida.Prevista'].dt.year
# Poderia ter feito assim:

print(brflights_a['Partida.Prevista'].dt.year.value_counts().sort_index())
# Agora sim, quantidade de vôos por ano de partida:

brflights_a.groupby(brflights_a['Partida.Prevista'].dt.year).count()['Voos'].plot.barh()
# Vamos estudar portanto apenas os voos do ano de 2015, importando para um novo DF;

brflights_a[brflights_a['Partida.Prevista'].dt.year == '2015']