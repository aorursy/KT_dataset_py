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
import numpy as np

import pandas as pd

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        
### PassengerId: Número de identificação do passageiro;

### Survived: Indica se o passageiro sobreviveu ao desastre. É atribuído o valor de 0 para aqueles que não sobreviveram, e 1 para quem sobreviveu;

### Pclass: Classe na qual o passageiro viajou. É informado 1 para primeira classe; 2 para segunda; e 3 para terceira;

### Name: Nome do passageiro;

### Sex: Sexo do passageiro;

### Age: Idade do passageiro em anos;

### SibSp: Quantidade de irmãos e cônjuges a bordo ;

### Parch: Quantidade de pais e filhos a bordo;

### Ticket: Número da passagem;

### Fare: Preço da passagem;

### Cabin: Número da cabine do passageiro;

### Embarked: Indica o porto no qual o passageiro embarcou. Há apenas três valores possíveis: Cherbourg, Queenstown e Southampton, indicados pelas letras “C”, “Q” e “S”, respectivamente.

import matplotlib.pyplot as plt                                                       #importando a biblioteca gráfica

dados = pd.read_csv('/kaggle/input/titanic-machine-learning-from-disaster/train.csv') #importando o csv para um dataframe

dados.head(5)                                                                         #mostrando as 5 primeiras linhas do dataframe
dados.shape  # retorna as dimensões do conjunto de dados
dados.info() # retorna informações gerais do conjunto de dados
dados.sample(10)  # visualizando uma amostra com 10 linhas do conjunto de dados 
dados[dados['Pclass']==1].mean()  #calculo da média de todos os atributos dos passageiros da 1ª classe
dados.mean() # calculo da média de todos os atributos (colunas)
dados.median() # calculo da mediana de todos os atributos (colunas)
dados.std() # retorna o desvio padrão de todos os atributos do conjunto de dados (final e começo)
dados['Fare'].mode()  # calculo da moda do atributo "Fare" (tarifa)
dados['Age'].min()  # retorna o menor valor do atributo 'Age'

dados['Age'].max()  # retorna o menor valor do atributo 'Age'
dados.quantile(0.25) # retorna o primeiro quartil de todos os atributos númericos do dataset
dados.count() # retorna a contagem de todos os atributos do dataset
dados.isnull().sum()   #conta quantos campos nulos ou vazios tem cada atributo, neste exemplo vemos que na coluna Age temos 177 campos vazios
dados.describe()  # retorna um resumo estatístico de todos os atributos numéricos do conjunto de dados
#dados = dados.dropna()               # apaga todas as linhas que contém pelo menos um valor de atributo vazio

dados = dados.dropna(subset=['Age'])  # apaga todas as linhas que contem o campo Age nulo ou vazio (NaN)
#dados = dados.drop(['Cabin'], axis=1) # apaga a coluna 'Cabin'

dados = dados.drop(['Cabin'], axis=1, inplace=True) # apaga a coluna 'Cabin'
dados.shape     # retorna as dimensões do dataset
dados.isnull().sum()  # retorna o número de campos vazios em cada atributo
dados['Pclass'].unique()  # retorna todos os valores sem repetição da coluna 'Pclass'
dados['Age'].nunique() # retorna o número de valores diferentes da coluna age
intervalo_idade = dados['Age'].max() - dados['Age'].min()  # calculo do intervalo de idades

print(intervalo_idade)

dados['Age'].value_counts() # conta o número de vezes que cada valor aparece no atribulo Age
#media_idade = dados['Age'].mean()                    #calcula a média das idades

#dados['Age'].fillna(value=media_idade, inplace=True) #preenche todos os campos cuja idade é nula com a média

#dados.isnull().sum()                                 # retorna o número de campos vazios em cada atributo
# antes de desenha o histograma:

idade_media = dados['Age'].mean()                      #retorna a idade média

desvio_padrao = dados['Age'].std()                     #retorna o desvio padrão da idade

str_std = "Desvio Padão ="+str(round(desvio_padrao,2)) #prepara a string

str_media = "Idade Média ="+str(round(idade_media,2))  #prepara a string



plt.hist(dados['Age'],bins=8, rwidth=0.9)              #gera o histograma onde bins é o número de cestas

plt.title('Histograma da Idade dos Passageiros')

plt.xlabel('Idade')

plt.ylabel('contagem')

plt.text(50, 150, str_std)                             #plt.text(x,y,string)

plt.text(50, 200, str_media)

plt.xlim(0, 100)

plt.ylim(0, 500)

plt.show()
print(dados['Survived'].value_counts())

plt.pie(dados['Survived'].value_counts(), labels=['Não Sobreviveram', 'Sobreviveram'])

plt.show()