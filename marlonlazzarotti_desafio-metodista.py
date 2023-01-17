# Desafio da Trilha de Algoritmos

# Disciplina de Algoritmos

# Ciência de Dados e Big Data



# Acadêmico: Marlon Augusto Lazzarotti



# QUESTÕES:

'''

a) apresente as informações sobre quais foram as notas dos alunos em cada questão;



b) acrescente uma coluna com a nota de cada aluno, que é a soma das notas das questões;



c) e apresente o quanto os dados estão dispersos com relação

à nota de cada aluno, facilitando, assim, a análise dos dados (considerar o desvio padrão e a variância).

'''



#importando pandas e configurando pasta do Kaggle:

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))







df = pd.read_csv('../input/desafio/desafio.csv')
df.head()
df.shape
df.info()
#Adicionando uma coluna nova com a Nota Final, somando as questões

df["Nota_Final"] = df.sum(axis=1)
df.head()
df.describe()
#Apresentando a variância das notas finais

print(df['Nota_Final'].var())
# um histograma para verificar a distribuicao da Nota Final

df['Nota_Final'].hist(bins=4)
