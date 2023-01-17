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

        

        import seaborn as sns

import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
# Aluno: André dos Santos Feitosa 







#ANÁLISE DESCRITIVA DOS CAMPEONATOS BRASILEIROS DE FUTEBOL - 2009 - 2018





#IMPORTAÇÃO DA TABELA 



df = pd.read_csv('/kaggle/input/campeonato-braileiro-20092018/tabelas/Tabela_Clubes.csv', delimiter=',')





#CONHECENDO AS VARIÁVEIS 



df.info()


# ELIMINAÇÃO DE COLUNAS VAZIAS



df = df.drop(columns=['Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16'])



df



#REVERIFICANDO AS VARIÁVEIS 



df.info()
#QUANTIDADE DE LINHAS E COLUNAS 





df.shape
#APRESENTAÇÃO DOS DADOS EM FORMATO DE TABELA - CINCO PRIMEIROS



df.head()
# QUANTIDADE DE VEZES QUE OS CLUBES PARTICIPARAM DO COMPEONATO ENTRE OS ANOS DE 2009 - 2018



qtd = df['Clubes'].value_counts()



qtd

#DEMONSTRAÇÃO GRÁFICA DA PARTICIPAÇÃO DOS CLUBES NO CAMPEONATO



%matplotlib inline

import seaborn as sns



df['Clubes'].value_counts().plot.bar()
#QUANTIDADE DE VITÓRIAS POR CLUBE - CAMPEONATO BRASILEIRO DE 2009 - 2018



qtd = df.groupby(['Clubes'])['Vitorias'].sum().reset_index().sort_values('Vitorias',ascending = False )



qtd
#DEMONSTRAÇÃO GRÁFICA DA QUANTIDADE DE VITORIAS POR CLUBE DURANTE OS 10 ANOS DE CAMPEONATO BRASILEIRO



sns.set(style="whitegrid")



f, ax = plt.subplots(figsize=(6, 15))



plt.title('QUANTIDADE DE VITÓRIAS POR CLUBE - 2009 - 2018')



sns.set_color_codes("muted")

sns.barplot(x="Vitorias", y="Clubes", data=qtd,

            label="Vitorias")
#QUANTIDADE DE VITÓRIAS POR ANO - CAMPEONATO BRASILEIRO DE 2009 - 2018



qtd1 = df.groupby(['Ano'])['Vitorias'].sum().reset_index().sort_values('Vitorias',ascending = False )



qtd1
#DEMONSTRAÇÃO GRÁFICA DA QUANTIDADE DE VITORIAS POR ANO 



plt.figure(figsize=(10,5))

plt.title('DEMONSTRAÇÃO GRÁFICA DA QUANTIDADE DE VITORIAS POR ANO')

plt.xticks(rotation=90)

sns.barplot(data = qtd1, x='Ano', y='Vitorias')



plt.show()
# ANALISANDO OS DADOS DAS VITÓRIAS - CAMPEONATO BRASILEIRO DE 2009 - 2018



qtd['Vitorias'].describe()

#QUANTIDADE DE DERROTAS POR CLUBE - CAMPEONATO BRASILEIRO DE 2009 - 2018



derrotas = df.groupby(['Clubes'])['Derrotas'].sum().reset_index().sort_values('Derrotas',ascending = False )



derrotas

sns.set(style="whitegrid")



f, ax = plt.subplots(figsize=(6, 15))



plt.title('QUANTIDADE DE DERROTAS POR CLUBE - 2009 - 2018')



sns.set_color_codes("muted")

sns.barplot(x="Derrotas", y="Clubes", data=derrotas,

            label="Derrotas")
#QUANTIDADE DE DERROTAS POR ANO - CAMPEONATO BRASILEIRO DE 2009 - 2018



qtd_derrotas = df.groupby(['Ano'])['Derrotas'].sum().reset_index().sort_values('Derrotas',ascending = False )



qtd_derrotas
#DEMONSTRAÇÃO GRÁFICA DA QUANTIDADE DE DERROTAS POR ANO 



plt.figure(figsize=(10,5))

plt.title('DEMONSTRAÇÃO GRÁFICA DA QUANTIDADE DE DERROTAS POR ANO')

plt.xticks(rotation=90)

sns.barplot(data = qtd_derrotas, x='Ano', y='Derrotas')



plt.show()
# ANALISANDO OS DADOS DAS DERROTAS - CAMPEONATO BRASILEIRO DE 2009 - 2018



derrotas['Derrotas'].describe()
#QUANTIDADE DE EMPATES POR CLUBE - CAMPEONATO BRASILEIRO DE 2009 - 2018



empates = df.groupby(['Clubes'])['Empates'].sum().reset_index().sort_values('Empates',ascending = False )



empates
sns.set(style="whitegrid")



f, ax = plt.subplots(figsize=(6, 15))



plt.title('QUANTIDADE DE EMPATES POR CLUBE - 2009 - 2018')



sns.set_color_codes("muted")

sns.barplot(x="Empates", y="Clubes", data=empates,

            label="Empates")
#QUANTIDADE DE EMPATES POR ANO - CAMPEONATO BRASILEIRO DE 2009 - 2018



qtd_empates = df.groupby(['Ano'])['Empates'].sum().reset_index().sort_values('Empates',ascending = False )



qtd_empates
#DEMONSTRAÇÃO GRÁFICA DA QUANTIDADE DE EMPATES POR ANO 



plt.figure(figsize=(10,5))

plt.title('DEMONSTRAÇÃO GRÁFICA DA QUANTIDADE DE EMPATES POR ANO')

plt.xticks(rotation=90)

sns.barplot(data = qtd_empates, x='Ano', y='Empates')



plt.show()
# ANALISANDO OS DADOS DOS EMPATES - CAMPEONATO BRASILEIRO DE 2009 - 2018



empates['Empates'].describe()


#SALDO DE GOLS POR CLUBE - CAMPEONATO BRASILEIRO DE 2009 - 2018



Saldo = df.groupby(['Clubes'])['Saldo'].sum().reset_index().sort_values('Saldo',ascending = False )



Saldo

#QUANTIDADE JOGADORES (ELENCO) POR CLUBE - CAMPEONATO BRASILEIRO DE 2009 - 2018



Qtd_Jogadores = df.groupby(['Clubes'])['Qtd_Jogadores'].sum().reset_index().sort_values('Qtd_Jogadores',ascending = False )



Qtd_Jogadores



sns.set(style="whitegrid")



f, ax = plt.subplots(figsize=(6, 15))



plt.title('SOMA DA QUANTIDADE DO ELENCO POR CLUBE - 2009 - 2018')



sns.set_color_codes("muted")

sns.barplot(x="Qtd_Jogadores", y="Clubes", data=Qtd_Jogadores,

            label="Qtd_Jogadores")
#QUANTIDADE JOGADORES POR ANO - CAMPEONATO BRASILEIRO DE 2009 - 2018



Qtd_Jogadores_ano = df.groupby(['Ano'])['Qtd_Jogadores'].sum().reset_index().sort_values('Qtd_Jogadores',ascending = False )



Qtd_Jogadores_ano
#DEMONSTRAÇÃO GRÁFICA DA QUANTIDADE DE JOGADORES POR ANO 



plt.figure(figsize=(10,5))

plt.title('DEMONSTRAÇÃO GRÁFICA DA QUANTIDADE DE JOGADORES POR ANO')

plt.xticks(rotation=90)

sns.barplot(data = Qtd_Jogadores_ano, x='Ano', y='Qtd_Jogadores')



plt.show()
# ANALISANDO OS DADOS DA QUANTIDADE DE JOGADORES



Qtd_Jogadores['Qtd_Jogadores'].describe()



#QUANTIDADE JOGADORES ESTRANGEIROS POR CLUBE - CAMPEONATO BRASILEIRO DE 2009 - 2018



Estrangeiros = df.groupby(['Clubes'])['Estrangeiros'].sum().reset_index().sort_values('Estrangeiros',ascending = False )



Estrangeiros



#DEMONSTRAÇÃO GRÁFICA DA QUANTIDADE DE JOGADORES ESTRANGEIROS POR CLUBES 



plt.figure(figsize=(10,5))

plt.title('QUANTIDADE DE JOGADORES ESTRANGEIROS POR CLUBES')

plt.xticks(rotation=90)

sns.barplot(data = Estrangeiros, x='Clubes', y='Estrangeiros')



plt.show()
#QUANTIDADE JOGADORES ESTRANGEIROS POR ANO - CAMPEONATO BRASILEIRO DE 2009 - 2018



Estrangeiros_ano = df.groupby(['Ano'])['Estrangeiros'].sum().reset_index().sort_values('Estrangeiros',ascending = False )



Estrangeiros_ano

#DEMONSTRAÇÃO GRÁFICA DA QUANTIDADE DE JOGADORES POR ANO 



plt.figure(figsize=(10,5))

plt.title('DEMONSTRAÇÃO GRÁFICA DA QUANTIDADE DE JOGADORES ESTRANGEIROS POR ANO')

plt.xticks(rotation=90)

sns.barplot(data = Estrangeiros_ano, x='Ano', y='Estrangeiros')



plt.show()
#CLUBE COM MAIOR VALOR TOTAL 





df[df['Valor_total'] == df['Valor_total'].max()]

#CLUBE COM MENOR VALOR TOTAL 





df[df['Valor_total'] == df['Valor_total'].min()]

#CLUBE COM MAIOR MEDIA DE VALOR  





df[df['Media_Valor'] == df['Media_Valor'].max()]

#CLUBE COM MENOR MEDIA DE VALOR  





df[df['Media_Valor'] == df['Media_Valor'].min()]

#LISTA DE CLASSIFICADOS EM PRIMEIRO LUGAR POR ANO



campeao = df[df['Pos.'] == 1].copy()

campeao

# QUANTIDADE DE TITULOS POR CLUBE - 2009 - 2018



campeao.groupby('Clubes')['Clubes'].count()
