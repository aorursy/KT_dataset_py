# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Análise Exploratória de Dados -> Neste notebook usaremos uma pesquisa recente nos EUA sobre o mercado de trabalho para programadores de software. 

# Nosso objetivo é fazer uma investigação inicial dos dados a fim de detectar problemas com os dados, necessidade de mais variáveis, falhas na organização e transformação.

# Pesquisa Salarial realizada pelo site https://www.freecodecamp.com/ com programadores de software nos EUA que frequentaram treinamentos Bootcamp



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#print console para testar

# Importando os pacotes e o dataset



%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import colorsys

plt.style.use('seaborn-talk')



#df = pd.read_csv("Dados-Pesquisa-2016.csv", sep = ',', low_memory=False)
# reading the dataset



data = pd.read_csv('../input/Dados-Pesquisa-2016.csv', sep = ',', low_memory=False)



# check the shape of the data

data.shape
# Exibindo as 05 Primeiras Linhas do dataset

print(data.head())
# Exibindo as Estatísticas Descritivas do Dataset - > Pesquisa Salarial -> 

print(data.describe())
# Distribuição de Idade (Visualização por um Histograma)

# A maioria dos profissionais que trabalham como programadores de 

# software estão na faixa de idade entre 20 e 30 anos, sendo 25 anos 

# a idade mais frequente.



# Gerando um histograma

data.Age.hist(bins = 60)

plt.xlabel("Idade")

plt.ylabel("Número de Profissionais")

plt.title("Distribuição de Idade")

plt.show()
# Distribuição por Sexo - Gráfico de Pizza.

# A grande maioria dos programadores são do sexo masculino



# Definindo a quantidade

labels = data.Gender.value_counts().index

num = len(data.EmploymentField.value_counts().index)



# Criando a lista de cores

listaHSV = [(x*1.0/num, 0.5, 0.5) for x in range(num)]

listaRGB = list(map(lambda x: colorsys.hsv_to_rgb(*x), listaHSV))



# Gráfico de Pizza

fatias, texto = plt.pie(data.Gender.value_counts(), colors = listaRGB, 

                        startangle = 90)

plt.axes().set_aspect('equal', 'datalim')

plt.legend(fatias, labels, bbox_to_anchor = (1.05,1))

plt.title("Sexo")

plt.show()
# Distribuição de Pessoas por Interesse

# O principal interesse profissional dos programadores é o desenvolvimento 

# web (Full-Stack, Front-End e Back-End, seguido pela área de Data Science.



# Definindo a quantidade

num = len(data.JobRoleInterest.value_counts().index)



# Criando a lista de cores

listaHSV = [(x*1.0/num, 0.5, 0.5) for x in range(num)]

listaRGB = list(map(lambda x: colorsys.hsv_to_rgb(*x), listaHSV))

labels = data.JobRoleInterest.value_counts().index

colors = ['OliveDrab', 'Orange', 'OrangeRed', 'DarkCyan', 'Salmon', 

          'Sienna', 'Maroon', 'LightSlateGrey', 'DimGray']



# Gráfico de Pizza

fatias, texto = plt.pie(data.JobRoleInterest.value_counts(), 

                        colors = listaRGB, startangle = 90)

plt.axes().set_aspect('equal', 'datalim')

plt.legend(fatias, labels, bbox_to_anchor = (1.25, 1))

plt.title("Interesse Profissional")

plt.show()
# A maioria dos programadores trabalha na área de desenvolvimento de 

# softwares e TI, mas outras áreas como finanças e saúde também são 

# significativas.



# Definindo a quantidade

num = len(data.EmploymentField.value_counts().index)



# Criando a lista de cores

listaHSV = [(x*1.0/num, 0.5, 0.5) for x in range(num)]

listaRGB = list(map(lambda x: colorsys.hsv_to_rgb(*x), listaHSV))

labels = data.EmploymentField.value_counts().index



# Gráfico de Pizza

fatias, texto = plt.pie(data.EmploymentField.value_counts(), 

                        colors = listaRGB, startangle = 90)

plt.axes().set_aspect('equal', 'datalim')

plt.legend(fatias, labels, bbox_to_anchor = (1.3, 1))

plt.title("Área de trabalho Atual")

plt.show()
# Preferências de trabalho por idade

# Perceba que à medida que a idade aumenta, o interesse por trabalho 

# freelance também aumenta, sendo o modelo preferido por profissionais 

# acima de 60 anos. Profissionais mais jovens preferem trabalhar em 

# Startups ou no seu próprio negócio. Profissionais entre 20 e 50 anos 

# preferem trabalhar em empresas de tamanho médio.



# Agrupando os dados

df_ageranges = data.copy()

bins=[0, 20, 30, 40, 50, 60, 100]

df_ageranges['AgeRanges'] = pd.cut(df_ageranges['Age'], bins, 

                                   labels=["< 20", "20-30", "30-40", 

                                           "40-50", "50-60", "< 60"]) 

df2 = pd.crosstab(df_ageranges.AgeRanges, 

                  df_ageranges.JobPref).apply(lambda r: r/r.sum(), axis=1)



# Definindo a quantidade

num = len(df_ageranges.AgeRanges.value_counts().index)



# Criando a lista de cores

listaHSV = [(x*1.0/num, 0.5, 0.5) for x in range(num)]

listaRGB = list(map(lambda x: colorsys.hsv_to_rgb(*x), listaHSV))



# Gráfico de Barras (Stacked)

ax1 = df2.plot(kind = "bar", stacked = True, 

               color = listaRGB, title = "Preferência de Trabalho por Idade")

lines, labels = ax1.get_legend_handles_labels()

ax1.legend(lines, labels, bbox_to_anchor = (1.51, 1))
# Visualizando o help

help(pd.crosstab)
# Realocação por idade

# A vontade de buscar um novo emprego diminui com a idade. Quase 80% das 

# pessoas abaixo dos 30 anos estão preparadas para isso.



# Agrupando os dados

df3 = pd.crosstab(df_ageranges.AgeRanges, 

                  df_ageranges.JobRelocateYesNo).apply(lambda r: r/r.sum(), 

                                                       axis = 1)



# Definindo a quantidade

num = len(df_ageranges.AgeRanges.value_counts().index)



# Criando a lista de cores

listaHSV = [(x*1.0/num, 0.5, 0.5) for x in range(num)]

listaRGB = list(map(lambda x: colorsys.hsv_to_rgb(*x), listaHSV))



# Gráfico de Barras (Stacked)

ax1 = df3.plot(kind = "bar", stacked = True, 

               color = listaRGB, title = "Realocação por Idade")

lines, labels = ax1.get_legend_handles_labels()

ax1.legend(lines,["No", "Yes"], loc = 'best')
# idade x Horas de aprendizagem

# A idade dos profissionais não afeta a quantidade de tempo gasto com 

# capacitação e treinamento.

import warnings

warnings.filterwarnings('ignore')



# Criando subset dos dados

df9 = data.copy()

df9 = df9.dropna(subset=["HoursLearning"])

df9 = df9[data['Age'].isin(range(0,70))]



# Definindo os valores de x e y

x = df9.Age

y = df9.HoursLearning



# Computando os valores e gerando o gráfico

m, b = np.polyfit(x, y, 1)

plt.plot(x, y, '.')

plt.plot(x, m*x + b, '-', color = "red")

plt.xlabel("Idade")

plt.ylabel("Horas de Treinamento")

plt.title("Idade por Horas de Treinamento")

plt.show()
# Investimentos em Capacitação x Ganhos Salarias..

# Os profissionais que investem tempo e dinheiro em capacitação e 

# treinamento, em geral, conseguem salários mais altos, embora alguns 

# profisisonais esperem altos salários, investindo 0 em treinamento.

import warnings

warnings.filterwarnings('ignore')



# Criando subset dos dados

df5 = data.copy()

df5 = df5.dropna(subset=["ExpectedEarning"])

df5 = df5[data['MoneyForLearning'].isin(range(0,60000))]



# Definindo os valores de x e y

x = df5.MoneyForLearning

y = df5.ExpectedEarning



# Computando os valores e gerando o gráfico

m, b = np.polyfit(x, y, 1)

plt.plot(x, y, '.')

plt.plot(x, m*x + b, '-', color = "red")

plt.xlabel("Investimento em Treinamento")

plt.ylabel("Expectativa Salarial")

plt.title("Investimento em Treinamento vs Expectativa Salarial")

plt.show()