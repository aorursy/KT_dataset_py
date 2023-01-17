import pandas as pd

import numpy as np
#Classifique todas as variáveis escolhidas, e construa um dataframe com sua resposta.

resposta = [["uf","Qualitativa Nominal"],["total_eleitores","Quantitativa Discreta"],["gen_feminino","Quantitativa Discreta"],["gen_masculino","Quantitativa Discreta"],["f_sup_79","Quantitativa Discreta"],["f_16","Quantitativa Discreta"],

            ["f_45_59","Quantitativa Discreta"],["f_25_34","Quantitativa Discreta"]]

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
# lendo o arquivo csv

df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')

# visualizando alguns dados do dataset

df.head(10)
# Analisando dados estatisticos do dataset

df.describe()
# Analisando os tipos das colunas

df.info()
# verificando se existe campos com nulos

df.isnull().sum()
# DEixando o dataset somente com as colunas desejadas

df[resposta["Variavel"]] 
# Tabela de frequência da variavel qualitativa

frequencia =  df['uf'].value_counts()

frequencia

#Item C - Representação Gráfica

%matplotlib inline

import matplotlib.pyplot as plt
#Totalizando a quantidade de eleitores femininas

total_fem = df['gen_feminino'].sum()

#Totalizando a qiuantidade de eleitores masculinos

total_masc = df['gen_masculino'].sum()

# Gerando grafico de pizza

fatias = [total_fem,total_masc]

labels = ['Eleitoras Femininas', 'Eleitores Masculinos']

cores  = ['m', 'b']



plt.title('Eleitores por Gêneros')

 

plt.pie(fatias, labels = labels, autopct='%1.1f%%' ,colors = cores, shadow = False)

 

plt.show()
# Criando dicionario para receber os dados dos eleitores por estados

dic_estados = {}

# Juntando os eleitores por estados

dic_estados['uf'] = df.groupby(by=['uf'])['total_eleitores'].sum()

# Criando e exibindo o grafico de barras

for i, v in dic_estados['uf'].items():

    plt.bar(i, v)

    plt.text(i, v, v, va='bottom', ha='center')

    

plt.title('Quantidade de Eleitores por Estados ')

plt.xlabel('Estados')

plt.style.use('seaborn')

plt.gca().axes.get_yaxis().set_visible(False)

plt.show()
# Grafico com as variaveis Quantitativa Discreta

x, y =["16","35 - 44","45 - 59",">79"], [df['f_16'].sum(), df['f_25_34'].sum(), df['f_45_59'].sum(),df['f_sup_79'].sum()]

# Criando um gráfico

plt.title("Quantidade de Eleitores por Idade")

plt.xticks(rotation='vertical')

plt.xlabel('Idades')

plt.ylabel('Quantidade (Milhões)')



plt.plot(x, y)

plt.show()