import pandas as pd

resposta = [["idade", "Quantitativa Discreta"],["sexo","Qualitativa Nominal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
import pandas as pd

df2 = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')

df2.head()
df2.info()
df2.isnull().sum()
resposta = [["f_16","Quantitativa Discreta"],

            ["f_17","Quantitativa Discreta"],

            ["f_18_20","Quantitativa Discreta"],

            ["f_21_24","Quantitativa Discreta"],

            ["f_25_34","Quantitativa Discreta"],

            ["f_35_44","Quantitativa Discreta"],

            ["gen_feminino","Qualitativa Nominal"],

            ["gen_masculino","Qualitativa Nominal"],

            ["gen_nao_informado","Qualitativa Nominal"]] 

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
variaveis_qualitativas = df2[resposta["Variavel"]]

variaveis_qualitativas.drop(['f_16','f_17','f_18_20','f_21_24','f_25_34',"f_35_44"], axis=1, inplace=True)

variaveis_qualitativas
variaveis_qualitativas["gen_feminino"].value_counts()
variaveis_qualitativas["gen_masculino"].value_counts()
variaveis_qualitativas["gen_nao_informado"].value_counts()


import numpy as np

import matplotlib.pyplot as plt
fatias = [len(df2["gen_feminino"].value_counts()),len(df2["gen_masculino"].value_counts()),len(df2["gen_nao_informado"].value_counts())]

atividades = ['Feminino', 'Masculino', 'Não informado']



colunas  = ['r', 'b', 'g']

 

plt.title('Gêneros')

plt.pie(fatias, labels = atividades, colors = colunas, startangle = 90, shadow = True, explode = (0.1, 0, 0))

 

plt.show()
x, y =["16","17","18_20","21_24","25_34","35_44"], [len(df2["f_16"].value_counts()),len(df2["f_17"].value_counts()),len(df2["f_18_20"].value_counts()),len(df2["f_21_24"].value_counts()),len(df2["f_25_34"].value_counts()),len(df2["f_35_44"].value_counts())]



plt.title("Faixa Etária")

plt.xticks(rotation='vertical')



plt.plot(x, y)

plt.show()
x, y =["16","17","18_20","21_24","25_34","35_44"], [len(df2["f_16"].value_counts()),len(df2["f_17"].value_counts()),len(df2["f_18_20"].value_counts()),len(df2["f_21_24"].value_counts()),len(df2["f_25_34"].value_counts()),len(df2["f_35_44"].value_counts())]



plt.bar(x, y, color = ['r','g','y','b','c','m'])



plt.title('Votação por Faixa Etária')



plt.show()