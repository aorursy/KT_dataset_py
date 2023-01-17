import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
tb_corona_por_UF = 'AC 11 11  AL 7 7  AP 1 1  AM 26 26  BA 55 49  CE 125 112  DF 131 117  ES 26 26  GO 21 21  MA 2 2  MT 2 2  MS 21 21  MG 83 83  PA 4 4  PB 1 1  PR 54 50  PE 37 37  PI 6 4  RJ 186 186  RN 9 9  RS 77 72  RO 3 3  RR 2 2  SC 68 57  SP 631 631  SE 10 10  TO 5 2'

tb_corona_por_UF
lst_casos_por_estado = tb_corona_por_UF.split()

lst_casos_por_estado
estados = []

secretarias = []

ministerio = []

for i in range(0, len(lst_casos_por_estado), 3):

    estados.append(lst_casos_por_estado[i])

    secretarias.append(int(lst_casos_por_estado[i+1]))

    ministerio.append(int(lst_casos_por_estado[i+2]))

print(estados)

print(secretarias)

print(ministerio)
#transformar os dados do covid em dataframe

df_casos_corona_por_UF = pd.DataFrame({

    "Estado": estados,

    "Secretarias": secretarias,

    "Ministerio": ministerio

})

df_casos_corona_por_UF
#soma dos casos confirmados pelas secretarias

df = df_casos_corona_por_UF 

df["Secretarias"].sum()
#soma dos casos confirmados pelo ministerio

df["Ministerio"].sum()
#nova coluna que indica o tamanho das strings da coluna "Secretarias" (as strings da coluna "Ministerio" possuem o mesmo tamanho)

df = df.astype({"Secretarias": str})

df["tam"] = df["Secretarias"].str.len()

df
#ordena o dataframe de acordo com os numeros da coluna "tam" em ordem decrescente. No topo estão os estados com mais pessoas contaminadas

df.sort_values(by = ["tam"], ascending = False)
#média de casos confirmados pelas secretarias por estado

df = df.astype({"Secretarias": int})

df.Secretarias.mean()
#média de casos confirmados pelo Ministério por estado

df.Ministerio.mean()
#média dos números da coluna "tam". Indica a média do número de digitos de contaminados por estado

df.tam.mean()
#divide a coluna "Secretarias" em grupos de acordo com o tamanho de cada número presente na coluna. Em seguida faz a soma de contaminados por cada grupo

df.groupby("tam").Secretarias.sum()
#calcula a média de contaminados de cada grupo

df.groupby("tam").Secretarias.mean()
#calcula o número de vezes que cada grupo aparece no dataframe

df.groupby("tam").tam.count()
#calcula a frequência de aparições de cada grupo no dataframe 

df.groupby("tam").Secretarias.count().divide(len(df))
#dados da frequência de aparições de cada grupo no dataframe mostrado através de um gráfico de barras

df_graf = df.groupby("tam").Secretarias.count().divide(len(df))

df_graf.plot(kind = "bar", figsize =(5,3))

plt.show()
#número de casos em cada estado mostrado através de um gráfico de barras

df.plot(kind = "bar", x = "Estado", y = "Secretarias")

plt.show()