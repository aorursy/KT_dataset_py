import os

import pandas as pd



filepath = []



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        filepath.append(os.path.join(dirname, filename))



# o item 0 deve ser o caminho para o dataset /kaggle/input/dataviz-facens-20182-aula-1-exerccio-2/BR_eleitorado_2016_municipio.csv

df = pd.read_csv(filepath[0])

print(df.info())

display(df.head(5))



import pandas as pd

resposta = [["total_eleitores", "Quantitativa Discreta"],

            ["uf","Qualitativa Nominal"],

            ["gen_feminino","Quantitativa Discreta"],

            ["gen_masculino","Quantitativa Discreta"],

            ["f_16","Quantitativa Discreta"],

            ["f_17","Quantitativa Discreta"],

            ["f_18_20","Quantitativa Discreta"],

            ["f_21_24","Quantitativa Discreta"],

            ["f_25_34","Quantitativa Discreta"],

            ["f_35_44","Quantitativa Discreta"],

            ["f_45_59","Quantitativa Discreta"],

            ["f_60_69","Quantitativa Discreta"],

            ["f_70_79","Quantitativa Discreta"],

            ["f_sup_79","Quantitativa Discreta"],

           ] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
tb = pd.Series(df[resposta['Variavel'][resposta['Classificação'] == 'Qualitativa Nominal']].values.ravel()).value_counts().to_frame(name="Uf_Freq")





tb.head(5) # exibindo somente 5 linhas, para exibir tudo, retirar o head(5)

# tratamento dos dados para facilitar a montagem dos gráficos

# removo as cidades e agrupo por estado para poder analisar o dataframe por estado

df_estado = df.drop(["nome_municipio","cod_municipio_tse"],axis=1)

df_estado = df_estado.groupby(["uf"]).sum()



df_estado = df_estado.reset_index()
import matplotlib.pyplot as plt



plt.figure(figsize=(10,10))



# Create horizontal bars

plt.barh(df_estado['uf'],df_estado['total_eleitores'], color=['green', 'yellow'])

plt.title('Eleitores por estado')



# Create names on the y-axis

plt.yticks(df_estado['uf'])

 

# Show graphic

plt.show()
import numpy as np



faixas = ['f_16','f_17','f_18_20','f_21_24','f_25_34','f_35_44','f_45_59','f_60_69','f_70_79','f_sup_79']

totais = df_estado[faixas]



totais = np.array([df_estado[i].sum() for i in faixas]) # realizo a soma de todos os valores das colunas

totais



faixas = ['16','17','18-20','21-24','25-34','35-44','45-59','60-69','70-79','79+']



plt.figure(figsize=(10,10))



plt.bar(faixas, totais)

plt.title('Eleitores por faixa etária')



plt.show()
generos = ['gen_feminino','gen_masculino','gen_nao_informado']

totais = df_estado[generos]



totais = np.array([df_estado[i].sum() for i in generos]) # realizo a soma de todos os valores das colunas





generos = ['Feminino','Masculino','Não informado']

explode = (0.2, 0.2, 0.3)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig1, ax1 = plt.subplots()



ax1.pie(totais, explode=explode, labels=generos, autopct='%1.1f%%',shadow=True, startangle=-90)



ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.title('Eleitores por gênero')



plt.show()
df = pd.read_csv('../input/anv.csv', delimiter=',')

df.head(1)