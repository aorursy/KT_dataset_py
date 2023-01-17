

import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt



df = pd.read_csv("../input/results.csv") #Leio o arquivo com os resultados

df.head() #Visualizacao inicial
df["total_goals"] = df["home_score"] + df["away_score"] #Crio uma feature com o total de gols na partida

df = df[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'total_goals', 'tournament', 'city', 'country', 'neutral' ]] #ajusto as colunas do meu dataset

df.head() #Resultado
df1 = df.groupby("away_team").size() + df.groupby("home_team").size() #Quero ver o quanto os paises jogaram

df1
df1 = df1.dropna() #Tiro os paises sem algum jogo ou com numero desconhecidos

df1 = df1.sort_values(ascending=False) #Ordeno os paises decrescente

df1
df1 = df1.iloc[:10] #Seleciono os 10 primeiros paises da lista

df1
ax = df1.plot(kind='bar',title="Top 10 Numero de Jogos Internacionais",ylim=[800,1000],color='red') 

ax.set(xlabel="Paises", ylabel="Numero de Jogos") #Plot do Grafico de barras e funcao auxiliar pra colocar os labels
df = df.set_index("date") #Trocando o indice

df.drop(["city","country","neutral","tournament"], axis=1)  #Retirando colunas do dataframe
dec50 = df.loc["1950-01-01":"1959-12-31"] #Separo os datasets por decadas

dec60 = df.loc["1960-01-01":"1969-12-31"]

dec70 = df.loc["1970-01-01":"1979-12-31"]

dec80 = df.loc["1980-01-01":"1989-12-31"]

dec90 = df.loc["1990-01-01":"1999-12-31"]

dec00 = df.loc["2000-01-01":"2009-12-31"]

dec10 = df.loc["2010-01-01":"2019-12-31"]

media50 = dec50.total_goals.mean() #armazeno a media de gols de cada media em variaveis para imprimir depois

media60 = dec60.total_goals.mean() 

media70 = dec70.total_goals.mean() 

media80 = dec80.total_goals.mean() 

media90 = dec90.total_goals.mean()

media00 = dec00.total_goals.mean() 

media10 = dec10.total_goals.mean()

aumento = media10/media50 * 100

print("SEGUNDO PASSO: MEDIA DE GOLS AO LONGO DAS DECADAS \n50: %f \n60: %f \n70: %f \n80: %f \n90: %f \n00: %f \n10: %f\n" % (media50,media60,media70,media80,media90,media00,media10))

print("A media de gols diminuiu %f por cento desde 1950 ate hoje"% (aumento))
Medias = [media50, media60, media70, media80, media90, media00, media10] #Armazeno as variaveis criadas em um vetor

df_Medias = pd.DataFrame(Medias) #Crio um dataframe a partir do vetor

df_Medias.columns = ["Media de Gols"] #renomeio coluna

df_Medias.rename(index = {0: "Anos 50",  #renomeando linhas

                     1:"Anos 60",

                         2:"Anos 70",

                         3:"Anos 80",

                         4:"Anos 90",

                         5:"Anos 00",

                         6:"Anos 10"}, 

                                 inplace = True)

df_Medias
ax2 = df_Medias.plot(kind='bar',title="Media de Gols ao Longo das Decadas",color='blue') 

ax2.set(xlabel="Decadas", ylabel="Media de Gols") #Plot do Grafico de barras e funcao auxiliar pra colocar os labels