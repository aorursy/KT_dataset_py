!pip install pandas
import pandas as pd 

import re 

import numpy as np 

import seaborn as sns

from matplotlib import pyplot as plt



import os

print(os.listdir("../input"))
data = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')

data.head()
regions = pd.read_csv("../input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv")

regions.head()
olympics_1 = pd.merge (data, regions, on="NOC", how="left")

olympics_1.head()
olympics_1.shape
olympics= olympics_1[(olympics_1.Season == "Summer")]

olympics.head()
olympics.isnull().any()
olympics_sex = olympics["Sex"]
valor_mundo = olympics_sex.value_counts()

valor_mundo
porcentagem_mundo = [100*x/valor_mundo.sum() for x in  valor_mundo]

porcentagem_mundo
olympics_brasil = olympics[(olympics.NOC == "BRA")]

olympics_brasil.head()
olympics_brasil_sex = olympics_brasil["Sex"]
valor_brasil = olympics_brasil_sex.value_counts()

valor_brasil
porcentagem_brasil= [100*x/valor_brasil.sum() for x in valor_brasil]

porcentagem_brasil
def pizza_mundo():

    #Data

    olympics_sex.dropna(inplace = True)

    labels =olympics_sex.value_counts().index

    colors= ["Blue", "orange"]

    explode= [0,0]

    sizes=olympics_sex.value_counts().values



    #visual

    plt.figure(figsize= (7,7))

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct="%1.2f%%")

    plt.title("Porcentagem entre homens e mulheres que participaram da olimpíada", color="black", fontsize= 20)

pizza_mundo()
def pizza_brasil():

    #Dados 

    olympics_brasil_sex.dropna(inplace = True)

    labels = olympics_brasil_sex.value_counts().index

    colors = ['Green','Yellow']

    explode = [0,0]

    sizes = olympics_brasil_sex.value_counts().values



    # visual

    plt.figure(figsize = (7,7))

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%')

    plt.title('Porcentagem entre homens e mulheres brasileiros que participaram da olimpíada',color = 'black',fontsize = 20)

pizza_brasil()
olympics_mundo_medalha = pd.DataFrame(olympics, columns = ['Medal','Sex', 'NOC'])

olympics_mundo_medalha.head()
olympic_mundo_so_medalha = olympics_mundo_medalha.dropna()
olympics_mundo_medalha_sexo = olympic_mundo_so_medalha["Sex"]
valor_mundo_medalha = olympics_mundo_medalha_sexo.value_counts()

valor_mundo_medalha
porcentagem_mundo_medalha = [100*x/valor_mundo_medalha.sum() for x in  valor_mundo_medalha]

porcentagem_mundo_medalha
olympic_medal_brasil = olympic_mundo_so_medalha[(olympic_mundo_so_medalha.NOC == "BRA")]

olympic_medal_brasil.head()
olympics_brasil_medalha_sexo = olympic_medal_brasil["Sex"]
valor_brasil_medalha = olympics_brasil_medalha_sexo.value_counts()

valor_brasil_medalha
porcentagem_brasil_medalha = [100*x/valor_brasil_medalha.sum() for x in  valor_brasil_medalha]

porcentagem_brasil_medalha
def pizza_mundo_medalha():

    #Data

    olympic_mundo_so_medalha.dropna(inplace = True)

    labels =olympic_mundo_so_medalha["Sex"].value_counts().index

    colors= ["Blue", "orange"]

    explode= [0,0]

    sizes=olympic_mundo_so_medalha["Sex"].value_counts().values



    #visual

    plt.figure(figsize= (7,7))

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct="%.2f%%")

    plt.title("Porcentagem entre homens e mulheres que ganharam uma medalha", color="black", fontsize= 20)

pizza_mundo_medalha()
def pizza_brasil_medalha():

    #Data

    olympics_brasil_medalha_sexo.dropna(inplace = True)

    labels =olympics_brasil_medalha_sexo.value_counts().index

    colors= ["Green", "Yellow"]

    explode= [0,0]

    sizes=olympics_brasil_medalha_sexo.value_counts().values



    #visual

    plt.figure(figsize= (7,7))

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct="%.2f%%")

    plt.title("Porcentagem entre homens e mulheres que participaram da olimpíada", color="black", fontsize= 20)

pizza_brasil_medalha()
olympics_medal_de_ouro_sexo = olympic_mundo_so_medalha[(olympic_mundo_so_medalha.Medal == "Gold")]

olympics_medal_de_ouro_sexo.head()
olympics_medal_de_ouro_sexo = olympics_medal_de_ouro_sexo["Sex"]
valor_da_medalhas_de_ouro_mundo = olympics_medal_de_ouro_sexo.value_counts()

valor_da_medalhas_de_ouro_mundo
porcentagem_mundo_medalha_ouro = [100*x/valor_da_medalhas_de_ouro_mundo.sum() for x in  valor_da_medalhas_de_ouro_mundo]

porcentagem_mundo_medalha_ouro
olympic_medal_brasil_ouro = olympic_medal_brasil[(olympic_medal_brasil.Medal == "Gold")]

olympic_medal_brasil_ouro_sexo = olympic_medal_brasil_ouro["Sex"]
valor_brasil_medalha_de_ouro = olympic_medal_brasil_ouro_sexo.value_counts()

valor_brasil_medalha_de_ouro
porcentagem_brasil_ouro = [100*x/valor_brasil_medalha_de_ouro.sum() for x in valor_brasil_medalha_de_ouro]

porcentagem_brasil_ouro
def pizza_mundo_ouro():  

    # Data

    olympics_medal_de_ouro_sexo.dropna(inplace = True)

    labels = olympics_medal_de_ouro_sexo.value_counts().index

    explode = [0,0]

    colors = ["Blue", "Orange"]

    sizes = olympics_medal_de_ouro_sexo.value_counts()



    # Visual

    plt.figure(figsize=(7,7))

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct="%.2f%%")

    plt.title("Porcentagem de homens e mulheres que ganharam ouro", color="Black", fontsize=20)

pizza_mundo_ouro()

def pizza_brasil_ouro():

    #Data

    olympic_medal_brasil_ouro_sexo.dropna (inplace = True)

    labels = olympic_medal_brasil_ouro_sexo.value_counts().index

    colors = ["Green", "Yellow"]

    explode = [0,0]

    sizes = olympic_medal_brasil_ouro_sexo.value_counts()



    #Visual

    plt.figure(figsize=(7,7))

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct = "%.2f%%")

    plt.title("Porcentagem de homens e mulheres brasileiros que ganharam uma medalha de ouro", fontsize=20)

pizza_brasil_ouro()