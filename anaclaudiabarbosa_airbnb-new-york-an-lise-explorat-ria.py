# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns# for data viz.

import geopandas



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Lê o arquivo AB_NYC_2019.csv e mostra

dados_airbnb=pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

dados_airbnb.head()
#Verificando o tipo de dados que a base utiliza

dados_airbnb.dtypes
#Avaliando se tem dados o suficiente para analise

dados_airbnb.isnull().sum()



#São poucas linhas sem name e host_name, não prejudica a análise

#O numero alto de null nos campos last_review e review_per_month, só mostra que essas residencias 

#não tiveram avaliação,então não prejudicam a análise
#Análise para descobrir quais regiões tem mais opções para locação

dados_airbnb.groupby(['neighbourhood_group','neighbourhood']).size().to_frame('size').reset_index().sort_values(['size'], ascending=[False])



#Distrito: Brookliyn e Bairro: Williamsburg é onde tem mais opções para locação 
#Média de preço por região

dados_airbnb.groupby(['neighbourhood_group']).mean()[['price']].sort_values(['price'], ascending=[False])

#Podemos constatar que Manhattan tem a média de preço mais alta dos distritos e o Bronx a mais baixa
#Definindo o tamanho do gráfico

plt.figure(figsize=(12,5))

#Inserindo grids no layout do gráfico

sns.set(style="darkgrid")



#Gerando o gráfico 

grafico = sns.countplot(x='room_type', hue='neighbourhood_group',data= dados_airbnb)



#É possível visualizar, tipos de acomodações X Distritos.

#"Entire home/ apt room_type" é o tipo de acomodação com mais opções em Manhattan

#"Private room" é o tipo de acomodação com mais opções em Brooklyn

#"Shared room" é o que menos tem opção para locação
#Análise de correlação

#Por padrão está sendo usado "Coeficiente de Correlação de Pearson"

#Apliquei cor para melhor visualização

dados_airbnb.corr().style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'),axis=1)



#Pode ser observado que não existe correlação entre os valores da base

#As campos com 1.0 são falsos positivos devido estarem se relacionando com a mesma coluna.

#Aguns campos como "id" não são relevantes para análise, então estou desconsiderando