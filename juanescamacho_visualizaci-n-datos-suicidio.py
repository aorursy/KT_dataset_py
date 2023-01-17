import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import seaborn as sns

print(os.listdir("../input"))

dataset = pd.read_csv('../input/master.csv')
dataset.columns
#Cambiando los nombres para un mejor analisis

dataset=dataset.rename(columns={'country':'Country','year':'Year','sex':'Gender','age':'Age','suicides_no':'SuicidesNo','population':'Population','suicides/100k pop':'Suicides100kPop','country-year':'CountryYear','HDI for year':'HDIForYear',' gdp_for_year ($) ':'GdpForYearMoney','gdp_per_capita ($)':'GdpPerCapitalMoney','generation':'Generation'})   
dataset.head()
#Revisando datos nulos en la muestra 

dataset.isnull().any()
#Se comprueba cuantos valores nulos hay en la muestra

dataset.isnull().sum()
#Se eliminan los valores nulos que hay en la muestra

dataset=dataset.drop(['HDIForYear','CountryYear'],axis=1)
unique_country = dataset['Country'].unique()

print(unique_country)
#Revisando estadisiticas de suicidio por paìs

alpha = 0.7

plt.figure(figsize=(10,25))

sns.countplot(y='Country', data=dataset, alpha=alpha)

plt.title('Datos por pais')

plt.ylabel('Pais')

plt.xlabel('Total')

plt.show()
### Set figure size

plt.figure(figsize=(16,7))

##Plot the graph

Gender = sns.countplot(x='Gender',data = dataset)
#Definiendo el primer y ultimo año de la muestra

min_year=min(dataset.Year)

max_year=max(dataset.Year)

print('Min Year :',min_year)

print('Max Year :',max_year)



#1985 min year,2016 max year.



dataset_country=dataset[(dataset['Year']==min_year)]



country_1985=dataset[(dataset['Year']==min_year)].Country.unique()

country_1985_male=[]

country_1985_female=[]



for country in country_1985:

    country_1985_male.append(len(dataset_country[(dataset_country['Country']==country)&(dataset_country['Gender']=='male')]))

    country_1985_female.append(len(dataset_country[(dataset_country['Country']==country)&(dataset_country['Gender']=='female')])) 

    

#Se encontro 

plt.figure(figsize=(10,10))

sns.barplot(y=country_1985,x=country_1985_male,color='red')

sns.barplot(y=country_1985,x=country_1985_female,color='yellow')

plt.ylabel('Paises')

plt.xlabel('Hombres vs Mujeres')

plt.title('Tasa de suicidio por genero 1985')

plt.show()



#Ahora para el último año de la muestra (2016)

dataset_country=dataset[(dataset['Year']==max_year)]



country_2016=dataset[(dataset['Year']==max_year)].Country.unique()

country_2016_male=[]

country_2016_female=[]



for country in country_2016:

    country_2016_male.append(len(dataset_country[(dataset_country['Country']==country)&(dataset_country['Gender']=='male')]))

    country_2016_female.append(len(dataset_country[(dataset_country['Country']==country)&(dataset_country['Gender']=='female')])) 

    



plt.figure(figsize=(10,10))

sns.barplot(y=country_2016,x=country_2016_male,color='red')

sns.barplot(y=country_2016,x=country_2016_female,color='yellow')

plt.ylabel('Paises')

plt.xlabel('Hombres vs Mujeres')

plt.title('Tasa de suicidio por genero 1985')

plt.show()
### Set figure size

plt.figure(figsize=(16,7))

cor = sns.heatmap(dataset.corr(), annot = True)
plt.figure(figsize=(16,7))

bar_age = sns.barplot(x = 'Gender', y = 'SuicidesNo', hue = 'Age',data = dataset)

dataset_year_min=dataset[(dataset['Year']==min_year)]

plt.figure(figsize=(16,7))

bar_age = sns.barplot(x = 'Gender', y = 'SuicidesNo', hue = 'Age',data = dataset_year_min)

dataset_year_max=dataset[(dataset['Year']==max_year)]

plt.figure(figsize=(16,7))

bar_age = sns.barplot(x = 'Gender', y = 'SuicidesNo', hue = 'Age',data = dataset_year_max)
plt.figure(figsize=(16,7))

bar_gen = sns.barplot(x = 'Gender', y = 'SuicidesNo', hue = 'Generation',data = dataset)
plt.figure(figsize=(16,7))

lp_suicidesno = sns.lineplot(x = 'Year', y = 'SuicidesNo', data = dataset)
