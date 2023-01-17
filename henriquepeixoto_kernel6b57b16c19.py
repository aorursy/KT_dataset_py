#Import modules.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()
#Link to dataset, I didn´t bought it because I wanted only the total and it was very expensive, so I typed what I needed ;) 

#Link https://www.statista.com/statistics/875319/number-flights-brazil/



total = pd.read_csv('Total_flights.csv')

total.set_index('Ano',inplace=True)
flightsT = pd.read_csv('BrFlights2.csv', encoding='latin1')
avi = pd.read_csv('accidents.csv')

avi1 = pd.DataFrame(avi)
flightsT.head()
flightsT.info()
avi1.info()
avi1.head()
avi1.corr()
avi1.var()
#Deleting columns that were used to ML and Brasil

column = list(avi1.columns.values)

column

avi1.drop([

 'fator_1',

 'fator_2',

 'fator_3',

 'fator_4',

 'fator_5',

 'fator_6',

 'fator_7',

 'fator_8',

 'fator_9',

 'fator_10',

 'fator_11',

 'fator_12',

 'fator_13',

 'fator_14',

 'fator_15',

 'fator_16',

 'fator_17',

 'fator_18',

 'fator_19',

 'fator_20',

 'fator_21',

 'fator_22',

 'fator_23',

 'fator_24',

 'fator_25',

 'fator_26',

 'fator_27',

 'fator_28',

 'fator_29',

 'fator_30',

 'fator_31',

 'fator_32',

 'fator_33',

 'fator_34',

 'fator_35',

 'fator_36',

 'fator_37',

 'fator_38',

 'fator_39',

 'fator_40',

 'fator_41',

 'fator_42',

 'fator_43',

 'fator_44',

 'fator_45',

 'fator_46',

 'fator_47',

 'fator_48',

 'fator_49',

 'fator_50',

 'fator_51',

 'fator_52',

 'fator_53',

 'fator_54',

 'fator_55',

 'fator_56',

 'fator_57',

 'fator_58',

 'fator_59',

 'fator_60',

 'fator_61',

 'fator_62',

 'fator_63',

 'fator_64',

 'fator_65',

 'fator_66',

 'fator_67',

 'fator_68',

 'fator_69',

 'fator_70',

 'fator_71',

 'fator_72',

 'fator_73',

 'fator_74',

 'fator_75',

 'fator_76',

 'fator_77',

 'fator_78',

 'fator_79',

 'fator_80',

 'fator_81',

 'fator_82',

 'fator_83',

 'fator_84',

 'fator_85',

 'fator_86',

 'fator_87',

'ocorrencia_pais'], inplace=True, axis=1)



"""The 'factor' was taking off because we are not using any ML to this model, and 'ocorrencia_pais' was also removed because all of the accidents happened in Brazil"""
#Creating year and changing date to 'datetime'.



avi1['ocorrencia_dia'] = pd.to_datetime(avi1['ocorrencia_dia'])

avi1['Dia'] = avi1['ocorrencia_dia'].map(lambda x: x.day)

avi1['Ano'] = avi1['ocorrencia_dia'].map(lambda x: x.year)

avi1['Mes'] = avi1['ocorrencia_dia'].map(lambda x: x.month)



avi1.head()
#Checking to see if I didn´t broke anything.

avi1.info()
"""Conseguimos observar que existem correlações obvias, como quantidades de assentos e falatalidade, ou quantidade de assentos e número de motos no avião. Com isso não  podemos tirar nenhuma conclusão avançada sobre o assunto."""



f, ax = plt.subplots(figsize=(15,6))

sns.heatmap(avi1.corr(), annot=True, fmt='.2f', ax=ax, linecolor='black', lw=.7)
#Marketshare of the brazilian market.

mshare = flightsT['Companhia.Aerea'].value_counts()

mask = mshare > 100000

mshare2 = mshare[mask]

pd.DataFrame(mshare2)



# summ = mshare.sum()

# minus = mshare2.sum()

# Other = mshare.sum() - minus
"""As we can see we have few relevant companies in the market."""

mshare2.plot(kind='pie', subplots=True, label="." ,figsize=(8, 8))
#The number of accidents versus the total number of flights.

gb3 = avi1[['Ano','quantidade_fatalidades']].groupby('Ano').sum()



percent = (gb3.values*100)/sum(total.values)



pd.DataFrame(percent)



total['Accidents'] = gb3.values

total['Accidents Percent'] = percent

total['Accidents'].astype(int)

total
#Total number graph:

"""It´s so small that in a plot bar you can´t see."""

total2 = total.drop('Accidents Percent', axis= 1)

total2



total2.plot.bar(figsize=(8, 4))

#Finding the cities with the most number of accidents:

mask = avi1['ocorrencia_uf'].value_counts() 

mask2 = mask > 60

mask2

city = mask[mask2]

city



city.plot.bar(color='indigo', figsize=(8, 4))



"""As you can see in the plot bellow most of the acccidents happens in the southeast of Brazil."""
#How many people died in each UF in total.

gb2 = avi1[['ocorrencia_uf','quantidade_fatalidades']].groupby('ocorrencia_uf').sum()

gb2.plot.bar(color='indigo', figsize=(8, 4))





"""Waaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaay more people dies in SP."""
flightR = flightsT['Estado.Origem'].value_counts()

flightR2 = flightR.apply(lambda x: (x*100)/sum(flightR))

flightP = pd.DataFrame(flightR2)   



city2 = city.apply(lambda x: (x*100)/sum(city))

flightP.columns = ['Percentage of flights']
flightP['Percentage of accidents'] = city2

flightP.head(4).plot.bar(figsize=(8, 4))



"""as you can see there is fewer accidents in SP compared to the percent of the number of flights."""
#Number of accidents through the years.

sns.distplot(avi1["Ano"], color='indigo')
#Number of deaths through the years.

"""We can get to the conclusion that in 2016 a lot of died in fewer accidents (the accidents were bloodier (I don´t know if this word exist))"""

gb2 = avi1[['Ano','quantidade_fatalidades']].groupby('Ano').sum()

gb2.plot(figsize=(8, 4))
#Category of the accidents.

# Tipos de classificações existentes na base de dados

plt.style.use("ggplot")



#Visualizando melhor em um gráfico

avi1['ocorrencia_classificacao'].value_counts().plot(kind='pie', subplots=True, label="Classificação de Acidentes" ,figsize=(6, 6))
#The most frequent kinds of accidents.

plt.style.use("ggplot")

a = avi1['ocorrencia_tipo'].value_counts()

a.head(10).plot(kind='barh', subplots=True, label="Tipos de ocorrência" ,figsize=(8, 4), color='indigo')

plt.xticks(rotation=80)

"""Como podemos ver, falha do motor em voo é a principal ocorrência."""
#Model and manifacture of planes that suffered the most number of accidents:

avi1['aeronave_modelo'].value_counts()

con = avi1[['aeronave_modelo','aeronave_fabricante']].groupby('aeronave_modelo').sum()

pd.DataFrame(con)



mask6 = avi1['aeronave_modelo'].value_counts() 

mask7 = mask6 > 60



model = mask6[mask7]

pd.DataFrame(model)

model.columns = 'Numero de acidentes'

model.plot.bar(color='indigo', figsize=(8, 4))
gb = avi1[['aeronave_modelo','aeronave_quantidade_assentos']].groupby('aeronave_modelo').mean()

#Manufacture of the models that fall the most.

model['aeronave_modelo'] = model.index



result = con.merge(model, how='right', left_on="aeronave_modelo", right_index=True)

result.columns = 'Fabricante', 'Acidentes'



result
#Manufacture of the models that fall the most (Now in a plot).



fabricantes = avi1['aeronave_fabricante'].value_counts().head(10)

fabricantes.plot(kind='bar', subplots=True, label="Fabricantes" ,figsize=(8, 4), color='indigo')

plt.xticks(rotation=80, color='indigo')



"""Don´t fly in Cessna Aircraft."""
#Category that has most accidents.

kind = avi1['aeronave_segmento_aviacao'].value_counts()

kind.plot.bar(color='indigo', figsize=(8, 4))
#Kind of engine that falls the most.

tipo_motor = avi1['aeronave_tipo_motor'].value_counts()

tipo_motor.plot.bar(color='indigo', figsize=(8, 4))
# #Getting the mean number of seats of the planes above.

# means = []

# seats = []

# for i in model.index:

#     seats.append(i)

  





# # for i in seats:

#        means.append(gb.loc[i])



# meansseats = pd.DataFrame(means)

# #Creating a function to transform from list to dictionary 

# #Yes, I know I could use lambda, but it didn´t worked, so I thought working is better than not working.

# """This means that smaller aircrafts have a bigger chance of suffer a accident"""



# # meansseats.plot.bar()