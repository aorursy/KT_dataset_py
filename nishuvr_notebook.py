%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pandas as pd

data = (pd.read_csv("/kaggle/input/geospatial-sao-paulo-crime-database/dataset-limpo.csv")

        )

import seaborn as sns

import warnings; warnings.filterwarnings(action='once')

#removal of the columns with more than 80% null values

data=data.loc[:,data.isnull().mean()<.8]

data
#crime rate per year

data.index = pd.DatetimeIndex(data.time)

data.groupby([data.index.year]).size().plot(kind='bar')

plt.ylabel('Year')

plt.xlabel('Number of crimes')

plt.title('Number of crimes per year')

plt.show()

#crime rate is decreasing with time
#removal of all null values in column bairro

data1= data[pd.notnull(data['bairro'])]

data1
value_counts=data1['bairro'].value_counts()

to_remove=value_counts[value_counts<50].index

data2=data1[~data1.bairro.isin(to_remove)]

plt.figure(figsize=(30,20))

plt.xticks(rotation=270)

pd.value_counts(data2['bairro']).plot.bar()



#Saeo paulo has the most no of crimes and hence it is unsafe to live in
#sex of the victim

data = (pd.read_csv("/kaggle/input/geospatial-sao-paulo-crime-database/dataset-limpo.csv")

        )

s=data['sexo'].value_counts()

print(s)

label=['male','female']

colors=['lightskyblue','lightcoral']

plt.pie(s,labels=label,colors=colors,autopct='%1.1f%%',shadow=True,startangle=140)

plt.axis('equal')

plt.show()

data['sexo'].value_counts(normalize=True)
import math

import statsmodels.api as sm

#COMPARING THE PROPORTION FOR TWO INDEPENDENT SAMPLES

#h0: proportion of female victim = proportion of male victim

#h1: proportion of female victim is not equal to proportion of male victim

#males proportion

prop_m = 0.612528 

male = 7901

#females proportion

prop_f = 0.387472

female = 4998

n= 12899

#assuming 95% #confidence

population1= np.random.binomial(1,prop_m,male)

population2= np.random.binomial(1,prop_f,female)

sm.stats.ttest_ind(population1,population2)

#since pvalue(~1.5258821953072503e-167) is very low we can reject h0;

#hence proportion of female victim is not equal to proportion of male victim

#no of registered vs unregistered crimes

s=data['registrou_bo'].value_counts()

print(s)

label=['registered','unregistered']

colors=['lightskyblue','lightcoral']

plt.pie(s,labels=label,colors=colors,autopct='%1.1f%%',shadow=True,startangle=140)

#only 59.3% of the crimes were registered this shows the carelessness of the poplulation
#Percentages of incidents involving phones monthly

data = (pd.read_csv("/kaggle/input/geospatial-sao-paulo-crime-database/dataset-limpo.csv", parse_dates=['time'])

        )



df1 = data[pd.notnull(data['bairro'])]

print(data['Celular'].count())

df1 = df1[df1.time > '2015-01-01']

plt.figure(figsize=(12, 4))

resampled_data = df1.resample("M", on='time')

phone_count = resampled_data.Celular.count()

phone_prop = phone_count / resampled_data['id'].count() * 100

phone_prop.plot()

plt.xlabel("Time")

plt.ylabel("Percentages of incidents involving phones ")

plt.title("Stolen phones per Month")

plt.show()
#Percentages of incidents involving phones yearly

df1 = data[pd.notnull(data['bairro'])]

print(data['Celular'].count())

plt.figure(figsize=(12, 4))

resampled_data = df1.resample("Y", on='time')

phone_count = resampled_data.Celular.count()

phone_prop = phone_count / resampled_data['id'].count() * 100

phone_prop.plot()

plt.xlabel("Time")

plt.ylabel("Percentages of incidents involving phones ")

plt.title("Stolen phones per Year")

plt.show()
#No of incidents involving documents yearly

df1 = data[pd.notnull(data['Documentos'])]

df1 = df1[df1.time > '2012-01-01']

print(data['Documentos'].count())

plt.figure(figsize=(12, 4))

resampled_data = df1.resample("Y", on='time')

doc_count = resampled_data.Celular.count()

doc_count.plot()

plt.xlabel("Time")

plt.ylabel("Percentages of incidents involving documents")

plt.title("Stolen Documents per Year")

plt.show()

#very less proportion of people carry documents these days they use softcopy and have everything saved in laptops 