# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

# Trabalho de python de Anál
# carregando os dados

# Carregando os dados

df= pd.read_csv('/kaggle/input/heart-disease-dataset/heart.csv')
# olhando o tamanho dos dados

print('heart:', df.shape)
# Dados do Heart (coração)

#Variáveis que constam na database

# 1.age ( idade)

# 2.sex  (genero)

# 3.chest pain type (4 values) ( dor no peito)

# 4.resting blood pressure ( Pressão sanguínia em repouso)

# 5.serum cholestoral in mg/dl (colestoro sérico)

# 6.fasting blood sugar > 120 mg/dl (açúcar no sangue em jejum)

# 7. resting electrocardiographic results (values 0,1,2) (resultados eletrocardiográficos em repouso)

# 8.maximum heart rate achieved ( máxima frequencia cardiaca alcançada)

# 9.exercise induced angina (angina induzida por exercício)

# 10.oldpeak = ST depression induced by exercise relative to rest (depressão do ST induzida pelo exercício em relação ao repouso)

# 11.the slope of the peak exercise ST segment (a inclinação do segmento ST do pico do exercício)

# 12. number of major vessels (0-3) colored by flourosopy ()

# 13.thal: 3 = normal; 6 = fixed defect; 7 = reversable defect ()

# 14. Target (objetivo)

df.head()
# olhando os últimos lançamentos

df.tail()
# informações

df.info()
# mostrar as colunas existentes

print('Data Show Columns:\n')

df.columns


# Quantas colunas e fileiras existem

print('Data Shape Show\n')

df.shape
# Ve se tem algum dado faltando. Missing Values

print('Data Sum of Null Values \n')

df.isnull().sum()
# Resumo estatístico do Dataframe

df.describe()
# Aumentando a area do gráfico

plt.figure(figsize=(15,6))

sns.heatmap(df.corr(),annot=True,fmt='.2f')

plt.show()

# Plotando a Age(idade)

df['age'].hist(grid=True, bins=10); 

plt.title('Age distribuition')
df.age.value_counts()[:10]

# data age show value counts for age least 10 # Mostrar a idade dos ultimos 10.
# Renomeando as colunas de acordo com os exercícios dados em sala de aula.  Só a primeira letra da palavra passará a ser maiúscula

df=df.rename(columns={'age':'Age','sex':'Sex','cp':'Cp','trestbps':'Trestbps','chol':'Chol','fbs':'Fbs','restecg':'Restecg','thalach':'Thalach','exang':'Exang','oldpeak':'Oldpeak','slope':'Slope','ca':'Ca','thal':'Thal','target':'Target'})
#display a nova coluna com nomes novos

df.columns
# analise da idade  que acontecem a doença do coração

sns.barplot(x=df.Age.value_counts()[:10].index,y=df.Age.value_counts()[:10].values)

plt.xlabel('Age')

plt.ylabel('Age Counter')

plt.title('Age Analysis System')

plt.show()
# Resumo dos dados da idadde.Total da soma (sum)

df.sum()

# minima dos dados estatísticos

df.min()
# soma das máximas

df.max()
#Index do menor valor

df.min()
# indexando a média dos valores

df.mean()
# Analisando em um conjunto total o impactato da idade no problema do coração. MINIMA E MÁXIMA

minAge=min(df.Age)

maxAge=max(df.Age)

meanAge=df.Age.mean()

print('Min Age :',minAge)

print('Max Age :',maxAge)

print('Mean Age :',meanAge)
# Analise do sexo

df.Sex.value_counts()
# Sexo (1 = male; 0 = female) 

sns.countplot(df.Sex)

plt.show()
total_genders_count=len(df.Sex)

male_count=len(df[df['Sex']==1])

female_count=len(df[df['Sex']==0])

print('Total Genders :',total_genders_count)

print('Male Count    :',male_count)

print('Female Count  :',female_count)
#  Analisando dor no peito (Cp)

df.Cp.value_counts()
sns.countplot(df.Cp)

plt.xlabel('Chest Type')

plt.ylabel('Count')

plt.title('Chest Type vs Count State')

plt.show()

#0 situação Mínima

#1 condição pequeno problema

#2 condição problema não tão grave

#3 condição Muito mal
#Visualizando os dados do analise

df.head(1)
# freqência cardiaca. thalac values quer dizer = frequência cardiaca. 

# mostraremos as primeiras 10 fileiras

df.Thalach.value_counts()[:10]
# Vamos plotar usando SNS gráficos SEABORN

sns.barplot(x=df.Thalach.value_counts()[:20].index,y=df.Thalach.value_counts()[:20].values)

plt.xlabel('Thalach')

plt.ylabel('Count')

plt.title('Thalach Counts')

plt.xticks(rotation=45)

plt.show()
# Idade entre 40 a 60 anos

sns.distplot(df['Age'],color='Red',hist_kws={'alpha':1,"linewidth": 2}, kde_kws={"color": "k", "lw": 3, "label": "KDE"})

df.head()
age_unique=sorted(df.Age.unique())

age_thalach_values=df.groupby('Age')['Thalach'].count().values

mean_thalach=[]

for i,age in enumerate(age_unique):

    mean_thalach.append(sum(df[df['Age']==age].Thalach)/age_thalach_values[i])

#data_sorted=data.sort_values(by='Age',ascending=True)

# Idade e frequência cardia, duas variáveis importantes para detectar um problema cadíaco

plt.figure(figsize=(10,5))

sns.pointplot(x=age_unique,y=mean_thalach,color='red',alpha=0.8)

plt.xlabel('Age',fontsize = 15,color='blue')

plt.xticks(rotation=45)

plt.ylabel('Thalach',fontsize = 15,color='blue')

plt.title('Age vs Thalach',fontsize = 15,color='blue')

plt.grid()

plt.show()
# variavel bem importante na causa de doença do coração (Colesterol)

df['Chol'].describe()
# plotando a variável do colesterol

ax, figure = plt.subplots(figsize = [8,6])

sns.distplot(df['Chol'], color = 'b');
# Plotando para ver idader(age) com o nivel de colesterol.

ax, figure = plt.subplots(figsize = [9,5])

sns.regplot(x="Age", y="Chol", data=df);
# Media de idade e nivel de colesterol 

df.groupby('Age')['Chol'].mean()
# agrupamento do sexo e colesterol.

df.groupby('Sex')['Chol'].mean()