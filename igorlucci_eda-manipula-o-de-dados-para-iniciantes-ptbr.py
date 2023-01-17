import pandas as pd #Manipulação

import numpy as np #Manipulação

import matplotlib.pyplot as plt #Visualização

import seaborn as sns #Visualização
df = pd.read_csv("../input/titanic/test.csv") 
df.dtypes
df.describe(include=[np.number]).T
df.describe(include='O')

pd.isnull(df)
df.isnull().any()
df.isnull().sum()
df.isnull().sum().sum()
np.mean(df['Age'])
df.loc[pd.isnull(df['Age']),'Age'] = np.mean(df['Age'])
df.describe(include=[np.number])
df.isnull().sum()
df = df.drop(['PassengerId', 'Parch','Fare','Cabin'],axis = 1) #axis = 1 -> Estou falando que quero que as colunas sejam removidas;





df.describe(include=[np.number])
df.describe(include='O')

df.loc[pd.isnull(df['Embarked']),'Embarked'] = 'S'
df.describe(include='O')
df.isnull().sum()
df = df.drop(['Name','Ticket'],axis=1)
df.describe(include='O')
df['Gender'] = 0
df.describe(include=[np.number])
df.head()
df.loc[df['Sex']=='female','Gender'] = 1
df.describe(include=[np.number])
df.describe(include='O')
dummies = pd.get_dummies(df['Embarked'],prefix='Embarked')
df=pd.concat([df,dummies],axis=1)
df.describe(include=[np.number])
df=df.drop(['Sex','Embarked'],axis=1)
df.dtypes
df.head()
df['Age'] = round(df['Age']).values.astype(np.int64)
df.dtypes
df.to_csv('titanic_numeric_only.csv',index=False)
plt.figure(figsize=(10,10))



df.Age.value_counts().plot(kind='bar')

plt.ylabel('Counts')

plt.xlabel('Age Groups')

plt.title('Frequence of Age Groups')

plt.show()
plt.figure(figsize=(10,10))



df.Gender.value_counts().plot(kind='bar')

plt.ylabel('Counts')

plt.xlabel('Gender')

plt.title('Gender Distribution')

plt.show()
df.Age.hist()

plt.xlabel('Idade')

plt.ylabel('Freq')

plt.show()

## Aqui percebemos o quanto é melhor um histograma para plotar grupos de idade, do que um bar chart, que foi  o primeiro gráfico exibido.
df.Age.plot()
#FIM Espero que este notebook possa ajudar os colegas falantes da língua portuguesa :)