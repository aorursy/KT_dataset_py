# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/Mall_Customers.csv')

#read csv for analysis
#Excluindo a coluna CustomerID

df = df.drop(['CustomerID'],axis =1)
#Mostrando os 5 primeiros registros - Ponta

df.head()
#Mostrando as ultimas linhas - Calda

df.tail()
#Outra forma de apresentar a ponta e a calda, com numero maximo de 10 linhas

pd.options.display.max_rows = 10

df
#Mostrando as Colunas

df.columns
#Contando valores nulos

df.isnull().sum()
#Tipo de dados

df.dtypes
df.describe()
#Correlaçao de dados

df.corr()
#Contando a quantidade de resgistro na coluna Genero

df['Gender'].value_counts()
#Idade - Masculina

print('Maximo :',max(df[df['Gender']=='Male'].Age))

print('Minimo  :',min(df[df['Gender']=='Male'].Age))

print('Média :',np.mean(df[df['Gender']=='Male'].Age))

print('Std  :',np.std(df[df['Gender']=='Male'].Age))
#Idade - Feminina

print('Maximo  :',max(df[df['Gender']=='Female'].Age))

print('Minimo  :',min(df[df['Gender']=='Female'].Age))

print('Média :',np.mean(df[df['Gender']=='Female'].Age))

print('Std  :',np.std(df[df['Gender']=='Female'].Age))
#Histograma em Annual Income - Distribuiçai da Renda Anual

df["Annual Income (k$)"].plot.hist(color ='g', title="Distribuição de Renda Anual",edgecolor='black')
#Grafico de barras horizontal demonstrando a quantidade de registros por Genero.

df['Gender'].value_counts().plot.barh(edgecolor='black', title="Qtd de pessoas por Gênero")
#Utilizando a Biblioteca seaborn.

plt.figure(1 , figsize = (15 , 5))

sns.countplot(y = 'Gender' , data = df)

plt.show()
#Scatter Plot

df.plot.scatter(x='Age', y='Spending Score (1-100)', color ='red', alpha=0.4)
plt.style.use('ggplot')

df.plot.scatter(x='Annual Income (k$)', y='Spending Score (1-100)')
#Plotanto uma relação linear simples entre duas variáveis

sns.lmplot(x='Age',y='Spending Score (1-100)',data=df)
#Com o Jointplot é possivel verificar  que a faixa etária de 20 a 40 anos é muito mais dispendiosa em comparação com grupos etários mais velhos.



sns.jointplot(x='Age',y='Spending Score (1-100)',data=df,kind='kde')
#Usando o pairplot -  o legal é que automaticamente podemos plotar com facilidade varias colunas do dataset, sem precisar definir os eixos



plt.style.use('seaborn-deep')

sns.pairplot(df)

plt.title('Usando o Pair plot', fontsize = 20)