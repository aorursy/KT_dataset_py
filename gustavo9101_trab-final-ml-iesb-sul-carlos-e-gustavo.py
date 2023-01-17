import numpy as np 

import pandas as pd 

from pandas import set_option

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

%matplotlib inline



import os

print(os.listdir("../input"))



filename = ("../input/housing.csv")

names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'CM', 'UA40', 'DIS', 'RAD', 'IMPOSTO', 'PROF_ALUNO', 'B', 'LSTAT', 'MEDV']

df = pd.read_csv(filename, delim_whitespace=True, names=names)
df.head()
df.shape
# Observa-se que a base não contempla valores missing



df.info()
# Verificando os tipos das colunas

print('========Contagem==========')

print(df.dtypes.value_counts())

print('==========================')

print('=======Percentual=========')

print(df.dtypes.value_counts(normalize=True).apply("{:.2%}".format))

print('==========================')
# Vizualização gráfica dos tipos de colunas



f, axes=plt.subplots(1,2, figsize=(15,6))

plt.suptitle('Caracteristicas das colunas', ha='center', fontsize=14)

P=df.dtypes.value_counts().plot.pie(autopct='%1.2f%%',ax=axes[0], label='',title='Tipos Colunas - Distr Percentual', legend=True)

bplot = df.dtypes.value_counts().plot(kind='bar',ax=axes[1],rot=0)

for b in bplot.patches:

    bplot.annotate(format(b.get_height(),'.0f'), \

                   (b.get_x() + b.get_width() / 2., \

                   b.get_height()), \

                   ha = 'center',\

                   va = 'center',\

                   xytext = (0, 7),\

                   textcoords = 'offset points')    

plt.title('Tipos Colunas - Contagem')

plt.xlabel('')

plt.yticks([])

plt.ylabel('Frequência',labelpad=3)



sns.despine(left=True)
df.describe()
df.isnull().sum().max()
#verificando visualmente a distribuição dos valores missing

msno.matrix(df,figsize=(12,5))
#Valores missing no data frame DF

df.isnull().sum().to_frame('Qtd. Missing')
# Identificando Outliers nos campos



from scipy import stats



fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))

index = 0

axs = axs.flatten()

for k,v in df.items():

    sns.boxplot(y=k, data=df, ax=axs[index])

    index += 1

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
# Percentual de outliers de cada campo



for k, v in df.items():

        q1 = v.quantile(0.25)

        q3 = v.quantile(0.75)

        irq = q3 - q1

        v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]

        perc = np.shape(v_col)[0] * 100.0 / np.shape(df)[0]

        print("Outliers da coluna %s = %.2f%%" % (k, perc))
# Verificando a taxa de criminalidade nos agrupamentos de Proximidade ao Rio e de Acessibilidade às rodovias radiais



sns.set(rc={'figure.figsize':(16.7,8.27)})

sns.swarmplot(x='CHAS', y='CRIM', data=df,hue='RAD')

plt.title('Taxa de Criminalidade por Proximidade ao Rio e Acessibilidade às rodovias')

plt.xlabel('Área Beira Rio (1 = Proximo, 0 = Distante)',labelpad=10)

plt.ylabel('Taxa de Criminalidade',labelpad=10)
# Avaliando o nível de correlação das variáveis explicativas com a variável resposta



plt.figure(figsize=(20, 10))

sns.heatmap(df.corr().abs(),  annot=True)
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

column_sels = ['LSTAT', 'INDUS', 'NOX', 'PROF_ALUNO', 'CM', 'IMPOSTO']

x = df.loc[:,column_sels]

y = df['MEDV']

x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)

fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))

index = 0

axs = axs.flatten()

for i, k in enumerate(column_sels):

    sns.regplot(y=y, x=x[k], ax=axs[i])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
X = df[['LSTAT', 'INDUS', 'NOX', 'PROF_ALUNO', 'CM', 'IMPOSTO']]

y = df['MEDV']
# Dividindo os dados em um conjunto de treinamento e um conjunto de testes. 



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression



lm = LinearRegression()



lm.fit(X_train,y_train)
# Printando a intercepção

print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

coeff_df
predictions = lm.predict(X_test)



plt.scatter(y_test,predictions)

from sklearn.metrics import r2_score



r2_score(y_test, predictions)