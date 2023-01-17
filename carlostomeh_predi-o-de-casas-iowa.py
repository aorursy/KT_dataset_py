# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Carregando base de dados de treino

main_file_path = "../input/train.csv"

treino = pd.read_csv(main_file_path) 

treino.describe()
main_file_path = "../input/test.csv"

teste = pd.read_csv(main_file_path)

teste.describe()
#Conhecendo as colunas 

print("\n Todas as colunas no conjunto de Dados de Treino\n\n",treino.columns.values)

print("\n Todas as colunas no conjunto de Dados de Teste\n\n",teste.columns.values)
# Analisando o tipo de dado para cada Feature

# No conjunto de treino

treino.get_dtype_counts()

# No conjunto de teste

teste.get_dtype_counts()

# Encontrando as colunas numericas e salvando na variavel num_treino

num_treino = treino.select_dtypes(include=[np.number])

#imprimindo os resultados

print('Total de features numéricas : \n', num_treino.shape)

num_treino.dtypes
# Visualizando o Target 'SalePrice'

# A idéia aqui é ver a distribuição do preço das casas



import seaborn as sns

plt.figure(1 , figsize = (15 , 7))  

sns.distplot(treino['SalePrice'], color='blue', bins=100);

plt.title('Preço das Casas')

plt.show()
#Entendendo a correlação entre as features e o Target para as variaveis numericas



corr = treino.corr()

# 10 maiores valores de correlação para as features 

print(corr['SalePrice'].sort_values(ascending=False)[1:11],'\n')

# 10 menores valores de correlação para as features 

print(corr['SalePrice'].sort_values(ascending=False)[-10:],'\n')

# Visualização para as correlações 

# Mapa de calor para as correlações entre as features x SalePrice



corrmat = treino.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True,cmap="YlGnBu")
# Visualização para as correlações 

# Mapa de calor para as correlações entre as features x SalePrice

# aqui eu diminui o intervalo de correlacao com corrmax = 0.9 e corrmin = 0.1, isso melhora a visualização



corrmat = treino.drop('Id', axis=1).corr()

plt.subplots(figsize=(13,13))

mask = np.zeros_like(corrmat, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corrmat, mask=mask, vmin = 0.1,vmax=0.9, cmap="YlGnBu", square=True, cbar_kws={"shrink": .5}, linewidths=0.6);

plt.title('Correlação dos Dados');
# Depois dessa breve analise sobre o Target agora o objetivo é tratar os dados

# É interessante aqui concatenar os conjuntos de dados de teste e treino

# Para que eles possam receber o mesmo tipo de tratamento, a fim de minimizar problemas futuros



dataframe = pd.concat([treino, teste], keys=['treino', 'teste'], axis=0, sort=False)



# Também é interessante aqui guardar o Target em um outro objeto

y_treino = treino.SalePrice.values



# E vamos retirar o Target desse conjunto de dados

# para podermos tratar valores nulos e etc

# também vou retirar a coluna Id que não é necessaria aqui



dataframe.drop(['SalePrice'], axis=1, inplace=True)

dataframe.drop(['Id'], axis=1, inplace=True)



print('O tamanho total do Dataframe concatenado é: ', dataframe.shape)



dataframe.head(5)

# TRATANDO VALORES NULOS

# Uma parte muito importante na preparação dos dados é tratar os 'missing values'

# Existem varias estratégias para repor ou retirar dados através da analise dos valores nulos





# Contando os missing values das features



nulos = pd.DataFrame(dataframe.isnull().sum().sort_values(ascending =False))

nulos.columns = ['Soma dos Nulos']

nulos.index.name = 'Feature'

nulos.head(19)





# Existe uma grande quantidade de valores nulos em algumas features

# Analisando a porcentagem dos valores nulos nos conjuntos de dados, podemos criar o grafico:



todosnulos = (dataframe.isnull().sum()/ len(dataframe))*100

todosnulos = todosnulos.drop(todosnulos[todosnulos == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Fatia de Missing Values' :todosnulos})

missing_data.head(20)
# Agora a idéia é imputar valores de modo que o conjunto não fique sem dados nulos

# Imputando valores faltantes que tem significado na descrição do dataset



# Imputando o valor zero seguindo recomendações das descrições dos dados

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath','MasVnrArea'):

    dataframe[col] = dataframe[col].fillna(0)



# Imputando a palavra Não, como recomemdado pela descrição dos dados

for col in ('Alley','PoolQC','MSSubClass','MiscFeature','BsmtQual','Fence','BsmtCond','BsmtExposure','BsmtFinType1', 'BsmtFinType2','MasVnrType','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','FireplaceQu'):

    dataframe[col] = dataframe[col].fillna('None')



# imputando o valor mais comum, nesses casos existem apenas um ou dois valores nulos no conjunto, portanto imputar o valor mais comum é plausivel

dataframe['MSZoning'] = dataframe['MSZoning'].fillna(dataframe['MSZoning'].mode()[0])

dataframe['Electrical'] = dataframe['Electrical'].fillna(dataframe['Electrical'].mode()[0])

dataframe['KitchenQual'] = dataframe['KitchenQual'].fillna(dataframe['KitchenQual'].mode()[0])

dataframe['Exterior1st'] = dataframe['Exterior1st'].fillna(dataframe['Exterior1st'].mode()[0])

dataframe['Exterior2nd'] = dataframe['Exterior2nd'].fillna(dataframe['Exterior2nd'].mode()[0])

dataframe['SaleType'] = dataframe['SaleType'].fillna(dataframe['SaleType'].mode()[0])



# imputando a média

dataframe["LotFrontage"] = dataframe.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



# retirando "Utilities" do dataframe, pois não há ganho de informação com essa feature

dataframe = dataframe.drop(['Utilities'], axis=1)



#Substituindo NA por Typ.

dataframe["Functional"] = dataframe["Functional"].fillna("Typ")





#verificando se ainda existem valores nulos



todosnulos = (dataframe.isnull().sum()/ len(dataframe))*100

todosnulos = todosnulos.drop(todosnulos[todosnulos == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Fatia de Missing Values' :todosnulos})

missing_data.head(20)
# Adicionando mais uma feature



dataframe['TotalSF'] = dataframe['TotalBsmtSF'] + dataframe['1stFlrSF'] + dataframe['2ndFlrSF']



# transformando algumas variaveis em categoricas



dataframe['MSSubClass'] =   dataframe['MSSubClass'].astype(str)

dataframe['KitchenAbvGr'] = dataframe['KitchenAbvGr'].astype(str)

dataframe['OverallCond'] =  dataframe['OverallCond'].astype(str)

dataframe['YrSold'] =       dataframe['YrSold'].astype(str)

dataframe['MoSold'] =       dataframe['MoSold'].astype(str)
# Agora estamos usando o comando Label Encoder, que transforma variaveis categoricas em numeros



from sklearn.preprocessing import LabelEncoder



for c in ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold'):

    lbl = LabelEncoder() 

    lbl.fit(list(dataframe[c].values)) 

    dataframe[c] = lbl.transform(list(dataframe[c].values))



#como será utilizado algoritmos de regressão, a idéia aqui é criar novas colunas com Get Dummies



dataframe = pd.get_dummies(dataframe)

print(dataframe.shape)

dataframe.head(5)

# Normalizando os valores



dataframe=(dataframe-dataframe.mean())/dataframe.std()

dataframe.head(5)