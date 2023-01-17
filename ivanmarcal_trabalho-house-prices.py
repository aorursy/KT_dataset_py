# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importando os dados para um DataFrame

df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/housetrain.csv')
# Verificando o tamanho do DataFrame

print('House Prices: ',  df.shape)
# Verificando as Informações das variáveis

df.info()
# Verificando os primeiros dados

df.head().T
#Renomeando a variável de "1stFlrSF" para "FirstFlrSF"

#Foi necessário renomear porque algumas funções não aceitam campo iniciando com número 

df = df.rename(columns={'1stFlrSF': 'FirstFlrSF'})
# Informações da variável "SalePrice"

df['SalePrice'].describe()
# Histogram da variável SalePrice

df[('SalePrice')].hist(bins=50)
 #Boxplot da variável SalePrice

df[('SalePrice')].plot.box()
# Histogram da variável OverallQual

df[('OverallQual')].hist(bins=20)
# Plot box da variável OverallQual

df[('OverallQual')].plot.box()
# Informações da variável "GrLivArea"

df['GrLivArea'].describe()
# Histogram da variável GrLivArea

df[('GrLivArea')].hist(bins=20)
# Plot box da variável GrLivArea

df[('GrLivArea')].plot.box()
# Informações da variável "GarageCars"

df['GarageCars'].describe()
# Histogram da variável GarageCars

df[('GarageCars')].hist(bins=50)
# Plot box da variável GarageCars

df[('GarageCars')].plot.box()
# Informações da variável "GarageArea"

df['GarageArea'].describe()
# Histogram da variável GarageArea

df[('GarageArea')].hist(bins=50)
# Plot box da variável GarageArea

df[('GarageArea')].plot.box()
# Informações da variável "TotalBsmtSF"

df['TotalBsmtSF'].describe()
# Histogram da variável TotalBsmtSF

df[('TotalBsmtSF')].hist(bins=50)
# Plot box da variável TotalBsmtSF

df[('TotalBsmtSF')].plot.box()
# Informações da variável 'FirstlrSF"

df['FirstFlrSF'].describe()
# Histogram da variável FirstFlrSF

df[('FirstFlrSF')].hist(bins=50)
# Plot box da variável FirstFlrSF

df[('FirstFlrSF')].plot.box()
# Informações da variável "FullBath"

df['FullBath'].describe()
# Histogram da variável FullBath

df[('FullBath')].hist(bins=50)
# Plot box da variável FullBath

df[('FullBath')].plot.box()
# Informações da variável "TotRmsAbvGrd"

df['TotRmsAbvGrd'].describe()
# Histogram da variável TotRmsAbvGrd

df[('TotRmsAbvGrd')].hist(bins=50)
# Plot box da variável TotRmsAbvGrd

df[('TotRmsAbvGrd')].plot.box()
# Informações da variável "Yearbuilt"

df['YearBuilt'].describe()
# Histogram da variável Yearbuilt

df[('YearBuilt')].hist(bins=50)
# Plot box da variável Yearbuilt

df[('YearBuilt')].plot.box()
# Informações da variável "YearRemodAdd"

df['YearRemodAdd'].describe()
# Histogram da variável YearRemodAdd

df[('YearRemodAdd')].hist(bins=50)
# Plot box da variável YearRemodAdd

df[('YearRemodAdd')].plot.box()
# Correlação das variáveis

df.corr()
# Exportando o resultado da correlação para uma planilha

# A intenção é verificar a correlação da variável SalePrice com as demais variáveis

dfCorr = df.corr()

dfCorr.to_excel('correlacao.xlsx',encoding='utf-8',index=False)
# Aumentando o gráfico

f, ax = plt.subplots(figsize=(20,10))



#Plotando a correlação

sns.heatmap(df.corr(), annot = True,fmt='.2f' , linecolor='black', ax=ax, lw=.7)
# box-plot da SalePrice por OverallQual (Qualidade geral do material e acabamento)

# 10   Very Excellent

#   9    Excellent

#   8    Very Good

#   7    Good

#   6    Above Average

#   5    Average

#   4    Below Average

#   3    Fair

#   2    Poor

#   1    Very Poor

sns.boxplot(y='SalePrice', x='OverallQual', data=df)
# Grafico do Violino - box-plot da SalePrice por OverallQual (Qualidade geral do material e acabamento)

sns.violinplot(y='SalePrice', x='OverallQual', width=1.8, data=df)
%matplotlib inline

import scipy as sp

import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
#SalePrice OverallQual YearBuilt

reg = sm.ols(formula='SalePrice~OverallQual',data=df).fit()

print(reg.summary())
SalePrice_hat = reg.predict()

res = df['SalePrice'] - SalePrice_hat

plt.hist(res, color='blue', bins=15)

plt.title('Histograma dos resíduos da regressão')

plt.show()

#SalePrice OverallQual YearBuilt

reg = sm.ols(formula='SalePrice~OverallQual+GrLivArea',data=df).fit()

#reg = sm.ols(formula='SalePrice~OverallQual+GrLivArea+GarageCars+GarageArea+TotalBsmtSF+FirstFlrSF+FullBath+TotRmsAbvGrd+YearBuilt+YearRemodAdd',data=df).fit()

#reg = sm.ols(formula='SalePrice~OverallQual+GrLivArea+GarageCars+GarageArea+TotalBsmtSF+FirstFlrSF+FullBath+TotRmsAbvGrd+YearBuilt+YearRemodAdd',data=df).fit()

#reg = sm.ols(formula='SalePrice~OverallQual+GrLivArea+GarageCars+GarageArea+TotalBsmtSF+FirstFlrSF+FullBath+TotRmsAbvGrd+YearBuilt+YearRemodAdd',data=df).fit()

#reg = sm.ols(formula='SalePrice~OverallQual+GrLivArea+GarageCars+GarageArea+TotalBsmtSF+FirstFlrSF+FullBath+TotRmsAbvGrd+YearBuilt+YearRemodAdd',data=df).fit()

print(reg.summary())
SalePrice_hat = reg.predict()

res = df['SalePrice'] - SalePrice_hat

plt.hist(res, color='blue', bins=15)

plt.title('Histograma dos resíduos da regressão')

plt.show()

#SalePrice OverallQual YearBuilt

#reg = sm.ols(formula='SalePrice~OverallQual+GrLivArea+GarageCars',data=df).fit()

reg = sm.ols(formula='SalePrice~OverallQual+GrLivArea+GarageCars+GarageArea',data=df).fit()

#reg = sm.ols(formula='SalePrice~OverallQual+GrLivArea+GarageCars+GarageArea+TotalBsmtSF+FirstFlrSF+FullBath+TotRmsAbvGrd+YearBuilt+YearRemodAdd',data=df).fit()

#reg = sm.ols(formula='SalePrice~OverallQual+GrLivArea+GarageCars+GarageArea+TotalBsmtSF+FirstFlrSF+FullBath+TotRmsAbvGrd+YearBuilt+YearRemodAdd',data=df).fit()

print(reg.summary())
SalePrice_hat = reg.predict()

res = df['SalePrice'] - SalePrice_hat

plt.hist(res, color='blue', bins=15)

plt.title('Histograma dos resíduos da regressão')

plt.show()
#SalePrice OverallQual YearBuilt

reg = sm.ols(formula='SalePrice~OverallQual+GrLivArea+GarageCars+GarageArea+TotalBsmtSF+FirstFlrSF+FullBath+TotRmsAbvGrd+YearBuilt+YearRemodAdd',data=df).fit()

print(reg.summary())
coefs = pd.DataFrame(reg.params)

coefs.columns = ['Coeficientes']

print(coefs)
