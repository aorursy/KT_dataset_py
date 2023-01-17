# Import libs

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from matplotlib.pyplot import figure

import seaborn as sns

from matplotlib.colors import ListedColormap

import os

#print(os.listdir("../input"))

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_moons, make_circles, make_classification



from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn import linear_model, svm, gaussian_process

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import accuracy_score
#Import datasets

df_test_original = pd.read_csv('../input/test.csv')

df_train_original = pd.read_csv('../input/train.csv')
# Create a new dataframe for manipulation of datas

df_train = df_train_original

df_test = df_test_original
# Drop columns, except Alley.



#df_train = df_train.drop(['Alley'], axis=1)

#df_test = df_test.drop(['Alley'], axis=1)



df_train = df_train.drop(['PoolQC'], axis=1)

df_test = df_test.drop(['PoolQC'], axis=1)



df_train = df_train.drop(['MiscFeature'], axis=1)

df_test = df_test.drop(['MiscFeature'], axis=1)



df_train = df_train.drop(['Fence'], axis=1)

df_test = df_test.drop(['Fence'], axis=1)



df_train = df_train.drop(['FireplaceQu'], axis=1)

df_test = df_test.drop(['FireplaceQu'], axis=1)
# View column by column to better analysis to datas

df_train['BsmtQual'].count() #1423 não nulos

df_train['BsmtCond'].count() #1423 não nulos

df_train['BsmtExposure'].count() #1422 não nulos

df_train['BsmtFinType1'].count() #1423 não nulos

df_train['BsmtFinSF1'].count() #all

df_train['BsmtFinType2'].count() #1422 não nulos

df_train['BsmtFinSF2'].count() #all

df_train['BsmtUnfSF'].count() #all

df_train['TotalBsmtSF'].count() #all

df_train['Heating'].count() #all

df_train['HeatingQC'].count() #all

df_train['CentralAir'].count() #all

df_train['Electrical'].count() #1459 não nulos

df_train['1stFlrSF'].count() #all

df_train['2ndFlrSF'].count() #all

df_train['LowQualFinSF'].count() #all

df_train['GrLivArea'].count() #all
df_train = df_train.drop(['SalePrice'], axis=1)
# Create a new dataframe

all_data = pd.concat((df_train.loc[:,'Id':'SaleCondition'],

                     df_test.loc[:,'Id':'SaleCondition']))
# Replace all values null



all_data['BsmtQual'] = all_data['BsmtQual'].fillna('')

all_data['BsmtCond'] = all_data['BsmtCond'].fillna('')

all_data['BsmtExposure'] = all_data['BsmtExposure'].fillna('')

all_data['BsmtFinType1'] = all_data['BsmtFinType1'].fillna('')

all_data['BsmtFinSF1'] = all_data['BsmtFinSF1'].fillna('0')

all_data['BsmtFinType2'] = all_data['BsmtFinType2'].fillna('')

all_data['BsmtFinSF2'] = all_data['BsmtFinSF2'].fillna('0')

all_data['BsmtUnfSF'] = all_data['BsmtUnfSF'].fillna('0')

all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna('0')

all_data['Heating'] = all_data['Heating'].fillna('')

all_data['HeatingQC'] = all_data['HeatingQC'].fillna('')

all_data['CentralAir'] = all_data['CentralAir'].fillna('')

all_data['Electrical'] = all_data['Electrical'].fillna('')

all_data['1stFlrSF'] = all_data['1stFlrSF'].fillna('0')

all_data['2ndFlrSF'] = all_data['2ndFlrSF'].fillna('0')

all_data['LowQualFinSF'] = all_data['LowQualFinSF'].fillna('0')

all_data['GrLivArea'] = all_data['GrLivArea'].fillna('0')

all_data['GarageType'] = all_data['GarageType'].fillna('')

all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna('0')

all_data['GarageFinish'] = all_data['GarageFinish'].fillna('')

all_data['GarageQual'] = all_data['GarageQual'].fillna('')

all_data['GarageCond'] = all_data['GarageCond'].fillna('')

all_data['MSZoning'] = all_data['MSZoning'].fillna('')

all_data['MSZoning'] = all_data['MSZoning'].fillna('')

all_data['LotFrontage'] = all_data['LotFrontage'].fillna('0')

all_data['Utilities'] = all_data['Utilities'].fillna('')

all_data['Exterior1st'] = all_data['Exterior1st'].fillna('')

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna('')

all_data['MasVnrType'] = all_data['MasVnrType'].fillna('')

all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna('0')

all_data['BsmtFullBath'] = all_data['BsmtFullBath'].fillna('0')

all_data['BsmtHalfBath'] = all_data['BsmtHalfBath'].fillna('0')

all_data['KitchenQual'] = all_data['KitchenQual'].fillna('')

all_data['Functional'] = all_data['Functional'].fillna('')

all_data['GarageType'] = all_data['GarageType'].fillna('')

all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna('0')

all_data['GarageCars'] = all_data['GarageCars'].fillna('0')

all_data['GarageArea'] = all_data['GarageArea'].fillna('0')

all_data['SaleType'] = all_data['SaleType'].fillna('')

all_data['Alley'] = all_data['Alley'].fillna('')
all_data.count()
#Get_Dummies  (para transformar categoricos em Numéricos)

all_data = pd.get_dummies(all_data)
all_data



#Aqui percebi que poderia analisar coluna por coluna para atribuir a melhor solução na substituição dos valores nulos.

#Talvez a melhor abordagem para alguns campos fosse adicionar a média
all_data = all_data.fillna(all_data.mean())
# Creating matrices for sklearn



#Cria Matriz X_train utilizando a Matriz com todos os dados all_data: do inicio da matriz (:) até o fim da matriz df_train.shape[0]¶

X_train = all_data[:df_train.shape[0]]
#Cria Matriz X_test utilizando a Matriz com todos os dados all_data: a partir do último registro matriz df_train.shape[0], ou seja, todos os registros que não estiverem em df_train

X_test = all_data[df_train.shape[0]:]
# Criando o y, ou seja, o que será previsto, apenas com o campo "SalePrice"

y = df_train_original.SalePrice
# 2) Aplly Gradient Boost Model

from sklearn.ensemble import GradientBoostingRegressor

import statsmodels.formula.api as smf

from sklearn.preprocessing import scale

gbr = GradientBoostingRegressor()



gbr.fit(X_train, y)
yhat_Train = gbr.predict(X_train)
yhat_Train
yhat_test = gbr.predict(X_test)
yhat_test
yhat_rounded = [round(x,ndigits=None) for x in yhat_test]

yhat_rounded = [int(x) for x in yhat_rounded]



yhat_gbr = yhat_rounded

print ('# # # # Esse é o yhat com o método Gradiente Descendente # # # #')

print ('# # # # Ou seja, a previsão se Esse é o yhat com o método Gradiente Descendente # # # #')

print (yhat_gbr)
# Gerando CSV

df_test_gbr = df_test

df_test_gbr['SalePrice'] = yhat_gbr

df_test_gbr = df_test_gbr.drop(df_test_gbr.columns[1:76], axis=1)

df_test_gbr.to_csv('HousePrice_GBR.csv', index = False)
df_test_gbr
# 3) Aplly Logistic Regression Model

from sklearn.linear_model import LogisticRegression



# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, y)
yhat_log = logreg.predict(X_test)
yhat_log
# Gerando CSV Logistic Regression Model

df_test_logreg = df_test

df_test_logreg['SalePrice'] = yhat_log

df_test_logreg = df_test_logreg.drop(df_test_logreg.columns[1:76], axis=1)

df_test_logreg.to_csv('HousePrice_LOG.csv', index = False)
df_test_logreg
# 5) Aplly XGBOOST Model

from xgboost import XGBClassifier
xgb = XGBClassifier()

xgb.fit(X_train, y)
yhat_xgb = xgb.predict(X_test)
yhat_xgb
# Gerando CSV XGBOOST Model

df_test_xgb = df_test

df_test_xgb['SalePrice'] = yhat_xgb

df_test_xgb = df_test_xgb.drop(df_test_xgb.columns[1:76], axis=1)

df_test_xgb.to_csv('HousePrice_XGB.csv', index = False)
# Aplly SVC Model

svc = SVC(probability=True)

svc.fit(X_train, y)
svc_test = svc.predict(X_test)
svc_test
# Gerando CSV SVC Model

df_test_svc = df_test

df_test_svc['SalePrice'] = svc_test

df_test_svc = df_test_svc.drop(df_test_svc.columns[1:76], axis=1)

df_test_svc.to_csv('HousePrice_SVC.csv', index = False)
# Aplly Decision Tree Model

dtc = DecisionTreeClassifier()

dtc.fit(X_train, y)
dtc_test = dtc.predict(X_test)
dtc_test
# Gerando CSV DTC Model

df_test_dtc = df_test

df_test_dtc['SalePrice'] = dtc_test

df_test_dtc = df_test_dtc.drop(df_test_dtc.columns[1:76], axis=1)

df_test_dtc.to_csv('HousePrice_DTC.csv', index = False)
# Aplly GaussianNB Model

gnb = GaussianNB()

gnb.fit(X_train, y)
gnb_test = gnb.predict(X_test)
gnb_test
# Gerando CSV GaussianNB Model

df_test_gnb = df_test

df_test_gnb['SalePrice'] = gnb_test

df_test_gnb = df_test_gnb.drop(df_test_gnb.columns[1:76], axis=1)

df_test_gnb.to_csv('HousePrice_GNB.csv', index = False)
# Aplly Neural Model

nn = MLPClassifier(hidden_layer_sizes=(100,100,50))

nn.fit(X_train, y)
nn_test = nn.predict(X_test)
nn_test
# Gerando CSV Neural Model

df_test_nn = df_test

df_test_nn['SalePrice'] = nn_test

df_test_nn = df_test_nn.drop(df_test_nn.columns[1:76], axis=1)

df_test_nn.to_csv('HousePrice_NN.csv', index = False)
# 6) Aplly KNeighbors Model

knn = KNeighborsClassifier()

knn.fit(X_train, y)
knn_test = knn.predict(X_test)
knn_test
# Gerando CSV KNeighbors Model

df_test_knn = df_test

df_test_knn['SalePrice'] = knn_test

df_test_knn = df_test_knn.drop(df_test_knn.columns[1:76], axis=1)

df_test_knn.to_csv('HousePrice_KNN.csv', index = False)