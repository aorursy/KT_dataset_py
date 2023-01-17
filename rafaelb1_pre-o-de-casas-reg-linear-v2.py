import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import pipeline



plt.style.use("seaborn-muted")

%matplotlib inline
treino = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

teste = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
treino["SalePrice"] = np.log(treino["SalePrice"])



var_venda = treino["SalePrice"]
treino.drop(["SalePrice"], axis = 1, inplace = True)
treino_index = treino.shape[0]
teste_index = teste.shape[0]
banco_geral = pd.concat(objs = [treino, teste], axis = 0).reset_index(drop = True)



banco_geral
miss_val = banco_geral.isnull().sum().sort_values(ascending = False)



dados_miss_val = pd.DataFrame(miss_val)



dados_miss_val = dados_miss_val.reset_index()



dados_miss_val.columns = ["Variável", "Quantidade"]



dados_miss_val = dados_miss_val[dados_miss_val["Quantidade"] > 0].sort_values(by = "Quantidade")



#-------------------------



plt.figure(figsize = [10, 6])

plt.barh(dados_miss_val["Variável"], dados_miss_val["Quantidade"], align = "center", color = "orangered")

plt.xlabel("Quantidade de valores faltantes", fontsize = 14, color = "black")

plt.ylabel("Variável", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelcolor = "black")

plt.tick_params(axis = "y", labelcolor = "black")

plt.show()
banco_geral = banco_geral.drop(["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"], axis = 1)



banco_geral.head()
banco_geral_quant = banco_geral[["YearBuilt", "YearRemodAdd", "TotalBsmtSF", "1stFlrSF", "GrLivArea", "FullBath", "TotRmsAbvGrd", "GarageArea"]]



banco_geral_quant.head()
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



banco_geral_quant[["TotalBsmtSF", "1stFlrSF", "GrLivArea", "GarageArea"]] = scaler.fit_transform(banco_geral_quant[["TotalBsmtSF", "1stFlrSF", "GrLivArea", "GarageArea"]])



banco_geral_quant.head()
banco_geral["MSSubClass"] = banco_geral["MSSubClass"].astype(str)

banco_geral["OverallQual"] = banco_geral["OverallQual"].astype(str)

banco_geral["OverallCond"] = banco_geral["OverallCond"].astype(str)
banco_geral_qualit = banco_geral.select_dtypes("object")



banco_geral_qualit.head()
banco_geral_qualit = banco_geral_qualit[["LotShape", "LandContour", "LandSlope", "LotConfig", "HouseStyle", "HeatingQC", "OverallQual"]]



banco_geral_qualit.head()
banco_geral_qualit = pd.get_dummies(banco_geral_qualit)



banco_geral_qualit.head()
banco_geral1 = pd.concat([banco_geral_quant, banco_geral_qualit], axis = 1)



banco_geral1.head()
treino1 = banco_geral1.iloc[:treino_index]



print("Dimensões do novos dados de treino")



treino1.shape
teste1 = banco_geral1.iloc[:teste_index]



print("Dimensões do novos dados de teste")



teste1.shape
treino1 = treino1.assign(SalePrice = var_venda)



treino1.head()
from sklearn.model_selection import train_test_split



x_treino, x_valid, y_treino, y_valid = train_test_split(treino1.drop("SalePrice", axis = 1), treino1["SalePrice"], train_size = 0.75, random_state = 1234)



print("Os dados de treino possui dimensões:", treino1.shape)

print("---")

print("x_treino possui dimensões:", x_treino.shape)

print("---")

print("y_treino possui dimensões:", y_treino.shape)

print("---")

print("x_valid possui dimensões:", x_valid.shape)

print("---")

print("y_valid possui dimensões:", y_valid.shape)
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm



reg = LinearRegression()



reg.fit(X = x_treino, y = y_treino)
r2_valid = reg.score(x_valid, y_valid)



print("O R^2 nos dados de validação foi de:", r2_valid)
previsoes = reg.predict(x_valid)



previsoes[:6]
from sklearn.metrics import mean_squared_error

from math import sqrt



rmse = sqrt(mean_squared_error(y_valid, previsoes))



print("O modelo obteve RMSE de:", rmse, "nos dados de teste.")
plt.figure(figsize = [10, 6])



plt.scatter(previsoes, y_valid, alpha=.7, color='b')

plt.xlabel("Preço predito", fontsize = 14, color = "black")

plt.ylabel("Preço atual", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")



plt.show()
previsoes_teste_exp = np.exp(reg.predict(teste1))



#--- Selecionando o ID



id_teste = teste["Id"]



subm = pd.DataFrame()



subm["Id"] = id_teste

subm["SalePrice"] = previsoes_teste_exp



subm.to_csv("subm3.csv", index = False)



subm.head()