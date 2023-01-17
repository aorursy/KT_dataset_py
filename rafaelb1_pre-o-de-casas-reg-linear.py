import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use("seaborn-muted")

%matplotlib inline
treino = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

teste = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
treino.head()
treino.info()
print("Dimensão dos dados de treino")



treino.shape
print("Dimensão dos dados de teste")



teste.shape
media_sales_price = treino["SalePrice"].mean()

var_sales_price = treino["SalePrice"].var()



plt.figure(figsize = [10, 6])

sns.distplot(treino["SalePrice"], color = "orangered", hist = True, label = "Histograma e\n Densidade")

plt.axvline(media_sales_price, color = 'k', linestyle = 'dashed', linewidth = 1, label = "Média")

plt.xlabel("Preço de venda ($)", fontsize = 14, color = "black")

plt.ylabel("Densidade", fontsize = 14, color = "black")

plt.title("Distribuição do Preço das Vendas", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")

plt.legend()

plt.show()
treino["SalePrice"].describe()
miss_val_treino = treino.isnull().sum()



dados_miss_val_treino = pd.DataFrame(miss_val_treino)



dados_miss_val_treino = dados_miss_val_treino.reset_index()



dados_miss_val_treino.columns = ["Variável", "Quantidade"]



plt.figure(figsize = [10, 6])

plt.barh(dados_miss_val_treino["Variável"], dados_miss_val_treino["Quantidade"], align = "center", color = "orangered")

plt.xlabel("Quantidade de valores faltantes", fontsize = 14, color = "black")

plt.ylabel("Variável", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelcolor = "black")

plt.tick_params(axis = "y", labelcolor = "black")

plt.show()
dados_miss_val_treino0 = dados_miss_val_treino[dados_miss_val_treino["Quantidade"] > 0].sort_values(by = "Quantidade")



plt.figure(figsize = [10, 6])

plt.barh(dados_miss_val_treino0["Variável"], dados_miss_val_treino0["Quantidade"], align = "center", color = "orangered")

plt.xlabel("Quantidade de valores faltantes", fontsize = 14, color = "black")

plt.ylabel("Variável", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")

plt.show()
treino["PoolQC"].value_counts()
treino["MiscFeature"].value_counts()
treino["Alley"].value_counts()
treino["Fence"].value_counts()
plt.figure(figsize = [10, 6])



#---



sns.boxplot(x = "MSZoning", y = "SalePrice", data = treino, palette = "Set3")

plt.xlabel("Zona da casa (Densidade)", fontsize = 14, color = "black")

plt.ylabel("Preço de venda ($)", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")





#---



plt.show()
plt.figure(figsize = [15, 8])



plt.subplot(1, 2, 1)



sns.boxplot(x = "OverallQual", y = "SalePrice", data = treino, palette = "Set3")

plt.xlabel("Avaliação do material da residência", fontsize = 14, color = "black")

plt.ylabel("Preço de venda ($)", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")



#---



plt.subplot(1, 2, 2)





sns.boxplot(x = "OverallCond", y = "SalePrice", data = treino, palette = "Set3")

plt.xlabel("Avaliação geral da residência", fontsize = 14, color = "black")

plt.ylabel("Preço de venda ($)", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")



#---



plt.show()
var_quant = treino._get_numeric_data()

var_quant = var_quant.drop(["Id", "MSSubClass", "OverallQual", "OverallCond"], axis = 1)



var_quant.head()
corr_var_quant = var_quant.corr()



plt.figure(figsize = [10, 6])

sns.heatmap(data = corr_var_quant, vmin = -1, vmax = 1, linewidths = 0.01, linecolor = "black", cmap = "BrBG")

plt.xlabel("", fontsize = 14, color = "black")

plt.ylabel("", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")

plt.show()
corr_var_quant["SalePrice"].round(2)
novo_banco = treino[["YearBuilt", "YearRemodAdd", "TotalBsmtSF", "1stFlrSF", "GrLivArea", "FullBath", "TotRmsAbvGrd", "GarageCars", "GarageArea", "SalePrice"]]

corr_novo_banco = novo_banco.corr()



plt.figure(figsize = [10, 6])

sns.heatmap(data = corr_novo_banco, vmin = -1, vmax = 1, linewidths = 0.01, linecolor = "black", cmap = "BrBG")

plt.xlabel("", fontsize = 14, color = "black")

plt.ylabel("", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")

plt.show()
corr_novo_banco["GarageCars"].round(2)
novo_banco = novo_banco.drop(["GarageCars"], axis = 1)



novo_banco.head()
plt.figure(figsize = [15, 9])



plt.subplot(4, 2, 1)



plt.scatter(treino["SalePrice"], treino["YearBuilt"])

plt.xlabel("Preço de venda ($)", fontsize = 14, color = "black")

plt.ylabel("YearBuilt", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")



plt.subplot(4, 2, 2)





plt.scatter(treino["SalePrice"], treino["YearRemodAdd"])

plt.xlabel("Preço de venda ($)", fontsize = 14, color = "black")

plt.ylabel("YearRemodAdd", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")





plt.subplot(4, 2, 3)





plt.scatter(treino["SalePrice"], treino["TotalBsmtSF"])

plt.xlabel("Preço de venda ($)", fontsize = 14, color = "black")

plt.ylabel("TotalBsmtSF", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")





plt.subplot(4, 2, 4)



plt.scatter(treino["SalePrice"], treino["1stFlrSF"])

plt.xlabel("Preço de venda ($)", fontsize = 14, color = "black")

plt.ylabel("1stFlrSF", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")





plt.subplot(4, 2, 5)



plt.scatter(treino["SalePrice"], treino["GrLivArea"])

plt.xlabel("Preço de venda ($)", fontsize = 14, color = "black")

plt.ylabel("GrLivArea", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")





plt.subplot(4, 2, 6)



plt.scatter(treino["SalePrice"], treino["FullBath"])

plt.xlabel("Preço de venda ($)", fontsize = 14, color = "black")

plt.ylabel("FullBath", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")





plt.subplot(4, 2, 7)



plt.scatter(treino["SalePrice"], treino["TotRmsAbvGrd"])

plt.xlabel("Preço de venda ($)", fontsize = 14, color = "black")

plt.ylabel("TotRmsAbvGrd", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")





plt.subplot(4, 2, 8)



plt.scatter(treino["SalePrice"], treino["GarageArea"])

plt.xlabel("Preço de venda ($)", fontsize = 14, color = "black")

plt.ylabel("GarageArea", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")





plt.show()
treino["SalePrice"] = np.log(treino["SalePrice"])



media_sales_price = treino["SalePrice"].mean()

var_sales_price = treino["SalePrice"].var()



plt.figure(figsize = [10, 6])

sns.distplot(treino["SalePrice"], color = "orangered", hist = True, label = "Histograma e\n Densidade")

plt.axvline(media_sales_price, color = 'k', linestyle = 'dashed', linewidth = 1, label = "Média")

plt.xlabel("Preço de venda ($)", fontsize = 14, color = "black")

plt.ylabel("Densidade", fontsize = 14, color = "black")

plt.title("Distribuição do Preço das Vendas", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")

plt.legend()

plt.show()
var_venda = treino["SalePrice"]



var_venda
treino.drop(["SalePrice"], axis = 1, inplace = True)



treino.head()
treino_index = treino.shape[0]



treino_index
teste_index = teste.shape[0]



teste_index
banco_geral = pd.concat(objs = [treino, teste], axis = 0).reset_index(drop = True)



banco_geral.shape
banco_geral_quant = banco_geral[["YearBuilt", "YearRemodAdd", "TotalBsmtSF", "1stFlrSF", "GrLivArea", "FullBath", "TotRmsAbvGrd", "GarageArea"]]



banco_geral_quant.head()
banco_geral["MSSubClass"] = banco_geral["MSSubClass"].astype(str)

banco_geral["OverallQual"] = banco_geral["OverallQual"].astype(str)

banco_geral["OverallCond"] = banco_geral["OverallCond"].astype(str)
banco_geral_qualit = banco_geral.select_dtypes("object")



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
teste.head()
from sklearn.model_selection import train_test_split



x_treino, x_valid, y_treino, y_valid = train_test_split(treino1.drop("SalePrice", axis = 1), treino1["SalePrice"], train_size = 0.7, random_state = 1234)
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

from sklearn.metrics import mean_squared_error

import statsmodels.api as sm



reg = LinearRegression()



reg.fit(X = x_treino, y = y_treino)
x_treino1 = sm.add_constant(x_treino)

reg1 = sm.OLS(y_treino, x_treino1).fit()

print(reg1.summary())
r2_valid = reg.score(x_valid, y_valid)



print("O R^2 nos dados de validação foi de:", r2_valid)
previsoes = reg.predict(x_valid)



previsoes[:6]
from sklearn.metrics import mean_squared_error





# print ('RMSE is: \n', mean_squared_error(y_test, predictions))



rmse = mean_squared_error(y_valid, previsoes)



print("O modelo obteve RMSE de:", rmse, "nos dados de teste.")
plt.figure(figsize = [10, 6])



plt.scatter(previsoes, y_valid, alpha=.7, color='b')

plt.xlabel("Preço predito", fontsize = 14, color = "black")

plt.ylabel("Preço atual", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")



plt.show()
#--- Como usamos o log para diminuir a variância, usamos a operação inversa dele (exponencial) para voltar aos números iniciais



previsoes_teste_exp = np.exp(reg.predict(teste1))



#--- Selecionando o ID



id_teste = teste["Id"]



id_teste
subm = pd.DataFrame()



subm["Id"] = id_teste

subm["SalePrice"] = previsoes_teste_exp
subm.head()
subm.to_csv("subm1.csv", index = False)