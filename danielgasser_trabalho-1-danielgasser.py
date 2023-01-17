#Carregando as bibliotecas básicas:
import numpy as np
import pandas as pd 
#Carregando as bibliotecas de geração de gráficos
import matplotlib.pyplot as plt
#Carregando as bibliotecas de machine learning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
dataset = pd.read_csv("../input/traffic-volume-in-brazils-highway-toll-plazas/all.csv",
                     sep=";",
                     decimal=",",
                     encoding="ISO-8859-1")
dataset.head()
dataset["Concessionaria"].unique()

dataset_dutra = dataset[dataset["Concessionaria"] == "01.Nova Dutra"]
dataset_dutra["Categoria"].unique()
dataset_dutra_carpasseio = dataset_dutra[dataset_dutra["Categoria"] == "Categoria 1"]
dataset_dutra_carpasseio["Praca"].unique()
dataset_dutra_carpasseio["mes_ano"] = pd.to_datetime(dataset_dutra_carpasseio["mes_ano"],
                                                    format = "%b-%y")
dataset_agrupado = dataset_dutra_carpasseio.groupby(["mes_ano"]).sum()
dataset_agrupado.head()
dataset_agrupado["mes"] =  np.arange(1, 121)

dataset_agrupado["mes"]
dataset_agrupado
X = dataset_agrupado[["mes"]]
Y = dataset_agrupado[["Volume_total"]]
plt.scatter(X,Y)
plt.xlabel("Meses")
plt.ylabel("Volume")
X = dataset_agrupado[["mes"]]
Y = dataset_agrupado[["Volume_total"]]
#Normalização
escala = StandardScaler()
escala.fit(X)

X_norm = escala.transform(X)
#Dividindo em treinamento e em teste
X_norm_train, X_norm_test, Y_train, Y_test = train_test_split(X_norm, Y, test_size=0.3)
RNA = MLPRegressor(hidden_layer_sizes=(700, 400, 200, 100, 50),
                   max_iter=2000,
                   tol=0.0000001,
                   solver="adam",
                   learning_rate_init=0.01,
                   activation="relu",
                   learning_rate="constant",
                   verbose=2)
RNA.fit(X_norm_train, Y_train)
reglin = SGDRegressor(max_iter=500,
                     tol=0.0000001,
                     eta0=0.01,
                     learning_rate="constant",
                     verbose=2,
                     )
reglin.fit(X_norm_train, Y_train)
#previsão do conjunto de teste:

Y_RNA_previsao = RNA.predict(X_norm_test)
Y_reglin_previsao = reglin.predict(X_norm_test)
#Calculo do R^2

r2_RNA = r2_score(Y_test,Y_RNA_previsao)
r2_reglin = r2_score(Y_test, Y_reglin_previsao)

print("R2 RNA:", r2_RNA),
print("R2 REGLIN:", r2_reglin)


X_test = escala.inverse_transform(X_norm_test)

plt.scatter(X_test, Y_test, alpha=0.5,label="Regressão")
plt.scatter(X_test, Y_RNA_previsao, alpha=0.5,label="MLP")
plt.scatter(X_test, Y_reglin_previsao, alpha=0.5,label="Reg Linear")
plt.xlabel("Mes")
plt.ylabel("Volume")
plt.title("Volume de trafego/mes")
plt.legend(loc=1)
#prever para mais 24 meses.
X_previsao_reglin = X
y_previsao_reglin = reglin.predict(escala.transform(X_previsao_reglin))

X_previsao_futura_reglin = np.array([np.arange(121,145)])
X_previsao_futura_reglin = X_previsao_futura_reglin.transpose()
Y_previsao_futura_reglin = reglin.predict(escala.transform(X_previsao_futura_reglin))


#regressão linear
plt.scatter(X, Y, label="Real")
plt.scatter(X_previsao_reglin, y_previsao_reglin, label="Aproximação")
plt.scatter(X_previsao_futura_reglin, Y_previsao_futura_reglin, label="Previsão")
plt.xlabel("mes")
plt.ylabel("Volume")
plt.title("Volume de trafego por mês (Regressão linear)")
plt.legend(loc="best")

#Dados Redes Neurais
X_previsao_RNA = X
Y_previsao_RNA = RNA.predict(escala.transform(X_previsao_RNA))

X_previsao_futura_RNA = np.array([np.arange(121.145)])
X_previsao_futura_RNA = X_previsao_futura_RNA.transpose()
Y_previsao_futura_RNA = RNA.predict(escala.transform(X_previsao_futura_RNA))

#redes neurais
plt.scatter(X, Y, label="Real")
plt.scatter(X_previsao_RNA, Y_previsao_RNA, label="Aproximação")
plt.scatter(X_previsao_futura_RNA, Y_previsao_futura_RNA, label="Previsão")
plt.xlabel("mes")
plt.ylabel("Volume")
plt.title("Volume de trafego por mês (Redes neurais)")
plt.legend(loc="best")