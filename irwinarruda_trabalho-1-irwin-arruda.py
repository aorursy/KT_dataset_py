#importar bibliotecas de análise e formatação de dados
import numpy as np
import pandas as pd
#importar biblioteca de plotar dados
import matplotlib.pyplot as plt
#bibliotecas de machinelearning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import SGDRegressor 
from sklearn.neural_network import MLPRegressor
#carregar os dados
dataset = pd.read_csv("../input/traffic-volume-in-brazils-highway-toll-plazas/all.csv", 
                      sep=";",
                      decimal=",",
                      encoding="ISO-8859-1")
dataset.head()
dataset["Concessionaria"].unique()
dataset_dutra = dataset[dataset["Concessionaria"] == "01.Nova Dutra"]
dataset_dutra
dataset_dutra["Categoria"].unique()
dataset_dutra_passeio = dataset_dutra[dataset_dutra["Categoria"] == "Categoria 1"]
dataset_dutra_passeio
dataset_dutra_passeio["Praca"].unique()
dataset_dutra_passeio["mes_ano"] = pd.to_datetime(dataset_dutra_passeio["mes_ano"], format="%b-%y")
dataset_dutra_passeio
dataset_agrupado = dataset_dutra_passeio.groupby(["mes_ano"]).sum()
dataset_agrupado
dataset_agrupado["mes"] = dataset_agrupado.index.month + ((dataset_agrupado.index.year - 2010) * 12)
dataset_agrupado
#separando os dados para os processos de machine learning
X = dataset_agrupado[["mes"]]
y = dataset_agrupado[["Volume_total"]]
plt.scatter(X, y)
plt.xlabel("Meses")
plt.ylabel("Volume")
#declarando a função de normalização
escala = StandardScaler()
X_norm = escala.fit_transform(X)
#dividindo os dados em teste e treino
X_norm_train, X_norm_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3)
#ativando a função MLPRegressor
rna = MLPRegressor(hidden_layer_sizes=(600, 300, 200, 100, 50),
                   max_iter=2000,
                   tol=0.0000001,
                   solver="adam",
                   learning_rate_init=0.01,
                   activation="relu",
                   learning_rate="constant",
                   verbose=2)

rna.fit(X_norm_train, y_train)
#ativando a função SGDRegressor
reg_linear = SGDRegressor(max_iter=500, 
                          eta0=0.01, 
                          tol=0.0000001, 
                          learning_rate="constant", 
                          verbose=1)
reg_linear.fit(X_norm_train, y_train) 
#previsão do conjunto de testes
y_reg_pred_test = reg_linear.predict(X_norm_test)
y_rna_pred_test = rna.predict(X_norm_test)
#calcular o r^2
r2_rna = r2_score(y_test, y_rna_pred_test)
r2_reg = r2_score(y_test, y_reg_pred_test)
print("R2 RNA: ", r2_rna)
print("R2 Reg: ", r2_reg)
#verificando se os testes estão coerentes
X_test = escala.inverse_transform(X_norm_test)
plt.scatter(X_test, y_test, alpha=0.5, label="Real")
plt.scatter(X_test, y_rna_pred_test, label="Redes Neurais")
plt.scatter(X_test, y_reg_pred_test, label="Regressão")
plt.xlabel("Mês")
plt.ylabel("Volume")
plt.title("Gráfico de volume por mês")
tamanho_dados_mes = X.size 
#preparando os dados da regressão linear
X_pred_reg = X
y_pred_reg = reg_linear.predict(escala.transform(X_pred_reg))

X_pred_futura_reg = np.array([np.arange(tamanho_dados_mes + 1, tamanho_dados_mes + 25)])
X_pred_futura_reg = X_pred_futura_reg.transpose()
y_pred_futura_reg = reg_linear.predict(escala.transform(X_pred_futura_reg))
#plotando os dados da regressão linear
plt.scatter(X, y, label="Real")
plt.scatter(X_pred_reg, y_pred_reg, label="Aproximação")
plt.scatter(X_pred_futura_reg, y_pred_futura_reg, label="Previsão")
plt.xlabel("mês")
plt.ylabel("Volume")
plt.title("Gráfico de volume por mês (Regressão linear)")
plt.legend(loc="upper left")
plt.savefig("./GraficoVolumeMesREG.png", dpi=300)
#preparando os dados das redes neurais
X_pred_rna = X
y_pred_rna = rna.predict(escala.transform(X_pred_rna))

X_pred_futura_rna = np.array([np.arange(tamanho_dados_mes + 1, tamanho_dados_mes + 25)])
X_pred_futura_rna = X_pred_futura_rna.transpose()
y_pred_futura_rna = rna.predict(escala.transform(X_pred_futura_rna))
#plotando os dados das redes neurais
plt.scatter(X, y, label="Real")
plt.scatter(X_pred_rna, y_pred_rna, label="Aproximação")
plt.scatter(X_pred_futura_rna, y_pred_futura_rna, label="Previsão")
plt.xlabel("mês")
plt.ylabel("Volume")
plt.title("Gráfico de volume por mês (Redes neurais)")
plt.legend(loc="upper left")
plt.savefig("./GraficoVolumeMesRNA.png", dpi=300)