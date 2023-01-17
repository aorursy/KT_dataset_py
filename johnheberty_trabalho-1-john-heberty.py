# BIBLIOTECAS BÁSICAS
import numpy as np
import pandas as pd
# BIBLIOTECAS DE GRÁFICOS
import matplotlib.pyplot as plt
# BIBLIOTECAS DE MACHINE LEARNING
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import explained_variance_score
dataset = pd.read_csv("../input/traffic-volume-in-brazils-highway-toll-plazas/all.csv",
                     sep=";",
                     decimal=",",
                     encoding="ISO-8859-1")
dataset.head()
dataset["Concessionaria"].unique()
dataset_dutra = dataset[dataset["Concessionaria"] == "01.Nova Dutra"]
dataset_dutra["Categoria"].unique()
dataset_dutra_passeio = dataset_dutra[dataset_dutra["Categoria"] == "Categoria 1"]
dataset_dutra_passeio.head()
dataset_dutra_passeio["Praca"].unique()
dataset_dutra_passeio["mes_ano"] = pd.to_datetime(dataset_dutra_passeio["mes_ano"],
                                              format="%b-%y")
dataset_dutra_passeio["mes_ano"]
dataset_agrupado = dataset_dutra_passeio.groupby(["mes_ano"]).sum()
dataset_agrupado
dataset_agrupado["mes"] = dataset_agrupado.index.month * (dataset_agrupado.index.year - 2009)
dataset_agrupado["mes"]
dataset_agrupado
dataset_agrupado["mes"] = np.arange(1, 121)

x = dataset_agrupado[["mes"]]
y = dataset_agrupado[["Volume_total"]]

plt.scatter(x,y)
plt.xlabel("Meses")
plt.ylabel("Volume")
x = dataset_agrupado[["mes"]]
y = dataset_agrupado[["Volume_total"]]

Intervalo = np.array([np.arange(121, 145)]) 
X_futuro = Intervalo.transpose()
escala = StandardScaler()
escala.fit(x)

X_norm = escala.transform(x)
X_norm_train, X_norm_test, y_train, y_test = train_test_split(X_norm,y,test_size=0.4) # DIVIDINDO OS DADOS
"""
reglinear = MLPRegressor(hidden_layer_sizes=(700,500,300,100),
                         max_iter=2000, 
                         verbose=True) # MODELO

reglinear.fit(X_norm_train, y_train) # TREINANDO MODELO
"""
reglinear = MLPRegressor(hidden_layer_sizes=(700,500,300,100),
                         max_iter=1500, 
                         verbose=True) # MODELO

reglinear.fit(X_norm_train, y_train) # TREINANDO MODELO
y_pred = reglinear.predict(X_norm_test)

plt.scatter(X_norm_test, y_test)
plt.scatter(X_norm_test, y_pred)
np.array(y_test)
y_pred
y_test_ = np.array(list(map(lambda z: z[0],np.array(y_test))))

y_pred_ = np.array(y_pred)
explained_variance_score(y_test_, y_pred, multioutput='uniform_average') # QUANTO MAIS PERTO DE 1 É UMA BOA FUNÇÃO DE REGREÇÃO
# Ultimo ano convertido era 120 em meses, logo somamos mais 24 para 2 anos no futuro
y_pred_2021 = reglinear.predict(X_futuro)
y_pred_2021 = int(round(y_pred_2021[0],0))

# ADICIONANDO PONTOS NO NUMERO PARA FACILITAR LEITURA
n_num = str(y_pred_2021)
y_pred_2021_ = [n_num[0:z]+"."+n_num[z:] for z in list(range(0,len(n_num),3))[::-1]][0:-1][::-1]
y_pred_2021_corrigido = y_pred_2021_[0]
for z in y_pred_2021_[1:]:
    corte_ponto = z.split('.')[-1]
    y_pred_2021_corrigido = y_pred_2021_corrigido[0:-len(corte_ponto)] + "." + corte_ponto

print("Previsão para 2021 teremos um tráfego de {} véiculos de categoria 1".format(y_pred_2021_corrigido))
