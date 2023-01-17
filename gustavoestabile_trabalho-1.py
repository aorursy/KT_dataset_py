#importar as bibliotecas basicas
import numpy as np
import pandas as pd
#importar as bibliotecas de geração de grqaficos
import matplotlib.pyplot as plt
#importar as bibliotecas de Machine learning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
dataset_dutra_passeio = dataset_dutra[dataset_dutra["Categoria"] == "Categoria 1"]
dataset_dutra_passeio["Praca"].unique()
dataset_dutra_passeio["mes_ano"] = pd.to_datetime(dataset_dutra_passeio["mes_ano"],
                                                 format="%b-%y")
dataset_agrupado = dataset_dutra_passeio.groupby(["mes_ano"]).sum()
dataset_agrupado.head()
dataset_agrupado["mes"] = np.arange(1, 121)
dataset_agrupado["mes"]
dataset_agrupado
X = dataset_agrupado[["mes"]]
Y = dataset_agrupado[["Volume_total"]]
plt.scatter(X,Y)
plt.xlabel("Mes")
plt.ylabel("Volume")
X = dataset_agrupado[["mes"]]
y = dataset_agrupado[["Volume_total"]]
escala = StandardScaler()
X_norm = escala.fit_transform(X)
X_norm_train, X_norm_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.4)
reglinear = MLPRegressor(hidden_layer_sizes=(700,500,300,100),
                         max_iter=1500, 
                         learning_rate="constant") 
reglinear.fit(X_norm_train, y_train)

y_pred = reglinear.predict(X_norm_test)
plt.scatter(X_norm_test, y_test)
plt.scatter(X_norm_test, y_pred)
intervalo = np.array([np.arange(121,145)])
Xfuturo= intervalo.transpose()
Xfuturo_normalizado = escala.transform(Xfuturo)
Xfuturo_normalizado
y_futuro=reglinear.predict(Xfuturo_normalizado)
y_futuro
y_dataset=reglinear.predict(X_norm)
plt.scatter(X,Y)
plt.xlabel("Mes")
plt.ylabel("Volume")
plt.scatter(Xfuturo,
           y_futuro)
plt.scatter(X,y_dataset)
