# Vamos carregar as bibliotecas básicas
import numpy as np
import pandas as pd
# Vamos importar as bibliotexas de geração de gráficos
import matplotlib.pyplot as plt
# Vamos caregar as bibliotecas de Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
dataset = pd.read_csv("../input/traffic-volume-in-brazils-highway-toll-plazas/all.csv",
                     sep= ";",
                     decimal= ",",
                     encoding= "ISO-8859-1")
dataset.head()

dataset["Concessionaria"].unique()
dataset_dutra= dataset[dataset["Concessionaria"] == "01.Nova Dutra"]
dataset_dutra["Categoria"].unique()
dataset_dutra_passeio = dataset_dutra[dataset_dutra["Categoria"] == "Categoria 1"]
dataset_dutra_passeio["Praca"].unique()
dataset_dutra_passeio["mes_ano"] = pd.to_datetime(dataset_dutra_passeio["mes_ano"],
                                                 format= "%b-%y")
dataset_dutra_passeio["mes_ano"]
dataset_agrupado = dataset_dutra_passeio.groupby(["mes_ano"]).sum()
dataset_agrupado.head()
dataset_agrupado["mes"] = dataset_agrupado.index.month * (dataset_agrupado.index.year - 2009)
dataset_agrupado["mes"]
dataset_agrupado
x =  dataset_agrupado[["mes"]]
y = dataset_agrupado[["Volume_total"]]
plt.scatter(x,y)
plt.xlabel("Meses")
plt.ylabel("Volume")
x =  dataset_agrupado[["mes"]]
y = dataset_agrupado[["Volume_total"]]

intervalo = np.array([np.arange(121, 145)])
x_futuro = intervalo.transpose()
escala = StandardScaler()
escala.fit(x)
x_norm = escala.transform(x)


x_norm_train, x_norm_test, y_train, y_test = train_test_split(x_norm, y, test_size = 0.4)
reglinear = MLPRegressor(hidden_layer_sizes=(700,500,300,100),
                         max_iter=1500, 
                         verbose=True)

reglinear.fit(x_norm_train, y_train) 
y_pred = reglinear.predict(x_norm_test)

plt.scatter(x_norm_test, y_test)
plt.scatter(x_norm_test, y_pred)
np.array(y_test)
y_pred
y_test_ = np.array(list(map(lambda z: z[0],np.array(y_test))))

y_pred_ = np.array(y_pred)
#Somar mais 24 meses

y_pred_2021 = reglinear.predict(x_futuro)
y_pred_2021 = int(round(y_pred_2021[0],0))

#Deixar a leitura mais facil:

n_num = str(y_pred_2021)
y_pred_2021_ = [n_num[0:z]+"."+n_num[z:] for z in list(range(0,len(n_num),3))[::-1]][0:-1][::-1]
y_pred_2021_corrigido = y_pred_2021_[0]
for z in y_pred_2021_[1:]:
    corte_ponto = z.split('.')[-1]
    y_pred_2021_corrigido = y_pred_2021_corrigido[0:-len(corte_ponto)] + "." + corte_ponto

print("Previsão para 2021 teremos um tráfego de {} véiculos de categoria 1".format(y_pred_2021_corrigido))