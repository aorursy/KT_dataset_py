# Bibliotecas básicas
import numpy as np
import pandas as pd
# Biblioteca para gerar gráficos
import matplotlib.pyplot as plt
# Vamos carregar as biblioteca para auxiliar Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor 
dataset = pd.read_csv("../input/autompg-dataset/auto-mpg.csv")
dataset.head()
plt.scatter(dataset[["weight"]], dataset[["mpg"]])
plt.xlabel("Peso (libras)")
plt.ylabel("Autonomia (mpg)")
plt.title("Relação entre Peso e Autonomia dos Veículos")

plt.savefig("/kaggle/working/f1.png", dpi=300)
X = dataset[ ["weight"] ]
Y = dataset[ ["mpg"] ]
X = X * 0.453592
Y = Y * 0.425144
escala = StandardScaler()
escala.fit(X)
X_norm = escala.transform(X)
X_norm_train, X_norm_test, Y_train, Y_test = train_test_split(X_norm, Y, test_size=0.3)
reglinear = SGDRegressor(max_iter=500,
                        eta0=0.01,
                        tol=0.0001,
                        verbose=1)
reglinear.fit(X_norm_train, Y_train)
Y_prev = reglinear.predict(X_norm_test)
plt.scatter(X_norm_test, Y_test, label="Real")
plt.scatter(X_norm_test, Y_prev, label="Previsto")
X_test = escala.inverse_transform(X_norm_test)
plt.scatter(X_test, Y_test, label="Real")
plt.scatter(X_test, Y_prev, label="Previsto")
plt.xlabel("Peso (kg)")
plt.ylabel("Autonomia (km/l)")
plt.title("Reg Linear Prevista")
plt.legend(loc=1)
plt.savefig("/kaggle/working/f2.png", dpi=300)
