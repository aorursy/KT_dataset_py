#importando as bibliotecas básicas

import pandas as pd

import numpy as np
#Geração de gráficos

import matplotlib.pyplot as plt
#Bibliotecas especificas para o aprendizado de máquina

#Permite normalizar os dados

from sklearn.preprocessing import StandardScaler



#Permite dividir conjunto de dados

from sklearn.model_selection import train_test_split

#Permite medir eficiencia do modelo

from sklearn.metrics import r2_score

#Permite comparar os dois tipos de modelos

from sklearn.linear_model import SGDRegressor

from sklearn.neural_network import MLPRegressor
dataset = pd.read_csv("../input/disjuntorsf63/SF6.csv")
dataset.head()
plt.scatter(dataset[["Age"]], dataset[["Health Index"]])

plt.xlabel("Idade(Anos)")

plt.ylabel("Índice de Saúde")

plt.title("Relação Idade x Saúde do Disjuntor")
X = dataset[["Age"]]

Y = dataset[["Health Index"]]
X.describe()
#Normalização

escala = StandardScaler()

escala.fit(X)

X_norm = escala.transform(X)

#Dividir em conjunto de treinamento e teste

X_norm_train, X_norm_test, Y_train, Y_test = train_test_split(X_norm, Y, test_size=0.3)
rna = MLPRegressor(hidden_layer_sizes=(10, 5),

                  max_iter=2000,

                  tol=0.0000001,

                  learning_rate_init=0.1,

                  solver="sgd",

                  activation="logistic",

                  learning_rate="constant",

                  verbose=2,

                  )

rna.fit(X_norm_train, Y_train)
reglinear = SGDRegressor(max_iter=2000,

                        tol=0.000001,

                        eta0=0.1,

                        learning_rate="constant",

                        verbose=2)
reglinear.fit(X_norm_train, Y_train)
#Previsão no conjunto de teste

Y_rna_previsao = rna.predict(X_norm_test)

Y_rl_previsao = reglinear.predict(X_norm_test)
#Calcula o R^2

r2_rna = r2_score(Y_test, Y_rna_previsao)

r2_rl = r2_score(Y_test, Y_rl_previsao)



print("R2 RNA:", r2_rna)

print("R2 RL", r2_rl)
X_test = escala.inverse_transform(X_norm_test)



plt.scatter(X_test, Y_test, alpha=0.5, label="Reais")

plt.scatter(X_test, Y_rna_previsao, alpha=0.5, label="MLP")

plt.scatter(X_test, Y_rl_previsao, alpha=0.5, label="Reg Linear")

plt.xlabel("Idade")

plt.ylabel("Índice de Saúde")

plt.title("Relação Idade x Saúde do Disjuntor")

plt.legend(loc=1)
#Prever para um novo dado

X_futuro = np.array([[20]])



X_futuro_norm = escala.transform(X_futuro.T)
y_rna_prev_futuro = rna.predict(X_futuro_norm)

y_reglinear_prev_futuro = reglinear.predict(X_futuro_norm)
print("RNA:", y_rna_prev_futuro)

print("Reg Linear:", y_reglinear_prev_futuro)
plt.scatter(X,Y, label="Real")

plt.scatter(X_futuro, y_rna_prev_futuro, label="MLP")

plt.scatter(X_futuro, y_reglinear_prev_futuro, label="Reg Linear")

plt.xlabel("Idade")

plt.ylabel("Índice de Saúde")

plt.title("Relação Idade x Saúde do Disjuntor")

plt.legend(loc=1)