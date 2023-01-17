# Vamos carregar as principais bibliotecas básicas
import numpy as np
import pandas as pd
# Vamos carregar as bibliotecas gráficas
import matplotlib.pyplot as plt
# Vamos carregar as bibliotecas de Machine Learning
from sklearn.preprocessing import StandardScaler     # Para normalizar os dados
from sklearn.model_selection import train_test_split # Para criar o conjunto de dados de treinamento e teste
from sklearn.linear_model import SGDRegressor        # Modelo de Regressão Linear (utilizando Grad Descendente)
# Carrega o dataset
dataset = pd.read_csv("../input/autompg-dataset/auto-mpg.csv")
# Vamos ver uma prévia do conjunto de dados
dataset.head()
plt.scatter(dataset[["weight"]], dataset[["mpg"]])
plt.xlabel("Peso (kg)")
plt.ylabel("Autonomia (mpg)")
plt.title("Relação entre Peso e Autonomia dos Veículos")

plt.savefig("/kaggle/working/grafico.png", dpi=300, bbox_inches='tight')
X = dataset[["weight"]]
y = dataset[["mpg"]]
##### Transforma de libras (lbs) para quilos (kg)
X = X * 0.4535923

##### Transforma de M/G (Miles per Galon) para KM/L (Quilometros por Litro)
y = y * 0.4251437
#### Escala
escala = StandardScaler()
escala.fit(X)

X_norm = escala.transform(X)
#### Dividir em conjunto de treinamento e teste
X_norm_train, X_norm_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3)
reglinear = SGDRegressor(max_iter=500,              # Número Máximo de Épocas
                         tol=1e-5,                  # Convergência quando melhora for menor do que 1^-5
                         eta0=1e-2,                 # Taxa de Aprendizagem 1^-2
                         learning_rate="constant",  # Taxa constante
                         verbose=1)  
reglinear.fit(X_norm_train, y_train) # Realize o Treinamento
### Prever
y_pred = reglinear.predict(X_norm_test)
### Escalando os dados de volta
X_test = escala.inverse_transform(X_norm_test)
# Relacionar a previsão com o real
plt.scatter(X_norm_test, y_test, label="Real")
plt.scatter(X_norm_test, y_pred, label="Previsto")

plt.scatter(X_test, y_test, label="Real")
plt.scatter(X_test, y_pred, label="Previsto")
plt.legend(loc=1)
plt.xlabel("Peso (kg)")
plt.ylabel("Autonomia (mpg)")
y_pred = reglinear.predict(X_norm)
plt.scatter(X_norm, y, label="Real")
plt.scatter(X_norm, y_pred, label="Previsto")
plt.legend(loc=1)
plt.xlabel("Peso (kg)")
plt.ylabel("Autonomia (mpg)")