# Vamos carregar as bibliotecas básicas
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
## Vamos carregar as bibliotecas gráficas
import matplotlib.pyplot as plt # Para plotar os gráficos
# %matplotlib inline
## Vamos carregar as bibliotecas de Machine Learning
from sklearn.preprocessing import StandardScaler     # Para normalizar os dados
from sklearn.model_selection import train_test_split # Para criar o conjunto de dados de treinamento e teste
from sklearn.metrics import mean_squared_error       # Para calcular o erro médio quadrado
from sklearn.metrics import r2_score                 # Para calcular o R^2
from sklearn.linear_model import SGDRegressor        # Modelo de Regressão Linear (utilizando Grad Descendente)
## Carrega o dataset
## O parâmetro usecols especifica quais colunas iremos utilizar
dataset = pd.read_csv("../input/autompg-dataset/auto-mpg.csv", usecols=['weight', 'mpg'])
## Vamos ver uma prévia do conjunto de dados
dataset.head()
## Vamos plotar o dataset para ver sua estrutura
# plt.figure(figsize=(20,10))
plt.scatter(dataset["weight"], dataset["mpg"])
plt.xlabel("Peso (kg)")
plt.ylabel("Autonomia (mpg)")

# O comando abaixo salva a figura no arquivo /kaggle/working/grafico.png 
# com uma resolução alta (dpi=300) e com margens justas (bbox_incheds='tight')
plt.savefig("/kaggle/working/grafico.png", dpi=300, bbox_inches = 'tight')
X = dataset[["weight"]]
y = dataset[["mpg"]]
##### Transforma de libras (lbs) para quilos (kg)
X = X * 0.4535923

##### Transforma de M/G (Miles per Galon) para KM/L (Quilometros por Litro)
y = y * 0.4251437
##### Número de Exemplos
m = y.size
#### Escala
escala = StandardScaler()
escala.fit(X)

X_norm = escala.transform(X)
#### Dividir em conjunto de treinamento e teste
X_norm_train, X_norm_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2)
reglinear = SGDRegressor(max_iter=500,              # Número Máximo de Épocas
                         tol=1e-5,                  # Convergência quando melhora for menor do que 1^-5
                         eta0=1e-2,                 # Taxa de Aprendizagem 1^-2
                         learning_rate="constant",  # Taxa constante
                         verbose=1)  

reglinear.fit(X_norm_train, y_train) # Realize o Treinamento
# Norma do vetor
print(reglinear.coef_)

# Bias (Theta0)
print(reglinear.intercept_)

# T - Numero de updates de peso
print(reglinear.t_)

# Shape
print(X_norm_train.shape[0] * 29)
### Prever
y_pred = reglinear.predict(X_norm_test)
### Calcular o R^2
r2 = r2_score(y_test, y_pred)
r2
### Escalando os dados de volta
X_test = escala.inverse_transform(X_norm_test)
plt.scatter(X_test, y_test, label="Real")
plt.scatter(X_test, y_pred, label="Previsto")
plt.legend(loc="upper right")
plt.xlabel("Peso (kg)")
plt.ylabel("Autonomia (mpg)")
plt.scatter(X_test, y_test, label="Real")

X_intervalo = np.linspace(np.min(X_test), np.max(X_test), 500).reshape((500, 1))
y_intervalo = reglinear.predict(escala.transform(X_intervalo))

plt.plot(X_intervalo, y_intervalo, linewidth=2, label="Previsto", color="orange")
plt.text(1600, 9, '$R^2$ = %0.2f' % r2, fontsize=12)

plt.legend(loc=1)
plt.xlabel("Peso (kg)")
plt.ylabel("Autonomia (mpg)")
