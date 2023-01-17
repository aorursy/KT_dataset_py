%matplotlib inline



import matplotlib

import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn import datasets

import matplotlib.pyplot as plt
# Load the diabetes dataset

X, y = datasets.load_diabetes(return_X_y=True)
# Usando apenas uma variável

X = X[:, np.newaxis, 2]
# Dividindo o X em treino e validação

X_train = X[:-20]

X_test = X[-20:]
# Dividindo o X em treino e validação

y_train = y[:-20]

y_test = y[-20:]
#treinamento do modelo

modelo_regressao = LinearRegression()

modelo_regressao.fit(X_train, y_train)
#predizendo a saída de teste

y_pred = modelo_regressao.predict(X_test)
#avaliação do resultado

print('Mean squared error:', mean_squared_error(y_test, y_pred))
#Visualização da saída

# Plot outputs

plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')

plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)



plt.xticks(())

plt.yticks(())

X = pd.DataFrame({"Nivel_escolaridade": np.linspace(start = 0, stop=10*np.pi, num= 300)})

ruido = np.random.rand(len(X)) * 1.1

y = pd.DataFrame({"Salario": (np.sin(X.values.ravel()) + ruido)})
X.head()
y.head()
plt.scatter(X,y)

plt.xlabel("Escolaridade")

plt.ylabel("Salario")
from sklearn.model_selection import train_test_split
test_size = 0.15 

seed = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size = test_size,

                                                    random_state=seed,

                                                   shuffle= False)
modelo_regressao = LinearRegression()

modelo_regressao.fit(X_train, y_train)
#predizendo a saída de teste

y_pred = modelo_regressao.predict(X_test)

y_pred_treino = modelo_regressao.predict(X_train)
#avaliação do resultado

print('Mean squared error:', mean_squared_error(y_test, y_pred))
#Visualização da saída

plt.scatter(X_train, y_train,  color='black')

plt.plot(X_train, y_pred_treino, color='blue', linewidth=3)



plt.xticks(())

plt.yticks(())
from sklearn.tree import DecisionTreeRegressor
decision_tree = DecisionTreeRegressor(min_samples_split=10)

decision_tree.fit(X_train, y_train)
y_pred_treino    = decision_tree.predict(X_train)

y_pred_validacao = decision_tree.predict(X_test)
#Visualização da saída

plt.scatter(X_train, y_train,  color='black')

plt.plot(X_train, y_pred_treino, color='blue', linewidth=3)



plt.xticks(())

plt.yticks(())
mse_treino    = mean_squared_error(y_true=y_train, y_pred=y_pred_treino)

mse_validacao = mean_squared_error(y_true=y_test, y_pred=y_pred_validacao)



print("Erro de treino: ", mse_treino)

print("Erro de validacao: ", mse_validacao)
