%matplotlib inline

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

X=np.linspace(0, 6, 7).reshape(-1,1)
Y=np.linspace(0, 12, 7).reshape(-1,1)

lr=LinearRegression().fit(X,Y)

print("Coeficiente: {}".format(lr.coef_))
print("Intercepto: {}".format(lr.intercept_))

plt.plot(X, Y)
plt.grid(True)
plt.show()


#criando dados de regressão
from sklearn.datasets import make_regression
X, Y = make_regression(n_samples = 200, n_features = 1, noise = 10, random_state = 5)

plt.scatter(X, Y)
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 5)



#Plotando o gráfico

linha = np.linspace(-3, 3, 500).reshape(-1, 1) #Para gerar a linha da predição
plt.plot(X_train, y_train, "^", markersize = 5)
plt.plot(X_test, y_test, "v", markersize = 7)
plt.plot(linha, lr.predict(linha))
plt.grid(True)


#Plotando o gráfico

linha = np.linspace(-3, 3, 500).reshape(-1, 1) #Para gerar a linha da predição
plt.plot(X_train, y_train, "^", markersize = 5)
plt.plot(X_test, y_test, "v", markersize = 7)
plt.plot(linha, lr.predict(linha))
plt.grid(True)
#plt.title("Modelo Linear\n Treino: {:.2df} - Teste: {:.2df}".format(lr.score(X_train, y_train), lr.score(X_test, y_test)))
plt.show()