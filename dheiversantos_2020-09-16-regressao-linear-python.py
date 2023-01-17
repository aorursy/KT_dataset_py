%matplotlib inline

import scipy as sp

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.formula.api as sm



x1 = sp.random.normal(size=100)

x2 = sp.random.normal(size=100)

x3 = sp.random.normal(size=100)

x4 = sp.random.normal(size=100)

categoria = sp.random.binomial(n=1, p=.5, size=100)

erro = sp.random.normal(size=100)
y = 3 + 2*x1 + 0.65*x2 + 3.14*x3 - 1.75*x4 + 1.5*categoria + erro

dic = {"y":y, "x1":x1, "x2":x2, "x3":x3, "x4":x4, "categoria":categoria}

dados = pd.DataFrame(data=dic)

dados   #Visualiza o DataFrame
# Elabora um histograma

plt.hist(y, color='red', bins=15)

plt.title('Histograma da variável resposta')

plt.show()
reg = sm.ols(formula='y~x1+x2+x3+x4+categoria', data=dados).fit()

print(reg.summary())
y_hat = reg.predict()

res = y - y_hat



plt.hist(res, color='orange', bins=15)

plt.title('Histograma dos resíduos da regressão')

plt.show()
plt.scatter(y=res, x=y_hat, color='green', s=50, alpha=.6)

plt.hlines(y=0, xmin=-10, xmax=15, color='orange')

plt.ylabel('$\epsilon = y - \hat{y}$ - Resíduos')

plt.xlabel('$\hat{y}$ ou $E(y)$ - Predito')

plt.show()
coefs = pd.DataFrame(reg.params)

coefs.columns = ['Coeficientes']

print(coefs)
plt.scatter(y=dados['y'], x=dados['x1'], color='blue', s=50, alpha=.5)

X_plot = sp.linspace(min(dados['x1']), max(dados['x1']), len(dados['x1']))

plt.plot(X_plot, X_plot*reg.params[1] + reg.params[0], color='r')

plt.ylim(-11,16)

plt.xlim(-2.5,3)

plt.title('Reta de regressão')

plt.ylabel('$y$ - Variável Dependente')

plt.xlabel('$x1$ - Preditor')

plt.show()