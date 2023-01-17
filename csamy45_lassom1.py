import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn import linear_model
diabetes = datasets.load_diabetes()
print("Le dataset contient les informations médicales de " + str(diabetes.data.shape[0]) + " personnes suivant " + str(diabetes.data.shape[1]) + " caractéristiques.")
diabetes_X_train = diabetes.data[:-20]

diabetes_y_train = diabetes.target[:-20]

diabetes_X_test = diabetes.data[-20:]

diabetes_y_test = diabetes.target[-20:]
df = pd.DataFrame(diabetes.data)
print(diabetes.DESCR)
df.head()
regr = linear_model.LinearRegression()

regr.fit(diabetes_X_train, diabetes_y_train)
list_alpha = np.linspace(0,0.05,10**3)[1:]
scoreslasso = np.zeros(list_alpha.size)

predictlasso = np.zeros(list_alpha.size)

for i in range(list_alpha.size):

    lasso = linear_model.Lasso(alpha = list_alpha[i])

    lasso.fit(diabetes_X_train, diabetes_y_train)

    scoreslasso[i] = lasso.score(diabetes_X_test, diabetes_y_test)

    predictlasso[i] = np.mean((lasso.predict(diabetes_X_test) - diabetes_y_test)**2)
scoresridge = np.zeros(list_alpha.size)

predictridge = np.zeros(list_alpha.size)

for i in range(list_alpha.size):

    ridge = linear_model.Ridge(alpha = list_alpha[i])

    ridge.fit(diabetes_X_train, diabetes_y_train)

    scoresridge[i] = ridge.score(diabetes_X_test, diabetes_y_test)

    predictridge[i] = np.mean((ridge.predict(diabetes_X_test) - diabetes_y_test)**2)
plt.subplot(2,1,1)

plt.title("Score R^2")

plt.plot(list_alpha, scoresridge, label = "Ridge")

plt.plot(list_alpha, scoreslasso, label = "Lasso")

plt.xlabel("Alpha")

plt.ylabel("Score R^2")

plt.hlines(regr.score(diabetes_X_test,diabetes_y_test),0,0.05, label="Régression linéaire")

plt.legend()



lasso = linear_model.Lasso(alpha = 0.05) # list_alpha[np.argmax(scoreslasso)

lasso.fit(diabetes_X_train, diabetes_y_train)

ridge = linear_model.Ridge(alpha = list_alpha[np.argmax(scoresridge)])

ridge.fit(diabetes_X_train, diabetes_y_train)
plt.figure(figsize=(10,5))

plt.stem(np.arange(lasso.coef_.size), lasso.coef_, label="Lasso")



plt.stem(np.arange(regr.coef_.size) + 0.2, regr.coef_, markerfmt='ro', label = "Regression lineaire")



plt.xlabel("Variable explicative")

plt.ylabel("Valeur du coefficient")

plt.xticks(range(10))

plt.legend()
lasso.coef_
regr.coef_
x,y = np.linspace(-1,3,1000), np.linspace(-1.2,2,1000)

X, Y = np.meshgrid(x,y)
Z = np.power(X-2,2) + np.power(Y-1,2) - (X-2)*(Y-1)
plt.contour(X,Y,Z, levels=np.logspace(-1.5,0.5,num=10), linewidths=0.8, linestyles='dashdot')

plt.contour(X,Y,np.abs(X)+np.abs(Y), levels=[1], colors = 'blue', linewidths=2)

plt.axhline(y=0, linestyle='--', color = 'black', alpha = 0.5)

plt.axvline(x=0, linestyle='--', color = 'black', alpha = 0.5)

plt.scatter([1],[0],marker='+', s=200, color = 'red')

plt.axis("equal")

plt.figure(figsize=(40,40))

plt.savefig("lol.png")
X1 = np.linspace(-1/2,1/2)

X2 = np.linspace(-1,-1/2)

X3 = np.linspace(1/2,1)

plt.plot(X1,np.zeros(X1.shape), color = 'b', label="Seuillage doux (Lasso)")

plt.plot(X2,X2 + 0.5, color = 'b')

plt.plot(X3,X3 - 0.5, color = 'b')

plt.plot(X1,X1, color = 'r', label="Aucun seuillage (régression linéaire)", linestyle="--")

plt.plot(X2,X2, color = 'r', linestyle="--")

plt.plot(X3,X3, color = 'r', linestyle = "--")

plt.xlabel(r'$<X_i,Y>$',fontsize=15)

plt.ylabel(r'$(\hat{\beta}_\lambda)_i$',fontsize=15)

plt.legend()