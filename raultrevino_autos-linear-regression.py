import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
data_auto = pd.read_csv("../input/auto-mpg.csv")

data_auto.head()
data_auto.shape
data_auto.columns.values.tolist()
data_auto.describe()
print("Numero de registros:"+str(data_auto.shape[0]))

for column in data_auto.columns.values:

    print(column + "-NAs:"+ str(pd.isnull(data_auto[column]).values.ravel().sum()))
data_auto = data_auto.dropna()
data_auto.shape
print("Correlaciones en el dataset:")

data_auto.corr()
plt.matshow(data_auto.corr())
%matplotlib inline

plt.plot(data_auto["cylinders"],data_auto["displacement"],"bo")

plt.xlabel("Cilindros")

plt.ylabel("Desplazamiento")

plt.title("Cilindros vs Desplazamiento")
%matplotlib inline

plt.plot(data_auto["horsepower"],data_auto["mpg"],"ro")

plt.xlabel("Caballos de Potencia")

plt.ylabel("Consumo (millas por galeón)")

plt.title("CV vs MPG")
%matplotlib inline

plt.plot(data_auto["weight"],data_auto["mpg"],"go")

plt.xlabel("Peso")

plt.ylabel("Consumo (millas por galeón)")

plt.title("Peso vs MPG")
%matplotlib inline

plt.plot(data_auto["acceleration"],data_auto["horsepower"],"ro")

plt.xlabel("Aceleracion")

plt.ylabel("Caballos de fuerza")

plt.title("Aceleracion vs Caballos de fuerza")
feature_cols = ['mpg',

                 'cylinders',

                 'displacement',

                 'weight',

                 'acceleration',

                 'origin']

X = data_auto[feature_cols]

Y = data_auto["horsepower"]
estimator = SVR(kernel="linear")

selector = RFE(estimator,2,step=1)

selector = selector.fit(X,Y)
selector.support_
z = zip(feature_cols,selector.support_)

list(z)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3, random_state=0)
X_pred = X_train[["cylinders","acceleration"]]
lm = LinearRegression()

lm.fit(X_pred,Y_train)
lm.intercept_
lm.coef_
print(len(X_test))

print(len(Y_test))
X_pred = X_test[["cylinders","acceleration"]]

lm.score(X_pred,Y_test)
%matplotlib inline

plt.plot(data_auto["acceleration"],data_auto["horsepower"],"ro")

plt.xlabel("Aceleracion")

plt.ylabel("Caballos de fuerza")

plt.title("Aceleracion vs Caballos de fuerza")
data_auto[(data_auto["acceleration"]>18) & (data_auto["horsepower"]>175)]
data_auto = data_auto.drop([34,])
data_auto[(data_auto["acceleration"]>13) & (data_auto["horsepower"]>175)]
data_auto = data_auto.drop([31,32,33])
%matplotlib inline

plt.plot(data_auto["acceleration"],data_auto["horsepower"],"ro")

plt.xlabel("Aceleracion")

plt.ylabel("Caballos de fuerza")

plt.title("Aceleracion vs Caballos de fuerza")
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3, random_state=0)

X_pred = X_train[["cylinders","acceleration"]]

lm = LinearRegression()

lm.fit(X_pred,Y_train)

X_pred = X_test[["cylinders","acceleration"]]

lm.score(X_pred,Y_test)
from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model
X_cylinders = X_train[["cylinders"]].values.tolist()

X_accelearation = X_train[["acceleration"]].values.tolist()
X_final_predictors = np.concatenate((X_cylinders,X_accelearation),axis=1)

poly = PolynomialFeatures(2)

X_final_predictors_train = poly.fit_transform(X_final_predictors)
lm = linear_model.LinearRegression()

lm.fit(X_final_predictors_train,Y_train)
X_cylinders = X_test[["cylinders"]].values.tolist()

X_accelearation = X_test[["acceleration"]].values.tolist()

X_final_predictors = np.concatenate((X_cylinders,X_accelearation),axis=1)

poly = PolynomialFeatures(2)

X_final_predictors_test = poly.fit_transform(X_final_predictors)

lm.score(X_final_predictors_test,Y_test)
lm.intercept_
lm.coef_