import pandas as pd

data = pd.read_csv('../input/ad.data', index_col=0)
display(data.head())
data.shape
import seaborn as sns
%matplotlib inline

sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=7, aspect=0.7, kind='reg')
atributos = ['TV', 'Radio', 'Newspaper']

X = data[atributos]
X.head()
y = data['Sales']
y.head()
from sklearn.model_selection import train_test_split

# omitindo o parametro test_size, considera-se 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=1) 

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
from sklearn.linear_model import LinearRegression

reglin = LinearRegression()

reglin.fit(X_train, y_train)
print(reglin.intercept_)
print(reglin.coef_)
list(zip(atributos, reglin.coef_)) # o comando zip junta dois vetores e forma tuplas
reglin.predict([[100,20,15]])
# CÓDIGO DO EXERCÍCIO 3

from sklearn.linear_model import LinearRegression

reglin = LinearRegression()

reglin.fit(X_train, y_train)

print(reglin.intercept_)
print(reglin.coef_)

list(zip(atributos, reglin.coef_)) # o comando zip junta dois vetores e forma tuplas

reglin.predict([[20,150,20]])
# CÓDIGO DO EXERCÍCIO 4

reglin.predict([[200,100,20],[520,20,15]])
# CÓDIGO DO EXERCÍCIO 5
reglin.predict([[0,0,0]])

true = [100, 50, 30, 20]
pred = [90, 50, 50, 30]
# calculando MAE na mão
print((10 + 0 + 20 + 10)/4.)

# calculando MAE com scikit
from sklearn import metrics
print(metrics.mean_absolute_error(true, pred))
# calculando MSE na mão
print((10**2 + 0**2 + 20**2 + 10**2)/4.)

# calculando MSE com scikit
print(metrics.mean_squared_error(true, pred))
# calculando RMSE na mão
import numpy as np
print(np.sqrt((10**2 + 0**2 + 20**2 + 10**2)/4.))

# calculando RMSE com scikit
print(np.sqrt(metrics.mean_squared_error(true, pred)))
y_pred = reglin.predict(X_test)

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
atributos = ['TV', 'Radio']

X = data[atributos]
y = data.Sales

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

reglin.fit(X_train, y_train)

y_pred = reglin.predict(X_test)

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# CÓDIGO DO EXERCÍCIO 7

import pandas as pd

data = pd.read_csv('../input/ad.data', index_col=0)

atributos = ['TV', 'Radio']

X = data[atributos]
X.head()

y = data['Sales']
y.head()


from sklearn.model_selection import train_test_split

# omitindo o parametro test_size, considera-se 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.75, random_state=1) # test_size = tamanho do teste

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression

reglin = LinearRegression()

reglin.fit(X_train, y_train)

print(reglin.intercept_)
print(reglin.coef_)

list(zip(atributos, reglin.coef_)) # o comando zip junta dois vetores e forma tuplas

y_pred = reglin.predict(X_test)

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.model_selection import cross_val_score
import pandas as pd

data = pd.read_csv('../input/ad.data', index_col=0)

atributos = ['TV', 'Radio', 'Newspaper']

X = data[atributos]
y = data.Sales

scores = cross_val_score(LinearRegression(), X, y, cv=10, scoring='neg_mean_squared_error')
scores = np.sqrt(-scores)

print(scores)
print("RMSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# CÓDIGO DO EXERCÍCIO 8from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score
import pandas as pd

data = pd.read_csv('../input/ad.data', index_col=0)

atributos = ['TV', 'Radio']

X = data[atributos]
y = data.Sales

scores = cross_val_score(LinearRegression(), X, y, cv=10, scoring='neg_mean_squared_error')
scores = np.sqrt(-scores)

print(scores)
print("RMSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))