from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
import math
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
arquivo = r'Auto_e.xlsx'
auto = pd.DataFrame(pd.read_excel(arquivo, sep=',', quotechar='"',decimal = '.',encoding="utf8",low_memory=True))
auto.head()
auto.describe
print(auto.isnull().sum())
import seaborn as sns
plt.figure(figsize=(15,6))
c = auto.corr()
sns.heatmap(c,cmap='BrBG',annot=True)
c
mh= auto.loc[:,('horsepower','mpg')]
mh.head()
round(auto['mpg'].corr(auto['horsepower']),3)
import plotly_express as px
aa = px.histogram(mh, x="horsepower", color="mpg")
aa.show()
aa = px.scatter(mh, x="horsepower", color="mpg")
aa.show()
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split( 
    mh.iloc[:, :-1], mh.iloc[:, -1],  
    test_size = 0.30) 
x_test.head()
y_test.head()
# Regressão Linear
lr = LinearRegression()
lr.fit(x_train, y_train)
regressao = smf.ols ('mpg~horsepower', data = mh).fit()
print(regressao.summary())
round(regressao.rsquared,3)
# Rsquare individual extraído da tabela acima, depois do cálculo
regressao.tvalues
# extração dos valores independentemente
regressao.fvalue
# idem acima, depois do cálculo
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
res = regressao.resid
fig = qqplot(res, line='s')
plt.show()
pred_train_lr= lr.predict(x_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_lr)))
print(r2_score(y_train, pred_train_lr))

pred_test_lr= lr.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_lr))) 
print(r2_score(y_test, pred_test_lr))
a=(y_test - lr.predict(x_test))
ax = sns.distplot(a)
mh.head()
mh["hp2"]=mh['horsepower']*mh['horsepower']
mh2= mh.loc[:,('hp2','mpg')]
mh2.head()
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split( 
    mh2.iloc[:, :-1], mh2.iloc[:, -1],  
    test_size = 0.30) 
from sklearn.linear_model import LinearRegression
# Regressão Linear
lr = LinearRegression()
lr.fit(x_train, y_train)
regressao = smf.ols ('mpg~hp2', data = mh2).fit()
print(regressao.summary())
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
res = regressao.resid
fig = qqplot(res, line='s')
plt.show()
pred_train_lr= lr.predict(x_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_lr)))
print(r2_score(y_train, pred_train_lr))

pred_test_lr= lr.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_lr))) 
print(r2_score(y_test, pred_test_lr))
a=(y_test - lr.predict(x_test))
ax = sns.distplot(a)
# Novo dataframe, chamado agora de "S"
S= auto.loc[:,('displacement','horsepower','weight','acceleration','mpg')]
S.head()
# Separação entre teste e treino
x_train, x_test, y_train, y_test = train_test_split( 
    S.iloc[:, :-1], S.iloc[:, -1],  
    test_size = 0.30) 
from sklearn.linear_model import RidgeCV
modelo_ridge = RidgeCV(alphas=[0.01,0.1,1,5,10,20], store_cv_values=True, scoring = 'neg_mean_squared_error',cv=None)
modelo_ridge.fit(x_train, y_train) 
pred_train_ridge= modelo_ridge.predict(x_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_ridge)))
print( r2_score(y_train, pred_train_ridge))
modelo_ridge.alpha_
pred_test_ridge= modelo_ridge.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_ridge))) 
print(r2_score(y_test, pred_test_ridge))

y_predicted = modelo_ridge.predict(x_test)
plt.figure(figsize=(15, 10))
plt.scatter(y_test, y_predicted, color = "green", s=20)
plt.xlabel('Valor Y - teste')
plt.ylabel('Valor previsto de Y - teste')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])
plt.tight_layout() 

pd.Series(modelo_ridge.coef_, index = x_test.columns)
mean_squared_error(y_test, y_predicted)
x_test.head()
x_test.describe
y_test.head()
y_predicted
w=y_predicted-y_test
w.describe
plt.figure(figsize=(15, 10))
plt.scatter(y_test,w, color="blue")
plt.xlabel('Valor Y - teste')
plt.ylabel('Valor Previsto-Real')
# https://www.analyticsvidhya.com/blog/2016/01/ridge-lasso-regression-python-complete-tutorial/
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
# definição dos valores de alpha que serão testados. Alpha é o coeficiente que multiplica o quadrado dos pesos.
alpha = [1e-4, 1e-3,1e-2, 1, 5, 10,20,50]
# instanciação do modelo
ridge = Ridge()
# valores de alpha que serão testados
parameters = {'alpha': [1e-4, 1e-3,1e-2, 1, 5, 10,20,50]}
# execução do modelo com cv = 3 - utilizando o 'GridSearch'
ridge_regressor = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=3)
ridge_regressor.fit(x_train, y_train)
# melhores parâmetros encontrados. Esse é o melhor coeficiente que deve multiplicar os quadrados dos pesos.
ridge_regressor.best_params_
# melhor score encontrado, o menor valor do "neg_mean_squared_error", para o caso com cv=3. 
# Partimos o conjunto em 3 pedaços.
# Um será para treino e os demais para teste.
ridge_regressor.best_score_
from sklearn.linear_model import LassoCV
modelo_lasso = LassoCV(alphas=[0.01,0.1,1,5,10,20,50])
modelo_lasso.fit(x_train, y_train) 
pred_train_lasso= modelo_lasso.predict(x_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_lasso)))
print( r2_score(y_train, pred_train_ridge))
modelo_lasso.alpha_
pred_test_lasso= modelo_lasso.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_lasso))) 
print(r2_score(y_test, pred_test_lasso))
y_predicted = modelo_lasso.predict(x_test)
plt.figure(figsize=(15, 8))
plt.scatter(y_test, y_predicted, color='red',s=20)
plt.xlabel('Valor Y')
plt.ylabel('Valor previsto de Y')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])
plt.tight_layout() 
pd.Series(modelo_lasso.coef_, index = x_test.columns)
w1=y_predicted-y_test
plt.figure(figsize=(15, 10))
plt.scatter(y_test,w1, color="blue")
plt.xlabel('Valor Y - teste')
plt.ylabel('Valor Previsto-Real')
R=w
L=w1
w2=R-L
plt.figure(figsize=(15, 10))
plt.scatter(y_test,w2, color="orange")
plt.xlabel('Valor Y - teste')
plt.ylabel('Ridge - Lasso')
