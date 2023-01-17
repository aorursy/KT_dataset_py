import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
arquivo = r'esteira.xls'
pd = pd.read_excel(arquivo, head='None', sep=';', decimal = ',')
pd.head()
round(pd['VO2'].corr(pd['IMC']),3)
round(pd['VO2'].corr(pd['carga']),3)
import seaborn as sns
ax = sns.distplot(pd['VO2'])
ax = sns.distplot(pd['IMC'])
ax = sns.distplot(pd['carga'])
x_train, x_test, y_train, y_test = train_test_split( 
    pd.iloc[:, :-1], pd.iloc[:, -1],  
    test_size = 0.30) 
print(x_train)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
lr = LinearRegression()
lr.fit(x_train, y_train)
pred_train_lr= lr.predict(x_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_lr)))
print(r2_score(y_train, pred_train_lr))

pred_test_lr= lr.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_lr))) 
print(r2_score(y_test, pred_test_lr))
rr = Ridge(alpha=0.01)
rr.fit(x_train, y_train) 
pred_train_rr= rr.predict(x_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_rr)))
print(r2_score(y_train, pred_train_rr))

pred_test_rr= rr.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_rr))) 
print(r2_score(y_test, pred_test_rr))
rr = Ridge(alpha=10)
rr.fit(x_train, y_train) 
pred_train_rr= rr.predict(x_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_rr)))
print(r2_score(y_train, pred_train_rr))

pred_test_rr= rr.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_rr))) 
print(r2_score(y_test, pred_test_rr))
model_lasso = Lasso(alpha=0.01)
model_lasso.fit(x_train, y_train) 
pred_train_lasso= model_lasso.predict(x_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_lasso)))
print(r2_score(y_train, pred_train_lasso))

pred_test_lasso= model_lasso.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_lasso))) 
print(r2_score(y_test, pred_test_lasso))

model_lasso = Lasso(alpha=0.10)
model_lasso.fit(x_train, y_train) 
pred_train_lasso= model_lasso.predict(x_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_lasso)))
print(r2_score(y_train, pred_train_lasso))

pred_test_lasso= model_lasso.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_lasso))) 
print(r2_score(y_test, pred_test_lasso))
#Elastic Net
model_enet = ElasticNet(alpha = 0.01)
model_enet.fit(x_train, y_train) 
pred_train_enet= model_enet.predict(x_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_enet)))
print(r2_score(y_train, pred_train_enet))

pred_test_enet= model_enet.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_enet)))
print(r2_score(y_test, pred_test_enet))

#Elastic Net
model_enet = ElasticNet(alpha = 1)
model_enet.fit(x_train, y_train) 
pred_train_enet= model_enet.predict(x_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_enet)))
print(r2_score(y_train, pred_train_enet))

pred_test_enet= model_enet.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_enet)))
print(r2_score(y_test, pred_test_enet))
import math
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
regressao = smf.ols ('VO2~IMC+carga', data = pd).fit()
print(regressao.summary())
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
stats.shapiro(pd['VO2'])
# Teste de Shapiro - teste da normalidade que devem ter os dados. pvalue > 0,05, então dados vieram de uma normal.
stats.kstest(pd['VO2'],cdf='norm')
# Teste de linearidade - Kolgomorov (verificar se as coisas vieram de uma normal)
from scipy.stats import normaltest
stat, p = normaltest(pd)
print(stat, p)
# valores p maiores que 0,05, distribuição assumida como normal
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
# definição dos valores de alpha que serão testados
alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
# instanciação do modelo
ridge = Ridge()
# valores de alpha que serão testados
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}
# execução do modelo com cv = 5 - utilizando o 'GridSearch'
ridge_regressor = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(x_train, y_train)
# melhores parâmetros encontrados
ridge_regressor.best_params_
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
# definição dos valores de alpha que serão testados
alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
# instanciação do modelo
ridge = Ridge()
# valores de alpha que serão testados
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}
# execução do modelo com cv = 5 - utilizando o 'GridSearch'
ridge_regressor = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(x_test, y_test)
# melhores parâmetros encontrados
ridge_regressor.best_params_
# melhor score encontrado
ridge_regressor.best_score_
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import cross_val_score
from math import sqrt
import warnings
warnings.filterwarnings("ignore")
regr_ridgeCV = RidgeCV(cv=10)
score_ridge = cross_val_score(regr_ridgeCV, x_train, y_train, cv=10, scoring="neg_mean_squared_error")
print(score_ridge.mean())
# Valor encontrado por validação cruzada
regr_ridgeCV.fit(x_train, y_train)
regr_ridgeCV.alpha_
from sklearn.linear_model import RidgeCV
regr_ridgeCV = RidgeCV(alphas=[0.01,0.1,1], store_cv_values=True)
regr_ridgeCV.fit(x_train, y_train) 
pred_train_ridge= regr_ridgeCV.predict(x_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_ridge)))
print(r2_score(y_train, pred_train_ridge))

pred_test_ridge= regr_ridgeCV.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_ridge))) 
print(r2_score(y_test, pred_test_ridge))

y_predicted = regr_ridgeCV.predict(x_test)
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_predicted, s=20)
plt.xlabel('Valor Y')
plt.ylabel('Valor previsto de Y')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])
plt.tight_layout() 
regr_lassoCV = LassoCV(alphas=[0.01,0.1,1], max_iter=1000, cv=10)
regr_lassoCV.fit(x_train, y_train) 
pred_train_lasso= regr_lassoCV.predict(x_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_lasso)))
print(r2_score(y_train, pred_train_lasso))

pred_test_lasso= regr_lassoCV.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_lasso))) 
print(r2_score(y_test, pred_test_lasso))

y_predicted = regr_lassoCV.predict(x_test)
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_predicted, s=20)
plt.xlabel('Valor Y')
plt.ylabel('Valor previsto de Y')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])
plt.tight_layout()
regr_enCV = ElasticNet(alpha = 1)
regr_enCV.fit(x_train, y_train) 
pred_train_regr_enCV= regr_enCV.predict(x_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_regr_enCV)))
print(r2_score(y_train, pred_train_en_regr))

pred_test_regr_enCV= en_regr.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_regr_enCV)))
print(r2_score(y_test, pred_test_regr_enCV))

y_predicted = regr_enCV.predict(X_test)
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_predicted, s=20)
plt.xlabel('Valor Y')
plt.ylabel('Valor previsto de Y')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])
plt.tight_layout() 
