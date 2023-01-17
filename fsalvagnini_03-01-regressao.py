import pandas as pd

df = pd.read_csv('../input/appliances-energy-prediction/KAG_energydata_complete.csv')
df.describe()
X = df.drop(['Appliances','date'],axis=1)
y = df.Appliances
print(X.shape)
print(y.shape)
from sklearn import model_selection
from sklearn import linear_model

X_train, X_test, y_train, y_test = model_selection.train_test_split(X[:20], y[:20], test_size=0.5, random_state=42)
model = linear_model.LinearRegression()

model.fit(X_train, y_train)
from sklearn import metrics
import numpy as np
erro_treino = metrics.mean_squared_error(y_train,model.predict(X_train))
print('RMSE no treino:', np.sqrt(erro_treino))

erro_teste = metrics.mean_squared_error(y_test,model.predict(X_test))
print('RMSE no teste:', np.sqrt(erro_teste))
r2 = model.score(X_train, y_train)
print('r² no treino:', r2)

r2 = model.score(X_test, y_test)
print('r² no teste:', r2)
%matplotlib inline
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
def plot_residuals_and_coeff(resid_train, resid_test, coeff):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].bar(np.arange(len(resid_train)), resid_train)
    axes[0].set_xlabel("núm. amostras")
    axes[0].set_ylabel("resíduo")
    axes[0].set_title("treino")
    axes[1].bar(np.arange(len(resid_test)), resid_test)
    axes[1].set_xlabel("núm. amostras")
    axes[1].set_ylabel("resíduo")
    axes[1].set_title("teste")
    axes[2].bar(np.arange(len(coeff)), coeff)
    axes[2].set_xlabel("núm. coeficientes")
    axes[2].set_ylabel("coeficiente")
    fig.tight_layout()
    return fig, axes

residuo_treino = y_train - model.predict(X_train)
residuo_teste  = y_test - model.predict(X_test)

fig, ax = plot_residuals_and_coeff(residuo_treino, residuo_teste, model.coef_)
model = linear_model.Ridge()
model.fit(X_train, y_train)

erro_treino = metrics.mean_squared_error(y_train,model.predict(X_train))
print('RMSE no treino:', np.sqrt(erro_treino))

erro_teste = metrics.mean_squared_error(y_test,model.predict(X_test))
print('RMSE no teste:', np.sqrt(erro_teste))
residuo_treino = y_train - model.predict(X_train)
residuo_teste  = y_test - model.predict(X_test)

fig, ax = plot_residuals_and_coeff(residuo_treino, residuo_teste, model.coef_)
model = linear_model.Lasso(alpha=2.5)
model.fit(X_train, y_train)

erro_treino = metrics.mean_squared_error(y_train,model.predict(X_train))
print('RMSE no treino:', np.sqrt(erro_treino))

erro_teste = metrics.mean_squared_error(y_test,model.predict(X_test))
print('RMSE no teste:', np.sqrt(erro_teste))
residuo_treino = y_train - model.predict(X_train)
residuo_teste = y_test - model.predict(X_test)

fig, ax = plot_residuals_and_coeff(residuo_treino, residuo_teste, model.coef_)
df = pd.read_csv('../input/appliances-energy-prediction/KAG_energydata_complete.csv')
X_train, X_test, y_train, y_test = model_selection.train_test_split(X[:20], y[:20], test_size=0.5, random_state=42)

rmse_treino = []
rmse_teste  = []
alpha = []

for a in range(-4,8):
    model = linear_model.Lasso(alpha=10**a)
    model.fit(X_train, y_train)
    
    print("################################################")
    print('Alpha:', 10**a)
    
    rmse_treino.append(np.sqrt(metrics.mean_squared_error(y_train, model.predict(X_train))))
    rmse_teste.append(np.sqrt(metrics.mean_squared_error(y_test, model.predict(X_test))))
    alpha.append(a)
    
    print('MSE no treino:', rmse_treino[-1])
    print('MSE no teste:', rmse_teste[-1])
    
    print("################################################")

plt.plot(alpha, rmse_treino, alpha, rmse_teste)
rmse_treino = []
rmse_teste  = []
alpha = []

for a in range(-4,8):
    model = linear_model.Ridge(alpha=10**a)
    model.fit(X_train, y_train)
    
    print("################################################")
    print('Alpha:', 10**a)
    
    rmse_treino.append(np.sqrt(metrics.mean_squared_error(y_train, model.predict(X_train))))
    rmse_teste.append(np.sqrt(metrics.mean_squared_error(y_test, model.predict(X_test))))
    alpha.append(a)
    
    print('MSE no treino:', rmse_treino[-1])
    print('MSE no teste:', rmse_teste[-1])
    
    print("################################################")

plt.plot(alpha, rmse_treino, alpha, rmse_teste)
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
# Utilizando o Lasso CV
lasso_reg = LassoCV(cv=5, random_state=0)
lasso_reg.fit(X_train, y_train)
best_alpha_lasso = lasso_reg.alpha_
print("Best alpha => ", best_alpha_lasso)
# Com este valor fazemos o cross validation para outros parâmetros
# Utilizando o Ridge CV
ridge_reg = RidgeCV(cv=5)
ridge_reg.fit(X_train, y_train)
best_alpha_ridge = ridge_reg.alpha_
print("Best alpha => ", best_alpha_ridge)
# Com este valor fazemos o cross validation para outros parâmetros
X = df.drop(['Appliances','date'],axis=1)
y = df.Appliances

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=42)

model = linear_model.Ridge(alpha = best_alpha_ridge)
model.fit(X_train, y_train)

erro_treino = np.sqrt(metrics.mean_squared_error(y_train,model.predict(X_train)))
print('RMSE no treino:', erro_treino)

erro_teste = np.sqrt(metrics.mean_squared_error(y_test,model.predict(X_test)))
print('RMSE no teste:', erro_teste)

residuo_treino = y_train - model.predict(X_train)
residuo_teste  = y_test - model.predict(X_test)

fig, ax = plot_residuals_and_coeff(residuo_treino, residuo_teste, model.coef_)
X = df.drop(['Appliances','date'],axis=1)
y = df.Appliances

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=42)

model = linear_model.Lasso(alpha = best_alpha_lasso)
model.fit(X_train, y_train)

erro_treino = np.sqrt(metrics.mean_squared_error(y_train,model.predict(X_train)))
print('RMSE no treino:', erro_treino)

erro_teste = np.sqrt(metrics.mean_squared_error(y_test,model.predict(X_test)))
print('RMSE no teste:', erro_teste)

residuo_treino = y_train - model.predict(X_train)
residuo_teste  = y_test - model.predict(X_test)

fig, ax = plot_residuals_and_coeff(residuo_treino, residuo_teste, model.coef_)
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

data_gs, data_cv, target_gs, target_cv = train_test_split(X, y, test_size=0.95, random_state=42)

pipeline = Pipeline([('scaler', StandardScaler()), ('clf', linear_model.Ridge())])

# utiliza-se GridSearchCV para achar os melhores parâmetros
from sklearn.model_selection import GridSearchCV
parameters = {'clf__alpha': [0.1,1, 1.5, 3, 5, 7,10,100,1000]} # quais parâmetros e quais valores serão testados
clf = GridSearchCV(pipeline, parameters, cv=3, iid=False) # clf vai armazenar qual foi a melhor configuração
clf.fit(data_gs, target_gs)

print(clf.best_params_ )

# utilizando validação cruzada para avaliar o modelo
scores = cross_val_score(clf, data_cv, target_cv, cv=5, scoring='neg_mean_squared_error')

scores = -scores
scores = np.sqrt(scores)

print('RMSE - %.2f +- %.2f' % (scores.mean(), scores.std()))
print('R²:', clf.score(X_test, y_test) )
erro_teste = np.sqrt(metrics.mean_squared_error(y_test, clf.predict(X_test)))
print('RMSE no treino:', erro_teste)
data_gs, data_cv, target_gs, target_cv = train_test_split(X, y, test_size=0.95, random_state=42)

pipeline = Pipeline([('scaler', StandardScaler()), ('clf', linear_model.Lasso())])

# utiliza-se GridSearchCV para achar os melhores parâmetros
from sklearn.model_selection import GridSearchCV
parameters = {'clf__alpha': [0.1,1, 1.5, 3, 5, 7,10,100,1000]} # quais parâmetros e quais valores serão testados
clf = GridSearchCV(pipeline, parameters, cv=3, iid=False) # clf vai armazenar qual foi a melhor configuração
clf.fit(data_gs, target_gs)

print(clf.best_params_ )

# utilizando validação cruzada para avaliar o modelo
scores = cross_val_score(clf, data_cv, target_cv, cv=5, scoring='neg_root_mean_squared_error')

scores = -scores
# scores = np.sqrt(scores)

print('RMSE - %.2f +- %.2f' % (scores.mean(), scores.std()))
print('R²:', clf.score(X_test, y_test) )
erro_teste = np.sqrt(metrics.mean_squared_error(y_test, clf.predict(X_test)))
print('RMSE no treino:', erro_teste)
data_gs, data_cv, target_gs, target_cv = train_test_split(X, y, test_size=0.95, random_state=42)

pipeline = Pipeline([('scaler', StandardScaler()), ('clf', linear_model.ElasticNet())])

# utiliza-se GridSearchCV para achar os melhores parâmetros
from sklearn.model_selection import GridSearchCV
parameters = {'clf__alpha': [0.1,1, 1.5, 3, 5, 7,10,100,1000]} # quais parâmetros e quais valores serão testados
clf = GridSearchCV(pipeline, parameters, cv=3, iid=False) # clf vai armazenar qual foi a melhor configuração
clf.fit(data_gs, target_gs)

print(clf.best_params_ )

# utilizando validação cruzada para avaliar o modelo
scores = cross_val_score(clf, data_cv, target_cv, cv=5, scoring='neg_mean_squared_error')

scores = -scores
scores = np.sqrt(scores)

print('RMSE - %.2f +- %.2f' % (scores.mean(), scores.std()))
print('R²:', clf.score(X_test, y_test) )
erro_teste = np.sqrt(metrics.mean_squared_error(y_test, clf.predict(X_test)))
print('RMSE no treino:', erro_teste)
residuo_treino = y_train - clf.predict(X_train)
residuo_teste  = y_test - clf.predict(X_test)

fig, ax = plot_residuals_and_coeff(residuo_treino, residuo_teste, clf.best_estimator_['clf'].coef_)