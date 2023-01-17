%matplotlib inline
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from IPython.display import Image

#regressão linear
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#validação cruzada
from sklearn.model_selection import cross_val_score
dadosTreino = pd.read_csv("../input/train.csv", na_values="?")
dadosTreino.shape
dadosTeste = pd.read_csv("../input/test.csv", na_values="?")
dadosTeste.shape
dados1=dadosTreino.drop("Id", axis=1)
correla=(dados1.corr()["median_house_value"])
correla
dados_limpos = dadosTreino.drop(["Id","longitude","population","total_bedrooms","median_house_value"], axis=1)
feature = dadosTreino["longitude"]
plt.scatter(feature, dadosTreino["median_house_value"],  color='black')
plt.xticks(())
plt.yticks(())
plt.show()
dados_X_train = dados_limpos[:-48] # deixa os primeiros 14400 dados de treino
dados_X_test = dados_limpos[-48:] # deixa os ultimos 48 dados de teste

dados_y_train = dadosTreino["median_house_value"]
dados_y_train = dados_y_train[:-48]
dados_y_test = dadosTreino["median_house_value"]
dados_y_test = dados_y_test[-48:]
%matplotlib inline
import scipy as sp
import statsmodels.formula.api as sm
reg = sm.ols(formula='median_house_value~latitude+total_rooms+median_income', data=dadosTreino).fit()
print(reg.summary())
y_hat = reg.predict()
res = dadosTreino["median_house_value"] - y_hat
plt.hist(res, color='blue', bins=100)
plt.title('Histograma dos resíduos da regressão')
plt.show()
# Make predictions using the testing set
dados_y_pred_linear = linear_model.LinearRegression()
dados_y_pred_linear.fit(dados_X_train, dados_y_train)
# The coefficients
print('Coefficients: \n', dados_y_pred_linear.coef_)
# The  residual squared mean error
scores = cross_val_score(dados_y_pred_linear, dados_X_train, dados_y_train, cv=6, scoring = "neg_mean_squared_error")
print("Erro RSME médio: ", np.sqrt(-scores.mean())/dados_y_train.mean())
from sklearn.model_selection import cross_val_score as cvs
from sklearn.neighbors import KNeighborsRegressor as knr
knn_scores = []
for i in range(1,100,5):
    knn = knr(n_neighbors = i)
    scores = cvs(knn, dados_X_train, dados_y_train, scoring='neg_mean_squared_error',
                cv = 5)
    knn_scores.append([i, -scores.mean()])
knn_scores = np.array(knn_scores)
knn_scores
knn_scores[np.where(knn_scores[:,1] == np.amin(knn_scores[:,1]))[0]]
from sklearn.linear_model import Lasso
l = Lasso(alpha=1.0)
scores = cvs(l, dados_X_train, dados_y_train,
             scoring='neg_mean_squared_error',cv = 5)
print(-scores.mean())
from sklearn.linear_model import Ridge
r = Ridge(alpha=1.0)
scores = cvs(r, dados_X_train, dados_y_train,
             scoring='neg_mean_squared_error',cv = 5)
print(-scores.mean())
scores = [6.88523446e+09,6218527332.368642, 6218527273.975189]
x = np.arange(3)
fig, ax = plt.subplots()
plt.bar(x, scores)
plt.xticks(x, ('Knn', 'Ridge', 'Lasso'))
plt.title("RMSE")
plt.show()
numeros_ordenados = sorted(scores)
numeros_ordenados
r.fit(dados_X_train, dados_y_train)
y_pred = r.predict(dados_X_test)
from sklearn.metrics import mean_squared_log_error
y_pred = pd.Series(y_pred)
for i in range(y_pred.shape[0]-1):
    if(y_pred[i] < 0):
        y_pred[i] = 0
y_pred.describe()
from math import log, sqrt
Sum = 0
for i in range(y_pred.shape[0]-1):
    Sum +=  (log(1+y_pred[i]) - log(1+np.array(dados_y_test)[i]))*(log(1+y_pred[i]) - log(1+np.array(dados_y_test)[i]))
score = sqrt(Sum/y_pred.shape[0])
print(score)
Xfut = dadosTeste.drop(["Id","longitude","population","total_bedrooms"], axis=1)
Yfut = r.predict(Xfut)

for i in range(Yfut.shape[0]-1):
    if(Yfut[i] < 0):
        Yfut[i] = 0
        
my_prev = pd.DataFrame(columns = ["Id", "median_house_value"])
my_prev["Id"] = dadosTeste["Id"]
my_prev["median_house_value"] = Yfut
my_prev.to_csv("Evaluation.csv")
my_prev.head()
my_prev.to_csv('pred.csv',index=False, sep=',', line_terminator='\n')
my_prev.head()