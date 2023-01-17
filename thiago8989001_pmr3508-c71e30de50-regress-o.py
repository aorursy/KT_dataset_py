import numpy as np 
import pandas as pd 
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings
import matplotlib.pyplot as plt

# Importando Base - Treino e Teste:
treino = pd.read_csv("../input/train.csv",
        engine='python')

teste = pd.read_csv("../input/test.csv",
        engine='python')
treino.shape
# Visualizando amostra de dados da base:
treino.sample(n=10)
# Separação das variáveis (X) e target (Y):
Y = treino[["median_house_value"]]
X = treino.iloc[:,1:9]
# RandomForestRegressor with Select K Best - avaliando número ideal de features para minimizar o RMSLE:
reg = RandomForestRegressor(n_estimators=20, criterion='mse', max_depth=None, min_samples_split=2, 
                            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                            max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, 
                            oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False)
warnings.filterwarnings('ignore')
score_list = []
maxscore = -1000000
kmax = 0
scaler = MinMaxScaler()
for num in range(1, 9):
    select = SelectKBest(score_func=f_regression, k=num)
    Xselect = select.fit_transform(X, Y)
    Xscaler = scaler.fit_transform(Xselect)
    scores = cross_val_score(reg, Xscaler, Y, cv=10, scoring='neg_mean_squared_log_error')
    score_list.append(scores.mean())
    if scores.mean() >= maxscore:
        maxscore = scores.mean()
        kmax = num
plt.plot(np.arange(1, 9), score_list)
plt.ylabel('Mean Squared Log Error')
plt.xlabel('Number of Features')
plt.show()
print(maxscore, kmax)
# RandomForestRegressor - avaliando número ideal de árvores na floresta para minimizar o RMSLE:
warnings.filterwarnings('ignore')
score_list = []
maxscore = -1000000
emax = 0
for num in range(1, 40):
    reg = RandomForestRegressor(n_estimators=num, criterion='mse', max_depth=None, min_samples_split=2, 
                            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                            max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, 
                            oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False)
    scores = cross_val_score(reg, X, Y, cv=10, scoring='neg_mean_squared_log_error')
    score_list.append(scores.mean())
    if scores.mean() >= maxscore:
        maxscore = scores.mean()
        emax = num
plt.plot(np.arange(1, 40), score_list)
plt.ylabel('Mean Squared Log Error')
plt.xlabel('Number of Trees')
plt.show()
print(maxscore, emax)
# RandomForestRegressor - avaliando a profundidade máxima das árvores para minimizar o RMSLE:
warnings.filterwarnings('ignore')
score_list = []
maxscore = -1000000
dmax = 0
for num in range(1, 40):
    reg = RandomForestRegressor(n_estimators=38, criterion='mse', max_depth=num, min_samples_split=2, 
                            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                            max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, 
                            oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False)
    scores = cross_val_score(reg, X, Y, cv=10, scoring='neg_mean_squared_log_error')
    score_list.append(scores.mean())
    if scores.mean() >= maxscore:
        maxscore = scores.mean()
        dmax = num
plt.plot(np.arange(1, 40), score_list)
plt.ylabel('Mean Squared Log Error')
plt.xlabel('Max Depth')
plt.show()
print(maxscore, dmax)
# K-Neighbors Regressor - avaliando o número ideal de vizinhos para minimizar o RMSLE:
warnings.filterwarnings('ignore')
score_list = []
maxscore = -1000000
kmax = 0
for num in range(1, 20):
    knn = KNeighborsRegressor(n_neighbors=num)
    scores = cross_val_score(knn, X, Y, cv=10, scoring='neg_mean_squared_log_error')
    score_list.append(scores.mean())
    if scores.mean() >= maxscore:
        maxscore = scores.mean()
        kmax = num
plt.plot(np.arange(1, 20), score_list)
plt.ylabel('Mean Squared Log Error')
plt.xlabel('K Neighbors')
plt.show()
print(maxscore, kmax)
X_test = teste.iloc[:,1:9]
Ids = teste['Id']
reg = RandomForestRegressor(n_estimators=38, criterion='mse', max_depth=22, min_samples_split=2, 
                            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                            max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, 
                            oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False)
scaler = MinMaxScaler()
X1 = scaler.fit_transform(X,Y)
reg.fit(X1,Y)
X_test_final = scaler.transform(X_test)
pred = reg.predict(X_test_final)
submission = pd.DataFrame({'Id':Ids,'median_house_value':pred})
submission.to_csv('submission.csv',index = False)