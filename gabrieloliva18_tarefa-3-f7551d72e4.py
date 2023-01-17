import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from geopy.distance import distance
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
treino = pd.read_csv("../input/tarefa3/train.csv")
teste = pd.read_csv("../input/tarefa3/test.csv")
treino.head()
treino.info()
ntreino = treino.drop(columns = "Id")
teste_Id = teste.Id
nteste = teste.drop(columns = "Id")
plt.figure(figsize=(10,10)) 
sns.heatmap(ntreino.corr(),annot=True,cmap='cubehelix_r') 
plt.show()
correlacao = ntreino.corr().abs()
s = correlacao.unstack()
so = s.sort_values(kind="heapsort", ascending = False)
n = so.drop_duplicates()
n.head(20)
def derivacoes(df):
    df.loc[:,'mean_rooms'] = df.loc[:,'total_rooms']/df.loc[:,'households']
    df.loc[:,'person_per_room'] = df.loc[:,'population']/df.loc[:,'total_rooms']
    df.loc[:,'mean_bedrooms'] = df.loc[:,'total_bedrooms']/df.loc[:,'households']
    df.loc[:,'bedrooms_per_person'] = df.loc[:,'total_bedrooms']/df.loc[:,'households']
    df.loc[:,'persons_per_household'] = df.loc[:,'population']/df.loc[:,'households']
    df.loc[:, 'median_income_per_person'] = df.loc[:,'median_income']/df.loc[:,'persons_per_household']
derivacoes(ntreino)
derivacoes(nteste)
nteste.head()
plt.figure(figsize=(10,10)) 
sns.heatmap(ntreino.corr(),annot=True,cmap='cubehelix_r') 
plt.show()
correlacao = ntreino.corr().abs()
s = correlacao.unstack()
so = s.sort_values(kind="heapsort", ascending = False)
n = so.drop_duplicates()
n
fig = ntreino.plot.scatter(x='total_bedrooms',y='households',color='red')
fig.set_xlabel("total_bedrooms")
fig.set_ylabel("households")
fig=plt.gcf()
fig.set_size_inches(5,3)
plt.show()
fig = ntreino.plot.scatter(x='median_income_per_person',y='median_house_value',color='blue')
fig.set_xlabel("median_income_per_person")
fig.set_ylabel("median_house_value")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()
def calculamenordistancia (local):
    x = (local['latitude'], local['longitude'])
    cidades = [( 34 +  3/60 + 14/3600, -(118 + 14/60 + 42/3600) ), # Los Angeles
               ( 32 + 46/60 + 46/3600, -(117 +  8/60 + 47/3600) ),  # San Diego
               ( 37 + 20/60 + 7/3600, - (121 + 53/60 + 35/3600)),   # San Jose
               ( 37 + 46/60 + 46/3600, -(122 + 25/60 +  9/3600) )]  # San Francisco
    menordistancia = distance(x, cidades[0]).km
    for i in cidades:
        distancia = distance(x, i)
        if distancia < menordistancia:
            menordistancia = distancia
    return float(str(menordistancia)[:-2])

def adicionamenordistancia(df):
    local = ntreino[['longitude', 'latitude']]
    distancias = np.array( [calculamenordistancia(local.loc[i]) for i in range(local.shape[0]) ])
    dist = pd.DataFrame( distancias, columns=['least_distance'] )
    df['least_distance'] = dist

adicionamenordistancia(ntreino)
adicionamenordistancia(nteste)
correlacao = ntreino.corr().abs()
s = correlacao.unstack()
so = s.sort_values(kind="heapsort", ascending = False)
n = so.drop_duplicates()
n.head(20)
Xtreino = ntreino.drop(columns = ["median_house_value"])
Ytreino = ntreino.median_house_value
knn = KNeighborsRegressor()
grid_knn = {"n_neighbors":[i for i in range(1,40)],"p":[1,2]}
grid = GridSearchCV(knn,grid_knn,cv=10)
grid.fit(Xtreino,Ytreino)
print(grid.best_estimator_)
print(grid.best_score_)
knn = grid.best_estimator_
scores = cross_val_score(knn, Xtreino, Ytreino, cv=10)
np.mean(scores)
reglasso = Lasso()
grid_lasso = {"alpha":np.linspace(0.5,5.5,51).tolist(),"normalize":[True,False]}
grid = GridSearchCV(reglasso,grid_lasso,cv=10)
grid.fit(Xtreino,Ytreino)
print(grid.best_estimator_)
print(grid.best_score_)
reglasso = grid.best_estimator_
scores = cross_val_score(reglasso, Xtreino, Ytreino, cv=10)
np.mean(scores)
regridge = Ridge()
grid_ridge = {"alpha":np.linspace(0.5,10.5,101).tolist()} 
grid = GridSearchCV(regridge,grid_ridge,cv=10)
grid.fit(Xtreino,Ytreino)
print(grid.best_estimator_)
print(grid.best_score_)
regridge = grid.best_estimator_
scores = cross_val_score(regridge, Xtreino, Ytreino, cv=10)
np.mean(scores)
regmlp = MLPRegressor()
grid_mlp = GridSearchCV(regmlp, param_grid={'hidden_layer_sizes': [i for i in range(1,15)],
              'activation': ['relu'],
              'solver': ['adam'],
              'learning_rate': ['constant'],
              'learning_rate_init': [0.001],
              'power_t': [0.5],
              'alpha': [0.0001],
              'max_iter': [200],
              'early_stopping': [False],
              'warm_start': [False]})
grid_mlp.fit(Xtreino, Ytreino)
regmlp = grid_mlp.best_estimator_
scores = cross_val_score(regmlp, Xtreino, Ytreino, cv=10)
np.mean(scores)
YtestPred = knn.predict(nteste)
result = pd.DataFrame()
result['Id'] = teste.Id
result['median_house_value'] = YtestPred
result.to_csv("results_knn.csv", index = False)

result
YtestPred = reglasso.predict(nteste)
result = pd.DataFrame()
result['Id'] = teste.Id
result['median_house_value'] = YtestPred
result.to_csv("results_lasso.csv", index = False)

result
YtestPred = regridge.predict(nteste)
result = pd.DataFrame()
result['Id'] = teste.Id
result['median_house_value'] = YtestPred
result.to_csv("results_ridge.csv", index = False)

result
YtestPred = regmlp.predict(nteste)
result = pd.DataFrame()
result['Id'] = teste.Id
result['median_house_value'] = YtestPred
result.to_csv("results_mlp.csv", index = False)

result