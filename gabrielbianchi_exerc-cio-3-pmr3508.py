#Imports
import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing

#Base
data = pd.read_csv("../input/atividade-3-pmr3508/train.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
data.head()
#Dados sem informações geográficas
data_no_geo = data.drop(columns=['Id','longitude','latitude'])#,'median_house_value'])

#Target
y = data['median_house_value']

import seaborn as sns
import matplotlib.pyplot as plt
corr_mat=data_no_geo.corr(method='pearson')
plt.figure(figsize=(20,10))
sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')
t = plt.title('Matriz de Correlações entre as variáveis da base', fontsize=18)
data.insert(len(data_no_geo.columns)-1,'pop_per_household',data_no_geo['population']/data_no_geo['households'])
data.insert(len(data_no_geo.columns)-1,'rooms_per_household',data_no_geo['total_rooms']/data_no_geo['households'])
data.insert(len(data_no_geo.columns)-1,'bedrooms_per_household',data_no_geo['total_bedrooms']/data_no_geo['households'])
data.head()
corr_mat=data_no_geo.corr(method='pearson')
plt.figure(figsize=(20,10))
sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')
t = plt.title('Matriz de Correlações entre as variáveis da base', fontsize=18)
from sklearn.model_selection import train_test_split

#Conjunto sem variaveis geograficas e sem as variaveis novas
X_no_geo = data.drop(columns=['median_house_value','latitude','longitude'])

#Conjunto sem variaveis geograficas e COM as variaveis novas
X_new_features = data.drop(columns=['median_house_value','latitude','longitude'])

#Conjunto com as variaveis originais (incluindo geograficas)
X_original = data.drop(columns=['median_house_value','pop_per_household','rooms_per_household','bedrooms_per_household'])

#Conjunto COM variaveis geo e COM variaveis novas
X_new_features_geo = data.drop(columns=['median_house_value'])

#Separacao treino e teste dos conjuntos
X_train_no_geo, X_test_no_geo, y_train_no_geo, y_test_no_geo = train_test_split(X_no_geo, y, test_size=0.2, random_state=42)
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new_features, y, test_size=0.2, random_state=42)
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_original, y, test_size=0.2, random_state=42)
X_train_new_geo, X_test_new_geo, y_train_new_geo, y_test_new_geo = train_test_split(X_new_features_geo, y, test_size=0.2, random_state=42)

X_test_new.head()
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
a = 1.0
#Treinamento dos modelos
ridge_model_no_geo, ridge_model_new, ridge_model_orig, ridge_model_inc =linear_model.Ridge(alpha=a), linear_model.Ridge(alpha=a), linear_model.Ridge(alpha=a), linear_model.Ridge(alpha=a)
ridge_model_new_geo = linear_model.Ridge(alpha=a)
ridge_model_no_geo.fit(X_train_no_geo,y_train_no_geo)
ridge_model_new.fit(X_train_new,y_train_new)
ridge_model_orig.fit(X_train_orig,y_train_orig)
ridge_model_new_geo.fit(X_train_new_geo,y_train_new_geo)
y_pred_no_geo=ridge_model_no_geo.predict(X_test_no_geo)
y_pred_new=ridge_model_new.predict(X_test_new)
y_pred_orig=ridge_model_orig.predict(X_test_orig)
y_pred_new_geo = ridge_model_new_geo.predict(X_test_new_geo)
#Exibição dos Resultados
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
print('Ridge Regression')
print('----------------------')
print('alpha:',ridge_model_orig.alpha)
print('----------------------')
print('Dataset c/ variaveis orignais')
print('Explained variance:',explained_variance_score(y_pred_orig,y_test_orig))
print('r2 score:', r2_score(y_pred_orig,y_test_orig))
print('----------------------')
print('Dataset sem variaveis geograficas')
print('Explained variance:',explained_variance_score(y_pred_no_geo,y_test_no_geo))
print('r2 score:', r2_score(y_pred_no_geo,y_test_no_geo))
print('----------------------')
print('Dataset sem variaveis geograficas e c/ variaveis novas')
print('Explained variance:',explained_variance_score(y_pred_new,y_test_new))
print('r2 score:', r2_score(y_pred_new,y_test_new))
print('----------------------')
print('Dataset c/ variaveis geograficas e c/ variaveis novas')
print('Explained variance:',explained_variance_score(y_pred_new_geo,y_test_new_geo))
print('r2 score:', r2_score(y_pred_new_geo,y_test_new_geo))
print('----------------------')

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
a = 1.0
#Treinamento dos modelos
lasso_model_no_geo, lasso_model_new, lasso_model_orig, lasso_model_inc =linear_model.Lasso(alpha=a), linear_model.Lasso(alpha=a), linear_model.Lasso(alpha=a), linear_model.Lasso(alpha=a)
lasso_model_new_geo = linear_model.Lasso(alpha=a)
lasso_model_no_geo.fit(X_train_no_geo,y_train_no_geo)
lasso_model_new.fit(X_train_new,y_train_new)
lasso_model_orig.fit(X_train_orig,y_train_orig)
lasso_model_new_geo.fit(X_train_new_geo,y_train_new_geo)
y_pred_no_geo=lasso_model_no_geo.predict(X_test_no_geo)
y_pred_new=lasso_model_new.predict(X_test_new)
y_pred_orig=lasso_model_orig.predict(X_test_orig)
y_pred_new_geo = lasso_model_new_geo.predict(X_test_new_geo)
#Exibição de resultados
print('Lasso Regression')
print('----------------------')
print('alpha:',lasso_model_orig.alpha)
print('----------------------')
print('Dataset c/ variaveis orignais')
print('Explained variance:',explained_variance_score(y_pred_orig,y_test_orig))
print('r2 score:', r2_score(y_pred_orig,y_test_orig))
print('----------------------')
print('Dataset sem variaveis geograficas')
print('Explained variance:',explained_variance_score(y_pred_no_geo,y_test_no_geo))
print('r2 score:', r2_score(y_pred_no_geo,y_test_no_geo))
print('----------------------')
print('Dataset sem variaveis geograficas e c/ variaveis novas')
print('Explained variance:',explained_variance_score(y_pred_new,y_test_new))
print('r2 score:', r2_score(y_pred_new,y_test_new))
print('----------------------')
print('Dataset c/ variaveis geograficas e c/ variaveis novas')
print('Explained variance:',explained_variance_score(y_pred_new_geo,y_test_new_geo))
print('r2 score:', r2_score(y_pred_new_geo,y_test_new_geo))
print('----------------------')
from sklearn.tree import DecisionTreeRegressor
regr_tree = DecisionTreeRegressor(max_depth=45)
regr_tree.fit(X_train_new_geo, y_train_new_geo)
y_pred_new_geo = regr_tree.predict(X_test_new_geo)
print('Primeiro teste com max_depth=45')                                   
print('Explained variance:',explained_variance_score(y_pred_new_geo,y_test_new_geo))
print('r2 score:', r2_score(y_pred_new_geo,y_test_new_geo))
r2_scores = []
for depth in range(1,500):
    regr_tree = DecisionTreeRegressor(max_depth=depth)
    regr_tree.fit(X_train_new_geo, y_train_new_geo)
    y_pred_new_geo = regr_tree.predict(X_test_new_geo)
    r2_scores.append(r2_score(y_pred_new_geo,y_test_new_geo))
plt.plot(r2_scores)
t = plt.title("Variação do r2 score com a variação da profundidade", fontsize=14)
np.argmax(r2_scores)
print('Teste com max_depth=11 ')                                   
print('Explained variance:',explained_variance_score(y_pred_new_geo,y_test_new_geo))
print('r2 score:', r2_score(y_pred_new_geo,y_test_new_geo))
