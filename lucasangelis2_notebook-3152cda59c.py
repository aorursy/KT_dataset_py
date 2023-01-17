# Imports
import pandas as pd
import sklearn
import numpy as np
import math
import matplotlib.pyplot as plt
# Leitura base de treino e base de testes

train_db = pd.read_csv("../input/train_data.csv",
                     sep=r'\s*,\s*',
                     engine='python',
                     na_values="?")

test_db = pd.read_csv("../input/test_data.csv",
                     sep=r'\s*,\s*',
                     engine='python',
                     na_values="?")
# Describe da Base de treino
train_db.describe()
# Head da Base de Treino
size_train = train_db.shape
train_db.head()
# Grau de escolaridade
train_db["education"].value_counts().plot(kind="bar");
# Estado Civil
train_db["marital.status"].value_counts().plot(kind="bar");
# Status de relacionamento
train_db["relationship"].value_counts().plot(kind="bar");
# Etnia
train_db["race"].value_counts().plot(kind="bar");
# Sexo
train_db["sex"].value_counts().plot(kind="bar");
# Ocupação
train_db["occupation"].value_counts().plot(kind="bar");
# Análise dos ">50k"
f_rich = train_db["income"] == ">50K" # filter
rich_db = train_db[f_rich][:] # taking filtered data
rich_db.describe() # Describe da rich_db
# Análise dos "<=50k"
f_n_rich = train_db["income"] == "<=50K" # filter
nrich_db = train_db[f_n_rich][:] # taking filtered data
nrich_db.describe() # Describe da rich_db
# Grau de escolaridade dos ricos
rich_db["education"].value_counts().plot(kind="bar"); 
# Estado Civil dos ricos
rich_db["marital.status"].value_counts().plot(kind="bar");
# Etnia dos ricos
rich_db["race"].value_counts().plot(kind="bar");
# Ocupação dos ricos
rich_db["occupation"].value_counts().plot(kind="bar");
# Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest,f_classif
# Tomando Label e Features da base de treino
rndf_X = train_db[["age","fnlwgt","education","education.num","marital.status","relationship","race","sex","capital.gain","capital.loss","hours.per.week"]]
rndf_X = rndf_X.apply(preprocessing.LabelEncoder().fit_transform)
rndf_Y = train_db["income"]
# Utilização do algoritmo SelectKBest para determinação das 'k' colunas que fornecem melhor acurácia para o classificador
selector = SelectKBest(score_func=f_classif, k=5)
# Treinando o seletor
trainX_select = selector.fit_transform(rndf_X,rndf_Y)
# Extraindo da base as 'k' colunas que foram selecionadas pelo algoritmo
ids = selector.get_support(indices = True)
rndf_X_t = rndf_X.iloc[:,ids]
# Definindo classificador
rndf_clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
# Fit na base de treino
rndf_clf.fit(rndf_X_t,rndf_Y)
# Cross-validation para avaliar classificador
scores = cross_val_score(rndf_clf, rndf_X_t, rndf_Y, cv = 5)
print("A media de scores eh: ",np.mean(scores))
# Random Hyperparameter Grid
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 3)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 3)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split =  [ 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [ 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Creating the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
# Use the random grid to search for best hyperparameters
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores

#rf_random = RandomizedSearchCV(estimator = rndf_clf, param_distributions = random_grid, n_iter = 5, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

#rf_random.fit(rndf_X_t, rndf_Y)

#FEITO UMA VEZ SÓ, DEMORA MUITO!!!!!!!!!!!!!!!
# Best parameters
#rf_random.best_params_
# Redefinindo classificador com base nos Best Parameters
rndf_clf = RandomForestClassifier(n_estimators=1100,min_samples_split = 10,min_samples_leaf = 4, max_features = 'sqrt', max_depth=10,bootstrap = True, random_state=0)
# Cross-validation para avaliar classificador
scores = cross_val_score(rndf_clf, rndf_X_t, rndf_Y, cv = 5)
print("A nova media de scores eh: ",np.mean(scores))
# Imports
from sklearn.ensemble import GradientBoostingClassifier
# Tomando Label e Features da base de treino
gb_X = train_db[["age","fnlwgt","education","education.num","marital.status","relationship","race","sex","capital.gain","capital.loss","hours.per.week"]]
gb_X = rndf_X.apply(preprocessing.LabelEncoder().fit_transform)
gb_Y = train_db["income"]
# Extraindo da base as 'k' colunas que foram selecionadas pelo algoritmo
ids = selector.get_support(indices = True)
gb_X_t = gb_X.iloc[:,ids]
# Definindo classificador
gb_clf = GradientBoostingClassifier(n_estimators=20, max_features=2, max_depth = 2, random_state = 0)
# Fit na base de treino
gb_clf.fit(gb_X_t,gb_Y)
# Cross-validation para avaliar classificador
scores = cross_val_score(gb_clf, gb_X_t, gb_Y, cv = 5)
print("A media de scores eh: ",np.mean(scores))
# Random Hyperparameter Grid

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 3)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 3)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [ 2, 4]
# Learning rate
learning_rate = [0.1, 0.5]
# Creating the random grid
random_grid_gb = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'learning_rate': learning_rate}
print(random_grid_gb)
# Use the random grid to search for best hyperparameters
# Random search of parameters, using 3 fold cross validation

#gb_random = RandomizedSearchCV(estimator = gb_clf, param_distributions = random_grid_gb, n_iter = 3, cv = 2, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

#gb_random.fit(gb_X_t, gb_Y)

#FEITO UMA VEZ SÓ, DEMORA MUITO!!!!!
# Best parameters
#gb_random.best_params_
# Redefinindo classificador com base nos best parameters
gb_clf = GradientBoostingClassifier(n_estimators=200, max_features='sqrt', max_depth = 60, min_samples_split = 5,
                                    min_samples_leaf = 4, learning_rate = 0.1)
# Cross-validation para avaliar classificador
scores = cross_val_score(gb_clf, gb_X_t, gb_Y, cv = 5)
print("A nova media de scores eh: ",np.mean(scores))
# Import MPL - Multi-Layer Perceptron
from sklearn.neural_network import MLPClassifier
# Definindo Classificador
nn_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# Tomando Label e Features da base de treino
nn_X = train_db[["age","fnlwgt","education","education.num","marital.status","relationship","race","sex","capital.gain","capital.loss","hours.per.week"]]
nn_X = rndf_X.apply(preprocessing.LabelEncoder().fit_transform)
nn_Y = train_db["income"]
# Extraindo da base as 'k' colunas que foram selecionadas pelo algoritmo
ids = selector.get_support(indices = True)
nn_X_t = nn_X.iloc[:,ids]
# Fit na base de treino
nn_clf.fit(nn_X_t,nn_Y)
# Cross-validation para avaliar classificador
scores = cross_val_score(nn_clf, nn_X_t, nn_Y, cv = 5)
print("A media de scores eh: ",np.mean(scores))
# Random Hyperparameter Grid

# Activation
activation = ["identity","logistic","relu"]
# Solver
solver = ["sgd","adam"]
# Max iter
max_iter = [100, 200, 300]
# Creating the random grid
random_grid_nn = {'activation': activation,
               'solver': solver,
               'max_iter': max_iter}
print(random_grid_nn)
# Use the random grid to search for best hyperparameters
# Random search of parameters, using 3 fold cross validation

#nn_random = RandomizedSearchCV(estimator = nn_clf, param_distributions = random_grid_nn, n_iter = 3, cv = 2, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

#nn_random.fit(nn_X_t, nn_Y)

#FEITO UMA VEZ SÓ, DEMORA MUITO!!!!!
# Best parameters
#nn_random.best_params_
# Redefinindo Classificador com base nos Best Parameters
nn_clf = MLPClassifier(solver='adam', max_iter = 100 , activation = "identity",alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# Cross-validation para avaliar classificador
scores = cross_val_score(nn_clf, nn_X_t, nn_Y, cv = 5)
print("A nova media de scores eh: ",np.mean(scores))