import pandas as pd

pd.set_option('display.float_format', lambda x: '%.3f' % x)

path_dataset = '../input/datos-properati-limpios-model/datos_properati_limpios_model.csv'

df = pd.read_csv(path_dataset)

df.tail()
# Hacé la separación en esta celda

X = df.drop(['price_aprox_usd'], axis=1)

y = df['price_aprox_usd']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
X_train.shape, \

y_train.shape, \

X_test.shape, \

y_test.shape



#De esta manera, simplemente corroboramos que la cantidad de Train y de Test tienen la misma geometria
# Esto es lo que hace

import numpy as np

np.random.seed(123)

from sklearn.model_selection import train_test_split

X = df.drop(['price_aprox_usd'], axis=1)

y = df['price_aprox_usd']

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42)



print(X_train.shape[0], X_test.shape[0])
# Creá en esta celda la variable param_grid

import numpy as np



values= [1,2,3,4,5]



param_grid = [

    {'max_depth': values, 'max_features': values},

]
# Importa y crea un GridSearchCV en esta celda

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeRegressor



tree_reg = DecisionTreeRegressor()





grid_search = GridSearchCV(tree_reg, param_grid, cv=5,

                           scoring='neg_mean_squared_error', 

                           return_train_score=True)



# Hace el fit de grid search en esta celda



grid_search.fit(X_train, y_train)

grid_search.scorer_
# Mostrá los grid_scores en esta celda

score = grid_search.cv_results_['mean_train_score']

score
# Mostrás los resultados en esta celda

grid_search.best_score_

grid_search.best_params_
def nmsq2rmse(score):

    return np.round(np.sqrt(-score), 2)

nmsq2rmse(score)
nmsq2rmse(score).mean()
#asignamos los distintos hiperparametros al objeto param_grid, estos seran utilizados luego en el grid_search



min_samples_split_hiperparams = [2, 10, 20]

max_depth_hiperparams = [None, 2, 5, 10, 15]

min_samples_leaf_hiperparams = [1, 5, 10, 15]

max_leaf_nodes_hiperparams = [None, 5, 10, 20]



param_grid = [

    {'min_samples_split': min_samples_split_hiperparams, 

     'max_depth': max_depth_hiperparams ,

     'min_samples_leaf': min_samples_leaf_hiperparams,

      'max_leaf_nodes':max_leaf_nodes_hiperparams },

]
#Asignamos los hiperparametros en grid_search con 5 Folds para el Cross Validation, asignamos el tipo de score y le pedimos que devuelva sus resultados

grid_search = GridSearchCV(tree_reg, param_grid, cv=5,

                           scoring='neg_mean_squared_error', 

                           return_train_score=True)
#Fiteamos para asi entrenar

grid_search.fit(X_train, y_train)
#Obtenemos los mejores Hiperparametros para el mejor Score

grid_search.best_params_
score = grid_search.cv_results_['mean_train_score']

grid_search.best_score_
def nmsq2rmse(score):

    return np.round(np.sqrt(-score), 2)

nmsq2rmse(score)
#Caluclamos la media del RMSE para asi comparar con el entrenamiento de mas arriba ... mejoro sustancialmente, bajando el error.

nmsq2rmse(score).mean()
optimised_decision_tree = grid_search.best_estimator_
from sklearn.metrics import mean_squared_error

y_opt_pred = optimised_decision_tree.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_opt_pred))

np.round(rmse)
val_real = pd.Series(y_test.values)

val_pred = pd.Series(y_opt_pred)
predicciones = pd.concat([val_real.rename('Valor real'),val_pred.rename('Valor Pred') ,abs(val_real-val_pred).rename('Dif(+/-)')] ,  axis=1)
predicciones.head(10)
results = pd.DataFrame(grid_search.cv_results_, columns=['rank_test_score','params','mean_test_score','std_test_score','mean_train_score','std_train_score'])

results.sort_values(by=['rank_test_score'],inplace=True)

results.head()