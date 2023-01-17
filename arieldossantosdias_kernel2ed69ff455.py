import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score

base = pd.read_csv("creditcard.csv")


base = base.drop("Time", axis = 1)
base = base.drop("Amount", axis = 1)
fraude = base[base["Class"] == 1]
fraude = fraude.drop("Class", axis = 1)
fraude = fraude.values


previsores = base.iloc[:,0:28].values

classe = base.iloc[:,28].values



def CriarRede(optimizer, loss, kernel_initializer, activation, neurons, dropout):
    classificador = Sequential()
    classificador.add(Dense(units= neurons, activation = activation , kernel_initializer = kernel_initializer, input_dim = 28))
    classificador.add(Dropout(dropout))
    classificador.add(Dense(units= neurons , activation = activation, kernel_initializer = kernel_initializer))
    classificador.add(Dropout(dropout))
    classificador.add(Dense(units= 1, activation = "sigmoid"))
    
    classificador.compile(optimizer = optimizer , loss = loss , metrics = ['binary_accuracy'])
    
    return classificador


classificador = KerasClassifier(build_fn= CriarRede)

parametros = {
    'batch_size':[10, 30],
    'epochs': [5,10],
    'optimizer':["adam","sgd"],
    'loss':["binary_crossentropy","hinge"],
    "kernel_initializer": ["random_uniform", "normal"],
    "activation":["relu", "tanh"],
    "neurons":[16,8],
    "dropout":[0.2,0.4]}

grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           scoring = "accuracy",
                           cv = 5)

grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_



