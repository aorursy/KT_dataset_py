from pandas import read_csv

from pandas import DataFrame

from scipy.stats import normaltest

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from xgboost import XGBClassifier

import numpy as np



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)
import os

os.listdir("../input")
data_train = read_csv("../input/dataset_treino.csv")

data_teste = read_csv("../input/dataset_teste.csv")
data_train.head()
data_teste.head()
print("Número de observações em data_train com Nan: %d " % len(data_train[data_train.isnull().any(axis=1)]))

print("Número de observações em data_teste com Nan: %d " % len(data_teste[data_teste.isnull().any(axis=1)]))
data_train.iloc[:, 1:8].describe()
data_train.iloc[:, 1:8].apply(normaltest)
data_train.corr()
print(data_train.groupby('classe').size())
valores_train = data_train.values



x_train = valores_train[:,1:9].astype(float)

y_train = valores_train[:,9].astype(float)



x_teste = data_teste.iloc[:, 1:9].values.astype(float)
scaler = StandardScaler().fit(x_train)

x_train_normal = scaler.transform(x_train)



scaler = StandardScaler().fit(x_teste)

x_teste_normal = scaler.transform(x_teste)
print("Media de x_train_normal", np.mean(x_train_normal, axis=0, dtype=int))

print("DP de x_train_normal", np.std(x_train_normal, axis=0))
print("Media de x_teste_normal", np.mean(x_teste_normal, axis=0, dtype=int))

print("DP de x_teste_normal", np.std(x_teste_normal, axis=0))
print(x_train_normal.shape)

print(y_train.shape)
modelos = []

modelos.append(('LR', LogisticRegression(), {"C": np.logspace(-3,3,7), "penalty": ["l1","l2"]}))



modelos.append(('KNN', KNeighborsClassifier(), {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]}))



modelos.append(('SVM', SVC(), {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 

                               'C': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]}))



modelos.append(('CART', DecisionTreeClassifier(), {'criterion':['gini','entropy'], 

                                                   'max_depth':[4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 

                                                                40, 50, 70, 90, 120, 150]}))



modelos.append(('LDA', LinearDiscriminantAnalysis(), {}))



modelos.append(('AB', AdaBoostClassifier(), {'n_estimators': [16, 32, 48, 60]}))



modelos.append(('GBM', GradientBoostingClassifier(), {"loss":["deviance"],

                                                      "learning_rate": [0.01, 0.05, 0.1],

                                                      "min_samples_split": np.linspace(0.1, 0.5, 3),

                                                      "min_samples_leaf": np.linspace(0.1, 0.5, 3),

                                                      "max_depth":[3,5],

                                                      "max_features":["log2","sqrt"],

                                                      "criterion": ["friedman_mse",  "mae"],

                                                      "subsample":[0.5, 0.8, 1.0],

                                                      "n_estimators":[5, 10, 15]}))



modelos.append(('RF', RandomForestClassifier(), {'n_estimators': [200, 500, 700],

                                                 'max_features': ['auto', 'sqrt', 'log2'],

                                                 'max_depth' : [4, 7, 8, 10],

                                                 'criterion' :['gini', 'entropy']}))



modelos.append(('ET', ExtraTreesClassifier(), {'n_estimators': [16, 32, 48, 60]}))



modelos.append(('XGB', XGBClassifier(), {'max_depth': [2, 4, 5],

                                         'subsample': [0.4, 0.5, 0.7],

                                         'colsample_bytree': [0.5, 0.7],

                                         'n_estimators': [1000, 2000],

                                         'reg_alpha': [0.01, 0.03]}))



for nome, modelo, valores_grid in modelos:

    

    kfold = model_selection.KFold(n_splits=10, random_state = 7)

    

    grid = model_selection.GridSearchCV(estimator = modelo, 

                                        param_grid = valores_grid, 

                                        cv = kfold, 

                                        scoring = 'accuracy',

                                        return_train_score=True)

    

    grid_result = grid.fit(x_train_normal, y_train)

    print("%s - Melhor Acurácia: %f utilizando %s" % (nome, grid_result.best_score_, grid_result.best_params_))
melhor_modelo = SVC(kernel='linear', C=0.1)
melhor_modelo.fit(x_train_normal, y_train)
previsoes = melhor_modelo.predict(x_teste_normal)
previsoes
# Salvar os resultados para envio

#out_file = open("predictions.csv", "w")

#out_file.write("id,classe\n")



#for i in range(len(previsoes)):

#    out_file.write(str(i+1) + "," + str(int(previsoes[i])) + "\n")

#out_file.close()