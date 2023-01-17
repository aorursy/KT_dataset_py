# importar pacotes necessários

import numpy as np

import pandas as pd
# importar os pacotes necessários para os algoritmos de regressão

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import VotingRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.linear_model import Lars

from sklearn.linear_model import LassoLars

from sklearn.linear_model import OrthogonalMatchingPursuit

from sklearn.linear_model import PassiveAggressiveRegressor

from sklearn.linear_model import BayesianRidge

from sklearn.linear_model import ARDRegression

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDRegressor

from sklearn.linear_model import Perceptron

from sklearn.linear_model import PassiveAggressiveRegressor

from sklearn.linear_model import RANSACRegressor

from sklearn.linear_model import TheilSenRegressor

from sklearn.linear_model import HuberRegressor

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neighbors import RadiusNeighborsRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.svm import SVR

from sklearn.svm import NuSVR

from sklearn.svm import LinearSVR

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor
# carregar arquivo de dados de treino

train_data = pd.read_csv('../input/abalone-train.csv', index_col='id')
# carregar arquivo de dados de teste

test_data = pd.read_csv('../input/abalone-test.csv', index_col='id')
# gerar "one hot encoding" em atributos categóricos

train_data = pd.get_dummies(train_data)

test_data = pd.get_dummies(test_data)
# encontrar e remover possíveis outliers

data = train_data

outliers = np.concatenate((

    data[(data['height'] < 0.01) | (data['height'] > 0.3)].index,

    data[(data['viscera_weight'] < 0.0001) | (data['viscera_weight'] > 0.6)].index

), axis=0)

train_data.drop(outliers, inplace=True)
# definir dados de treino



X_train = train_data.drop(['rings'], axis=1) # tudo, exceto a coluna alvo

y_train = train_data['rings'] # apenas a coluna alvo



print('Forma dos dados de treino:', X_train.shape, y_train.shape)
# definir dados de teste



X_test = test_data # tudo, já que não possui a coluna alvo



print('Forma dos dados de teste:', X_test.shape)
models = []



# Generalized Linear Models

models.append(('LinReg', LinearRegression(n_jobs=-1, fit_intercept=True, normalize=True)))

models.append(('LogReg', LogisticRegression(n_jobs=-1, random_state=42, multi_class='auto', C=1000, solver='newton-cg')))

models.append(('OMP', OrthogonalMatchingPursuit(n_nonzero_coefs=7, fit_intercept=True, normalize=True)))

models.append(('PAR', PassiveAggressiveRegressor(random_state=42, C=0.1, fit_intercept=True, max_iter=1000, tol=0.001)))

models.append(('PP', Perceptron(random_state=42, penalty='l2', alpha=1e-3, fit_intercept=True, max_iter=1000, tol=1e-3)))

models.append(('RANSAC', RANSACRegressor(random_state=42, min_samples=0.75)))

models.append(('Ridge', Ridge(random_state=42, alpha=0.001, fit_intercept=True, normalize=True)))

models.append(('SGD', SGDRegressor(random_state=42, alpha=1e-6, fit_intercept=True, penalty=None, tol=1e-3)))

models.append(('TSR', TheilSenRegressor(random_state=42, n_jobs=-1, fit_intercept=True)))



# Decision Trees

models.append(('DTR', DecisionTreeRegressor(random_state=42, max_depth=4, min_samples_split=0.25)))



# Gaussian Processes

models.append(('GPR', GaussianProcessRegressor(random_state=42, alpha=0.01, normalize_y=False)))



# Kernel Ridge Regression

models.append(('KRR', KernelRidge(alpha=0.1)))



# Naïve Bayes

models.append(('GNB', GaussianNB(var_smoothing=0.001)))



# Nearest Neighbors

models.append(('kNN', KNeighborsRegressor(n_jobs=-1, n_neighbors=11, weights='distance')))



# Support Vector Machines

models.append(('SVM', SVR(gamma='auto', kernel='linear')))



# Neural network models

models.append(('MLP', MLPRegressor(random_state=42, max_iter=500,

                     activation='tanh', hidden_layer_sizes=(5, 2), solver='lbfgs')))



# Ensemble Methods

models.append(('RFR', RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=100, max_depth=9)))

models.append(('GBR', GradientBoostingRegressor(random_state=42, learning_rate=0.05, n_estimators=100,

                                  subsample=0.8, max_depth=6, max_features=0.75)))

models.append(('ETR', ExtraTreesRegressor(random_state=42, n_jobs=-1, n_estimators=200, max_features=0.75)))

models.append(('BDTR', BaggingRegressor(random_state=42, n_jobs=-1, base_estimator=DecisionTreeRegressor(),

                                       max_features=0.75, n_estimators=75)))

models.append(('ABDTR', AdaBoostRegressor(random_state=42, n_estimators=200, base_estimator=DecisionTreeRegressor())))



# XGBoost

models.append(('XGBR', XGBRegressor(random_state=42, n_jobs=-1, learning_rate=0.1,

                                    n_estimators=50, max_depth=5, objective='reg:squarederror')))



# Voting

models.append(('VR', VotingRegressor(estimators=[

    ('MLP', MLPRegressor(random_state=42, max_iter=500,

                     activation='tanh', hidden_layer_sizes=(5, 2), solver='lbfgs')),

    ('GPR', GaussianProcessRegressor(random_state=42, alpha=0.01, normalize_y=False)),

    ('GBR', GradientBoostingRegressor(random_state=42, learning_rate=0.05, n_estimators=100,

                                  subsample=0.8, max_depth=6, max_features=0.75))

], n_jobs=-1, weights=(2,1,1))))
!mkdir submissions
sufixo_arquivo = '05set'



for name, model in models:

    print(model, '\n')

    

    # treinar o modelo

    model.fit(X_train, y_train)

    

    # executar previsão usando o modelo

    y_pred = model.predict(X_test)

    

    # gerar dados de envio (submissão)

    submission = pd.DataFrame({

      'id': X_test.index,

      'rings': y_pred

    })

    submission.set_index('id', inplace=True)



    # gerar arquivo CSV para o envio

    filename = 'abalone-submission-p-%s-%s.csv' % (sufixo_arquivo, name.lower())

    submission.to_csv(filename)
# verificar conteúdo dos arquivos gerados

!head abalone-submission-p-*.csv