# importar pacotes necessários

import numpy as np

import pandas as pd
# definir parâmetros extras

#pd.set_option('precision', 2)

pd.set_option('display.max_columns', 100)
import matplotlib.pyplot as plt

%matplotlib inline
# importar pacotes usados na seleção do modelo e na medição da precisão

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold



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

filename = '../input/abalone-train.csv'

data = pd.read_csv(filename, index_col='id')



# mostrar tamanho

print(data.shape)



# mostrar alguns exemplos de registros

data.head()
# gerar "one hot encoding" em atributos categóricos

data = pd.get_dummies(data)

data.head()
# encontrar possíveis outliers

outliers = np.concatenate((

    data[(data['height'] < 0.01) | (data['height'] > 0.3)].index,

    data[(data['viscera_weight'] < 0.0001) | (data['viscera_weight'] > 0.6)].index

), axis=0)



# exibir outliers

data[data.index.isin(outliers)].head(10)
# remover outliers detectados

print("Número de outliers a serem removidos: %d" % len(outliers))



print("Antes:", data.shape)

data.drop(outliers, inplace=True)

print("Depois:", data.shape)
# definir dados de entrada

X = data.drop(['rings'], axis=1) # tudo, exceto a coluna alvo

y = data['rings'] # apenas a coluna alvo



print('Forma dos dados originais:', X.shape, y.shape)
from sklearn.metrics import make_scorer, mean_squared_error



# cria função para cálculo do RMSE (REMQ)

def root_mean_squared_error(ground_truth, predictions):

    return mean_squared_error(ground_truth, predictions) ** 0.5



RMSE = make_scorer(root_mean_squared_error, greater_is_better=False)
from datetime import datetime



# avalia o desempenho do modelo, retornando o valor do RMSE

def evaluate_model_cv(model, X=X, y=y):

    start = datetime.now()

    kfold = KFold(n_splits=10, random_state=42)

    results = cross_val_score(model, X, y, cv=kfold, scoring=RMSE, verbose=1)

    end = datetime.now()

    elapsed = int((end - start).total_seconds() * 1000)

    score = (-1) * results.mean()

    stddev = results.std()

    print(model, '\nScore: %.2f (+/- %.2f) [%5s ms]' % (score, stddev, elapsed))

    return score, stddev, elapsed
# faz o ajuste fino do modelo, calculando os melhores hiperparâmetros

def fine_tune_model(model, params, X=X, y=y):

  print('\nFine Tuning Model:')

  print(model, "\nparams:", params)

  kfold = KFold(n_splits=10, random_state=42)

  grid = GridSearchCV(estimator=model, param_grid=params, scoring=RMSE, cv=kfold, verbose=1)

  grid.fit(X, y)

  print('\nGrid Best Score: %.2f' % (grid.best_score_ * (-1)))

  print('Best Params:', grid.best_params_)

  return grid
model = LinearRegression(n_jobs=-1, fit_intercept=True, normalize=True)

evaluate_model_cv(model)



params = dict(

    fit_intercept=[True, False],

    normalize=[True, False]

)

#fine_tune_model(model, params)
model = LogisticRegression(n_jobs=-1, random_state=42, multi_class='auto', C=1000, solver='newton-cg')

evaluate_model_cv(model)



params = dict(

    solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

    C=np.logspace(-3, 3, 7)

)

#fine_tune_model(model, params)
model = OrthogonalMatchingPursuit(n_nonzero_coefs=7, fit_intercept=True, normalize=True)

evaluate_model_cv(model)



params = dict(

    n_nonzero_coefs=[None, 1, 2, 5, 7],

    fit_intercept=[True, False],

    normalize=[True, False]

)

#fine_tune_model(model, params)
model = PassiveAggressiveRegressor(random_state=42, C=0.1, fit_intercept=True, max_iter=1000, tol=0.001)

evaluate_model_cv(model)



params = {

    'C': [0.1, 0.2, 0.4, 0.8, 1.0],

    'fit_intercept': [True, False],

}

#fine_tune_model(model, params)
model = Perceptron(random_state=42, penalty='l2', alpha=1e-3, fit_intercept=True, max_iter=1000, tol=1e-3)

evaluate_model_cv(model)



#penalty=None, alpha=0.0001, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, eta0=1.0,

#n_jobs=None, random_state=0, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, 

#class_weight=None, warm_start=False



params = {

    'penalty': [None, 'l2', 'l1', 'elasticnet'],

    'alpha': np.logspace(-6, -1, 6),

    'fit_intercept': [True, False]

}

#fine_tune_model(model, params)
model = RANSACRegressor(random_state=42, min_samples=0.75)

evaluate_model_cv(model)



#base_estimator=None, min_samples=None, residual_threshold=None, is_data_valid=None, is_model_valid=None,

#max_trials=100, max_skips=inf, stop_n_inliers=inf, stop_score=inf, stop_probability=0.99, loss=’absolute_loss’,

#random_state=None



params = {

    'min_samples': [None, 0.1, 0.25, 0.5, 0.75, 1.0]

}

#fine_tune_model(model, params)
model = Ridge(random_state=42, alpha=0.001, fit_intercept=True, normalize=True)

evaluate_model_cv(model)



#alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver=’auto’,

#random_state=None



params = {

    'alpha': np.logspace(-6, -1, 6),

    'fit_intercept': [True, False],

    'normalize': [True, False]

}

#fine_tune_model(model, params)
model = SGDRegressor(random_state=42, alpha=1e-06, fit_intercept=True, penalty=None, tol=1e-3)

evaluate_model_cv(model)



#loss=’squared_loss’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001,

#shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate=’invscaling’, eta0=0.01, power_t=0.25,

#early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=False, average=False



params = {

    'penalty': [None, 'l2', 'l1', 'elasticnet'],

    'alpha': np.logspace(-6, -1, 6),

    'fit_intercept': [True, False]

}

#fine_tune_model(model, params)
model = TheilSenRegressor(random_state=42, n_jobs=-1, fit_intercept=True)

evaluate_model_cv(model)



#fit_intercept=True, copy_X=True, max_subpopulation=10000.0, n_subsamples=None, 

#max_iter=300, tol=0.001, random_state=None, n_jobs=None, verbose=False



params = {

    'fit_intercept': [True, False]

}

#fine_tune_model(model, params)
model = DecisionTreeRegressor(random_state=42, max_depth=4, min_samples_split=0.25)

evaluate_model_cv(model)



#criterion=’mse’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, 

#min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, 

#min_impurity_decrease=0.0, min_impurity_split=None, presort=False



params = dict(

    max_depth=[4, 6, 8, 10, 12, 14],

    min_samples_split=[0.25, 0.5, 0.75, 1.0]

)

#fine_tune_model(model, params)
model = GaussianProcessRegressor(random_state=42, alpha=0.01, normalize_y=False)

evaluate_model_cv(model)



#kernel=None, alpha=1e-10, optimizer=’fmin_l_bfgs_b’, n_restarts_optimizer=0,

#normalize_y=False, copy_X_train=True, random_state=None



params = dict(

    alpha=np.logspace(-6, -1, 6),

    normalize_y=[True, False]

)

#fine_tune_model(model, params)
model = KernelRidge(alpha=0.1)

evaluate_model_cv(model)



#alpha=1, kernel=’linear’, gamma=None, degree=3, coef0=1, kernel_params=None



params = dict(

    alpha=np.logspace(-6, -1, 6)

)

#fine_tune_model(model, params)
model = GaussianNB(var_smoothing=0.001)

evaluate_model_cv(model)



#priors=None, var_smoothing=1e-09



params = dict(

    var_smoothing=np.logspace(-9, -1, 5)

)

#fine_tune_model(model, params)
model = KNeighborsRegressor(n_jobs=-1, n_neighbors=11, weights='distance')

evaluate_model_cv(model)



#n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’,

#metric_params=None, n_jobs=None



params = {

    'n_neighbors': [1, 3, 5, 7, 9, 11, 13],

    'weights': ['uniform', 'distance']

}

#fine_tune_model(model, params)
model = SVR(gamma='auto', kernel='linear')

evaluate_model_cv(model)



#kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, tol=0.001, C=1.0, 

#epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1



params = {

    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']#, 'precomputed']

}

#fine_tune_model(model, params)
model = MLPRegressor(random_state=42, max_iter=500,

                     activation='tanh', hidden_layer_sizes=(5, 2), solver='lbfgs')

evaluate_model_cv(model)



#hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’, alpha=0.0001, batch_size=’auto’, 

#learning_rate=’constant’, learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, 

#random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 

#early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10



params = dict(

    hidden_layer_sizes=[(100,), (50,), (50,2)],

    activation=['identity', 'logistic', 'tanh', 'relu'],

    solver=['lbfgs', 'sgd', 'adam']

)

#fine_tune_model(model, params)
model = RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=100, max_depth=9)

evaluate_model_cv(model)



#n_estimators=’warn’, criterion=’mse’, max_depth=None, min_samples_split=2, min_samples_leaf=1, 

#min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, 

#min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, 

#verbose=0, warm_start=False



params = {

    'n_estimators': [5, 10, 25, 50, 75, 100],

    'max_depth': [None, 3, 5, 7, 9, 11, 13]

}

#fine_tune_model(model, params)
model = GradientBoostingRegressor(random_state=42, learning_rate=0.05, n_estimators=100,

                                  subsample=0.8, max_depth=6, max_features=0.75)

evaluate_model_cv(model)



#loss=’ls’, learning_rate=0.1, n_estimators=100, subsample=1.0, criterion=’friedman_mse’, min_samples_split=2,

#min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, 

#min_impurity_split=None, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, 

#max_leaf_nodes=None, warm_start=False, presort=’auto’, validation_fraction=0.1, n_iter_no_change=None, 

#tol=0.0001



params = dict(

    n_estimators=[100, 250, 500],

    max_features=[0.75, 0.85, 1.0],

    max_depth=[4, 6, 8, 10],

    learning_rate=[0.05, 0.1, 0.15],

    subsample=[0.4, 0.6, 0.8]

)

#fine_tune_model(model, params)
model = ExtraTreesRegressor(random_state=42, n_jobs=-1, n_estimators=200, max_features=0.75)

evaluate_model_cv(model)



#n_estimators=’warn’, criterion=’mse’, max_depth=None, min_samples_split=2, min_samples_leaf=1, 

#min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, 

#min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0,

#warm_start=False



params = dict(

    n_estimators=[50, 75, 100, 200],

    max_features=['auto', 0.75, 0.85, 1.0]

)

#fine_tune_model(model, params)
model = BaggingRegressor(random_state=42, n_jobs=-1, base_estimator=DecisionTreeRegressor(),

                        max_features=0.75, n_estimators=75)

evaluate_model_cv(model)



#base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, 

#bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0



params = dict(

    n_estimators=[50, 75, 100, 200],

    max_features=[0.5, 0.75, 1.0]

)

#fine_tune_model(model, params)
model = AdaBoostRegressor(random_state=42, n_estimators=200, base_estimator=DecisionTreeRegressor())

evaluate_model_cv(model)



# base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm=’SAMME.R’, random_state=None



params = dict(

    n_estimators=[50, 75, 100, 200]

)

#fine_tune_model(model, params)
from xgboost import XGBRegressor



model = XGBRegressor(random_state=42, n_jobs=-1, learning_rate=0.1,

                     n_estimators=50, max_depth=5, objective='reg:squarederror')

evaluate_model_cv(model)



#max_depth=3, learning_rate=0.1, n_estimators=100, verbosity=1, silent=None, objective='reg:squarederror',

#booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, 

#colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, 

#base_score=0.5, random_state=0, seed=None, missing=None, importance_type='gain'



params = dict(

    max_depth=[3, 5, 7, 9],

    n_estimators=[50, 75, 100, 200]

)

#fine_tune_model(model, params)
# https://towardsdatascience.com/ensemble-learning-using-scikit-learn-85c4531ff86a



# classe não disponível no pacote :P

#from sklearn.ensemble import VotingRegressor

#conda update scikit-learn

#pip install -U scikit-learn



estimators =  [

    ('MLP', MLPRegressor(random_state=42, max_iter=500,

                     activation='tanh', hidden_layer_sizes=(5, 2), solver='lbfgs')),

    ('GPR', GaussianProcessRegressor(random_state=42, alpha=0.01, normalize_y=False)),

    ('GBR', GradientBoostingRegressor(random_state=42, learning_rate=0.05, n_estimators=100,

                                  subsample=0.8, max_depth=6, max_features=0.75))

    #('RFR', RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=100, max_depth=7)),

    #('ETR', ExtraTreesRegressor(random_state=42, n_jobs=-1, n_estimators=100)),

    #('XGBR', XGBRegressor(random_state=42, n_jobs=-1, learning_rate=0.1,

    #             n_estimators=50, max_depth=5, objective='reg:squarederror'))

]



model = VotingRegressor(estimators, n_jobs=-1, weights=(2,1,1))

evaluate_model_cv(model)



#estimators, weights=None, n_jobs=None



params = dict(

    weights=[(1,1,1), (5,4,3), (2,1,1), (3,2,1)]

)

#fine_tune_model(model, params)
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
results = []

names = []

scores = []

stddevs = []

times = []



for name, model in models:

    score, stddev, elapsed = evaluate_model_cv(model, X=X, y=y)

    results.append((score, stddev))

    names.append(name)

    scores.append(score)

    stddevs.append(stddev)

    times.append(elapsed)
# comparar desempenhos dos algoritmos

fig = plt.figure(figsize=(16,8))

fig.suptitle('Comparação dos Algoritmos')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
results_df = pd.DataFrame({'Model': names, 'Score': scores, 'Std Dev': stddevs, 'Time (ms)': times})

results_df.sort_values(by='Score', ascending=True)