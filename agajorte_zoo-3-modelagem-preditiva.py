# importar pacotes necessários

import numpy as np

import pandas as pd
# definir parâmetros extras

#pd.set_option('precision', 2)

pd.set_option('display.max_columns', 100)
import matplotlib.pyplot as plt

%matplotlib inline
# importar pacotes usados na seleção do modelo e na medição da precisão

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix



# importar os pacotes necessários para os algoritmos de classificação

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Perceptron

from sklearn.linear_model import Ridge

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
# carregar arquivo de dados de treino

data = pd.read_csv('../input/zoo-train.csv', index_col='animal_name')



# mostrar tamanho

print(data.shape)



# mostrar alguns exemplos de registros

data.head()
# carregar arquivo de dados de treino

data2 = pd.read_csv('../input/zoo-train2.csv', index_col='animal_name')



# mostrar tamanho

print(data2.shape)



# mostrar alguns exemplos de registros

data2.head()
# unir ambos os dados de treinamento

data = data.append(data2)



# mostrar tamanho

print(data.shape)



# mostrar alguns exemplos de registros

data.tail()
# transformar y/n em 0/1



bool_cols = data.columns.values.tolist()

bool_cols.remove('legs')

bool_cols.remove('class_type')



for col in bool_cols:

    data[col] = data[col].map({'n': 0, 'y': 1}).astype(int)
# definir dados de entrada



X = data.drop(['class_type'], axis=1) # tudo, exceto a coluna alvo

y = data['class_type'] # apenas a coluna alvo



print('Forma dos dados originais:', X.shape, y.shape)
def evaluate_model_cv(model, X=X, y=y):

  kfold = KFold(n_splits=10, random_state=42)

  results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy', verbose=1)

  score = results.mean() * 100

  stddev = results.std() * 100

  print(model, '\nCross-Validation Score: %.2f (%.2f) %%' % (score, stddev))

  return score
# faz o ajuste fino do modelo, calculando os melhores hiperparâmetros

def fine_tune_model(model, params, X=X, y=y):

  print('\nFine Tuning Model:')

  print(model, "\nparams:", params)



  kfold = KFold(n_splits=10, random_state=42)

  grid = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=kfold, verbose=1)

  grid.fit(X, y)



  print('\nGrid Score: %.2f %%' % (grid.best_score_ * 100))

  print('Best Params:', grid.best_params_)

 

  return grid
# A) Logistic Regression

model = LogisticRegression(random_state=42, solver='lbfgs', multi_class='auto', max_iter=500, C=100)

#TODO: testar LogisticRegression(multi_class='multinomial', solver='newton-cg')

evaluate_model_cv(model)



params = {'solver':['liblinear', 'lbfgs'], 'C':np.logspace(-3,3,7)}

#fine_tune_model(model, params)
# B) Decision Tree

model = DecisionTreeClassifier(random_state=42, criterion='gini', max_depth=11)

evaluate_model_cv(model)



params = {'criterion':['gini','entropy'], 'max_depth':[3,5,7,11]}

#fine_tune_model(model, params)
# C) K-Nearest Neighbours

model = KNeighborsClassifier(n_neighbors=1)

evaluate_model_cv(model)



params = {'n_neighbors':[1, 3, 5, 7, 9]}

#fine_tune_model(model, params)
# D) Support Vector Machine (SVM)

model = SVC(random_state=42, C=10, gamma=0.1, kernel='rbf')

# SVC(kernel='linear', C=1)

evaluate_model_cv(model)



params = {'C':[0.001, 0.01, 0.1, 1, 10, 100], 'gamma':[0.001, 0.01, 0.1, 1, 10, 100], 'kernel':['linear', 'rbf']}

#fine_tune_model(model, params)
# E) Random Forest

model = RandomForestClassifier(random_state=42, max_features='auto', n_estimators=10)

evaluate_model_cv(model)



params = {'n_estimators':[10, 50, 100, 500], 'max_features':['auto', 'sqrt', 'log2']}

#fine_tune_model(model, params)
# F) Stochastic Gradient Descent (SGD)

model = SGDClassifier(random_state=42, max_iter=100, tol=0.1)

evaluate_model_cv(model)



params = {'max_iter':[100, 200, 350, 500, 1000], 'tol':[0.1, 0.01]}

#fine_tune_model(model, params)
# G) Perceptron

model = Perceptron(random_state=42, max_iter=100, tol=0.01)

# Perceptron(eta0=1, random_state=1)

evaluate_model_cv(model)



params = {'max_iter':[100, 200, 350, 500, 1000], 'tol':[0.1, 0.01, 0.001]}

#fine_tune_model(model, params)
# H) Naïve Bayes

model = GaussianNB(priors=None, var_smoothing=1e-08)

evaluate_model_cv(model)



params = {'priors': [None], 'var_smoothing': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]}

#fine_tune_model(model, params)
# I) Linear SVM

model = LinearSVC(random_state=42, max_iter=1000, C=10)

evaluate_model_cv(model)



params = {'C':[0.001, 0.01, 0.1, 1, 10, 100]}

#fine_tune_model(model, params)
# J) Ada Boost

model = AdaBoostClassifier(DecisionTreeClassifier(random_state=42), n_estimators=5)

evaluate_model_cv(model)



params = {'n_estimators':[1,3,5,7,11]}

#fine_tune_model(model, params)
# K) Gradient Boosting

model = GradientBoostingClassifier(random_state=42, max_depth=3)

evaluate_model_cv(model)



'''

params = {

    "learning_rate":[0.01, 0.05, 0.1],

    "max_depth":[3, 5, 7],

    "max_features":["log2", "sqrt"],

    "criterion":["friedman_mse", "mae"],

    "subsample":[0.5, 0.75, 1.0],

}

'''



params = {'max_depth':[3, 5, 7]}

#fine_tune_model(model, params)
# M) Multi-Layer Perceptron (MLP)

model = MLPClassifier(random_state=42, solver='lbfgs', alpha=0.1, hidden_layer_sizes=(15,))

evaluate_model_cv(model)



params = {'alpha':[1,0.1,0.01,0.001,0.0001,0]}

#fine_tune_model(model, params)
# N) Linear Discriminant Analysis (LDA)

model = LinearDiscriminantAnalysis(solver='svd')

evaluate_model_cv(model)



params = {'solver':['svd', 'lsqr']} #, 'eigen']}

#fine_tune_model(model, params)
models = []

models.append(('LR', LogisticRegression(random_state=42, solver='lbfgs', multi_class='auto', max_iter=500, C=100)))

models.append(('DT', DecisionTreeClassifier(random_state=42, criterion='gini', max_depth=11)))

models.append(('KNN', KNeighborsClassifier(n_neighbors=1)))

models.append(('SVM', SVC(random_state=42, C=10, gamma=0.1, kernel='rbf')))

models.append(('RF', RandomForestClassifier(random_state=42, max_features='auto', n_estimators=10)))

models.append(('SGD', SGDClassifier(random_state=42, max_iter=100, tol=0.1)))

models.append(('NN', Perceptron(random_state=42, max_iter=100, tol=0.01)))

models.append(('NB', GaussianNB(priors=None, var_smoothing=1e-08)))

models.append(('LSVM', LinearSVC(random_state=42, max_iter=1000, C=10)))

models.append(('ABDT', AdaBoostClassifier(DecisionTreeClassifier(random_state=42), n_estimators=5)))

models.append(('GB', GradientBoostingClassifier(random_state=42, max_depth=3)))

models.append(('MLP', MLPClassifier(random_state=42, solver='lbfgs', alpha=0.1, hidden_layer_sizes=(15,))))

models.append(('LDA', LinearDiscriminantAnalysis(solver='svd')))
results = []

names = []

scores = []

stddevs = []



for name, model in models:

  kfold = KFold(n_splits=10, random_state=42)

  cv_results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

  results.append(cv_results)

  names.append(name)

  scores.append(cv_results.mean() * 100)

  stddevs.append(cv_results.std() * 100)

  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

  print(msg)
# boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
results = pd.DataFrame({'Model': names, 'Score': scores, 'Std Dev': stddevs})

results.sort_values(by='Score', ascending=False)