import numpy as np

import pandas as pd

import os

import seaborn as sb

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/iris/Iris.csv')

data.head()
data.info()
data.drop('Id', axis=1, inplace=True)
sb.pairplot(data, hue='Species')
data['Species'] = LabelEncoder().fit_transform(data['Species'])

data.iloc[[0,1,-2,-1],:]



array = data.values

X = array[:,0:4]

Y = array[:,4]
models = []

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='auto')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('SVC', SVC(gamma='scale')))

models.append(('KNC', KNeighborsClassifier(n_neighbors=5)))

models.append(('DTC', DecisionTreeClassifier()))

models.append(('RFC', RandomForestClassifier(n_estimators=100)))

models.append(('GBC', GradientBoostingClassifier()))





# evaluate each model in turn

results = []

names = []

scoring = 'accuracy'

for name, model in models:    

    cv_results = cross_val_score(model, X, Y, cv=10, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)    


c = np.linspace(0.1,1.2,12)

kernels = np.array(['linear', 'rbf'])

gammas = np.array(['auto', 'scale'])

param_gridSvc = dict(C=c, kernel=kernels, gamma=gammas)

modelSvc = SVC()

gridSvc = GridSearchCV(estimator=modelSvc, param_grid=param_gridSvc, cv=10, iid=True)

gridSvc.fit(X, Y)

print(gridSvc.best_score_)

print(gridSvc.best_estimator_.C)

print(gridSvc.best_estimator_.kernel)

print(gridSvc.best_estimator_.gamma)
solvers = np.array(['svd', 'lsqr', 'eigen'])

param_gridLda = dict(solver=solvers)

modelLda = LinearDiscriminantAnalysis()

gridLda = GridSearchCV(estimator=modelLda, param_grid=param_gridLda, cv=10, iid=True)

gridLda.fit(X, Y)

print(gridLda.best_score_)

print(gridLda.best_estimator_.solver)