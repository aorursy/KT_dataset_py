import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV



from sklearn.preprocessing import StandardScaler



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



from sklearn.pipeline import Pipeline



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier



import os

import warnings

warnings.filterwarnings('ignore')

pd.set_option('precision', 2)
df = pd.read_csv('../input/sonar-data-set/sonar.all-data.csv', header=None)

df.head()   

df.columns
df.hist(sharex=False, sharey=False, xlabelsize=1,ylabelsize=1); plt.show()

df.plot(kind='density', subplots=True, layout=(8,8), sharex=False, sharey=False, legend=False, fontsize=1); plt.show()

df.plot(kind='density', subplots=True, layout=(8,8), sharex=False, sharey=False, legend=False, fontsize=1); plt.show()
x = df.iloc[:,0:60]

y = df.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=7)
models = []

models.append(('LR', LogisticRegression(max_iter=1000)))

models.append(('DT', DecisionTreeClassifier()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('NB', GaussianNB()))

models.append(('SVC', SVC()))
my_names = []

my_results = []



for name, model in models:

    kfold = KFold(n_splits=10, random_state=7)

    cv = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')

    my_names.append(name)

    my_results.append(cv)

    msg = ('%s %f (%f)' % (name, cv.mean(), cv.std()))

    print(msg)    
fig = plt.figure()

fig.suptitle('Comparison of Algorithm on Unscaled Data (Baseline)')

ax = fig.add_subplot(111)

plt.boxplot(my_results)

ax.set_xticklabels(my_names)

plt.show()
pipeline = []

pipeline.append(('ScalerLR', Pipeline([('Scaler', StandardScaler()),('LR',LogisticRegression(max_iter=1000))])))

pipeline.append(('ScalerDT', Pipeline([('Scaler', StandardScaler()),('DT', DecisionTreeClassifier())])))

pipeline.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))

pipeline.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))

pipeline.append(('ScaledSVC', Pipeline([('Scaler', StandardScaler()),('SVC', SVC())])))
my_cv = []

my_names = []



for name, model in pipeline:

    kfold = KFold(n_splits=10, random_state=7)

    cv = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')

    my_names.append(name)

    my_cv.append(cv)

    msg = ('%s %f (%f)' % (name, cv.mean(), cv.std()))

    print(msg)
fig = plt.figure()

fig.suptitle('Comparison of Algorithm on Scaled Data')

ax = fig.add_subplot(111)

plt.boxplot(my_cv)

ax.set_xticklabels(my_names)

plt.show()
scaler = StandardScaler().fit(x_train)

rescaled_x = scaler.transform(x_train)

c_vals = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]

kernel_vals = ['poly', 'rbf', 'linear', 'sigmoid']

param_grid = dict(C=c_vals, kernel=kernel_vals)



model = SVC()

kfold = KFold(n_splits = 10, random_state=7)

grid = GridSearchCV(estimator = model, param_grid=param_grid, scoring = 'accuracy', cv=kfold)

grid_result = grid.fit(rescaled_x, y_train)

    

print('Best score %f using %s' % (grid_result.best_score_, grid_result.best_params_))



means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']



for mean, stdev, param in zip(means, stds, params):

    print('%f (%f) with %r' % (mean, stdev, param))
ensembles = []

ensembles.append(('AdaBoost', AdaBoostClassifier()))

ensembles.append(('GradientBoost', GradientBoostingClassifier()))

ensembles.append(('RandomForest', RandomForestClassifier()))

ensembles.append(('ExtraTrees', ExtraTreesClassifier()))
my_cv = []

my_names = []



for name, model in ensembles:

    kfold = KFold(n_splits=10, random_state=7)

    cv = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')

    my_cv.append(cv)

    my_names.append(name)

    msg = '%s %f (%f)' % (name, cv.mean(), cv.std())

    print(msg)
fig = plt.figure()

fig.suptitle('Comparison of Algorithm on Scaled Data')

ax = fig.add_subplot(111)

plt.boxplot(my_cv)

ax.set_xticklabels(my_names)

plt.show()
scaler = StandardScaler().fit(x_train)

rescaled_x = scaler.transform(x_train)

model = SVC() #using default hyperparameters of C=1.0 and kernel='rbf'

model.fit(rescaled_x, y_train)
rescaled_x_test = scaler.transform(x_test)

predictions = model.predict(rescaled_x_test)

print(accuracy_score(y_test, predictions))

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))