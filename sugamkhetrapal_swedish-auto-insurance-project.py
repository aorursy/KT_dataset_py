# Load libraries
import numpy
from numpy import arange
from numpy import set_printoptions
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')
# Load dataset
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

filename = '/kaggle/input/auto-insurance-in-sweden-small-dataset/insurance.csv'
data = read_csv(filename, skiprows=list(range(0,5)), header=None, names=['claims','payment'])
peek = data.head()
print(peek)
shape = data.shape
print(shape)
types = data.dtypes
print(types)
# Statistical Summary
set_option('display.width', 100)
set_option('precision', 3)
description = data.describe()
print(description)
# Pairwise Pearson correlations
correlations = data.corr(method='pearson')
print(correlations)
# Skew for each attribute
skew = data.skew()
print(skew)
# Univariate Histograms
data.hist()
pyplot.show()
# Univariate Density Plots
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
pyplot.show()
# Box and Whisker Plots
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
pyplot.show()
# Correction Matrix Plot
correlations = data.corr()
# plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
pyplot.show()
# Scatterplot Matrix
g = sns.PairGrid(data, diag_sharey=False)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot, colors="C0")
g.map_diag(sns.kdeplot, lw=2)
# Split-out validation dataset
array = data.values
X = array[:,0:1]
Y = array[:,1]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'
# Spot-Check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
models.append(('Ridge', Ridge()))
models.append(('RidgeCV', RidgeCV(alphas=numpy.logspace(-6, 6, 13))))
models.append(('LassoLars', LassoLars(alpha=0.1)))
models.append(('BayesianRidge', BayesianRidge()))
models.append(('ARDRegression', ARDRegression()))
models.append(('OrthogonalMatchingPursuit', OrthogonalMatchingPursuit()))

for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# Standardize the dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
pipelines.append(('ScaledRidge', Pipeline([('Scaler', StandardScaler()),('Ridge', Ridge())])))
pipelines.append(('ScaledRidgeCV', Pipeline([('Scaler', StandardScaler()),('RidgeCV', RidgeCV(alphas=numpy.logspace(-6, 6, 13)))])))
pipelines.append(('ScaledLassoLars', Pipeline([('Scaler', StandardScaler()),('LassoLars', LassoLars(alpha=0.1))])))
pipelines.append(('ScaledBayesianRidge', Pipeline([('Scaler', StandardScaler()),('BayesianRidge', BayesianRidge())])))
pipelines.append(('ScaledARDRegression', Pipeline([('Scaler', StandardScaler()),('ARDRegression', ARDRegression())])))
pipelines.append(('ScaledOrthogonalMatchingPursuit', Pipeline([('Scaler', StandardScaler()),('OrthogonalMatchingPursuit', OrthogonalMatchingPursuit())])))
results_standardized = []
names_standardized = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results_standardized.append(cv_results)
    names_standardized.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results) # Baseline
ax.set_xticklabels(names) # Baseline
pyplot.show()
# ensembles
ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor())])))
results_ensembles = []
names_ensembles = []
for name, model in ensembles:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results_ensembles.append(cv_results)
    names_ensembles.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# prepare the model
model = RidgeCV(alphas=numpy.logspace(-6, 6, 13))
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print(mean_squared_error(Y_validation, predictions))
# prepare the model
model = LinearRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print(mean_squared_error(Y_validation, predictions))
# prepare the model
model = Lasso() # ElasticNet(), KNeighborsRegressor(), DecisionTreeRegressor(), SVR(), Ridge(), RidgeCV(), LassoLars(), BayesianRidge(), ARDRegression(), OrthogonalMatchingPursuit()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print(mean_squared_error(Y_validation, predictions))