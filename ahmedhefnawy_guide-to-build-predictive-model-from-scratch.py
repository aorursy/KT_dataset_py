# scipy

import scipy

print('scipy: {}'.format(scipy.__version__))

# numpy

import numpy

print('numpy: {}'.format(numpy.__version__))

# matplotlib

import matplotlib

print('matplotlib: {}'.format(matplotlib.__version__))

# pandas

import pandas

print('pandas: {}'.format(pandas.__version__))

# scikit-learn

import sklearn

print('sklearn: {}'.format(sklearn.__version__))
import pandas as pd

DF_1 = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

DF_1
DF_1.head(10)
DF_1.shape
DF_1.dtypes
DF_1.info()

print("_________________________\n_________________________")

DF_1.describe()
DF_1.groupby('Outcome').size()
DF_1.corr(method='spearman')

# DF_1.corr(method='pearson')

# DF_1.corr(method='kendall')
DF_1.skew()
from matplotlib import pyplot

DF_1.hist(figsize=(16,16))

pyplot.show()
# Univariate Density Plots

from matplotlib import pyplot

DF_1.plot(kind='density', subplots=True, layout=(3,3), sharex=False)

pyplot.show()
DF_1.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)

pyplot.show()
# Correction Matrix Plot

from matplotlib import pyplot

from pandas import read_csv

import numpy



correlations = DF_1.corr()

# plot correlation matrix

fig = pyplot.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(correlations, vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = numpy.arange(0,9,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(DF_1)

ax.set_yticklabels(DF_1)

pyplot.show()
correlations = DF_1.corr()

# plot correlation matrix

fig = pyplot.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(correlations, vmin=-1, vmax=1)

fig.colorbar(cax)

pyplot.show()
from pandas.plotting import scatter_matrix

scatter_matrix(DF_1)

pyplot.show()
# Rescale data (between 0 and 1)

from pandas import read_csv

from numpy import set_printoptions

from sklearn.preprocessing import MinMaxScaler

array = DF_1.values

# separate array into input and output components

X = array[:,0:8]

Y = array[:,8]

#print(Y)

scaler = MinMaxScaler(feature_range=(0, 1))

rescaledX = scaler.fit_transform(X)

# summarize transformed data0

set_printoptions(precision=3)

print(rescaledX[0:5,:])
# Standardize data (0 mean, 1 stdev)

from sklearn.preprocessing import StandardScaler

from pandas import read_csv

from numpy import set_printoptions

array = DF_1.values

# separate array into input and output components

X_input = array[:,0:8]

Y_output = array[:,8]

scaler = StandardScaler().fit(X_input)

rescaledX = scaler.transform(X_input)

# summarize transformed data

set_printoptions(precision=4) # how to diaplay the otput

print(rescaledX[0:5,:])
# Normalize data (length of 1)

from sklearn.preprocessing import Normalizer

from pandas import read_csv

from numpy import set_printoptions

array = DF_1.values



# separate array into input and output components

X = array[:,0:8]

#print(X)

#print(Y)

Y = array[:,8]

#print(X[0:5,:],'\n','___________________','\n')



scaler = Normalizer(norm='l1').fit(X)

print(scaler)

normalizedX = scaler.transform(X)



# summarize transformed data

set_printoptions(precision=3)

print(normalizedX[0:5,:])
# Normalize data (length of 1)

from sklearn.preprocessing import Normalizer

from pandas import read_csv

from numpy import set_printoptions

array = DF_1.values



# separate array into input and output components

X = array[:,0:8]

Y = array[:,8]

#print(X[0:5,:],'\n','___________________','\n')

scaler = Normalizer(norm='l2').fit(X)

normalizedX = scaler.transform(X)



# summarize transformed data

set_printoptions(precision=3)

print(normalizedX[0:5,:])
# binarization

from sklearn.preprocessing import Binarizer

from pandas import read_csv

from numpy import set_printoptions

array = DF_1.values

# separate array into input and output components

X_input = array[:,0:8]

Y = array[:,8]

binarizer = Binarizer(threshold=0.0).fit(X_input)

binaryX = binarizer.transform(X_input)

# summarize transformed data

set_printoptions(precision=3)

print(binaryX[0:5,:])
from numpy import set_printoptions

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

# load data

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]



# feature extraction

test = SelectKBest(score_func=chi2, k=4)

fit = test.fit(X, Y)



# summarize scores

set_printoptions(precision=3)

print(fit.scores_)

features = fit.transform(X)



# summarize selected features

print(features[0:5,:])

#print(features)
# Feature Extraction with RFE

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression



# load data

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]



# feature extraction

model = LogisticRegression()

rfe = RFE(model, 3)

fit = rfe.fit(X, Y)

print("Num Features: {} \nEstimator : {} \n ____________________________\n".format(fit.n_features_ , fit.estimator))

print("Selected Features: {}".format(fit.support_))

print("Feature Ranking: {}".format(fit.ranking_))
# Feature Extraction with PCA

from sklearn.decomposition import PCA

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

# feature extraction

pca = PCA(n_components=3)

fit = pca.fit(X)

# summarize components

print("Explained Variance: {}".format(fit.explained_variance_ratio_) ,'_________________\n') 

print(fit.components_)
# Feature Importance with Extra Trees Classifier

from pandas import read_csv

from sklearn.ensemble import ExtraTreesClassifier

# load data

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

# feature extraction

model = ExtraTreesClassifier()

model.fit(X, Y)

print(model.feature_importances_)
# Evaluate using a train and a test set

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

test_size = 0.33

seed = 7

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,

random_state=seed)

model = LogisticRegression()

model.fit(X_train, Y_train)

result = model.score(X_test, Y_test)

print("Accuracy: {} % ".format(result*100.0)) 
# Evaluate using Cross Validation

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression



array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

num_folds = 10

seed = 7

kfold = KFold(n_splits=num_folds, random_state=seed)

model = LogisticRegression()

results = cross_val_score(model, X, Y, cv=kfold)

print("Accuracy: {} %  [ {} %] ".format(results.mean()*100.0, results.std()*100.0))
# Evaluate using Leave One Out Cross Validation

from sklearn.model_selection import LeaveOneOut

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression



array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

num_folds = 10

loocv = LeaveOneOut()

model = LogisticRegression()  

results = cross_val_score(model, X, Y, cv=loocv)

print("Accuracy: {} % [ {} %]".format(results.mean()*100.0, results.std()*100.0))
# Evaluate using Shuffle Split Cross Validation



from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression



array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

n_splits = 10

test_size = 0.33

seed = 7

kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)

model = LogisticRegression()

results = cross_val_score(model, X, Y, cv=kfold)

print("Accuracy: {} % --  [{} %]".format(results.mean()*100.0, results.std()*100.0))

# >>> Accuracy: 76.496% (1.698%)
import pandas as pd

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

DF_1 = pd.read_csv('F:\Careers\Machine Learning\Data Stets--\diabetes.csv')

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

kfold = KFold(n_splits=10, random_state=7)

model = LogisticRegression()

scoring = 'accuracy'

results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())
# Cross Validation Classification LogLoss

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

kfold = KFold(n_splits=10, random_state=7)

model = LogisticRegression()

scoring = 'neg_log_loss'

results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print("Logloss: %.3f (%.3f)") % (results.mean(), results.std())
# Cross Validation Classification ROC AUC

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

kfold = KFold(n_splits=10, random_state=7)

model = LogisticRegression()

scoring = 'roc_auc'

results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print("AUC: %.3f (%.3f)") % (results.mean(), results.std())
# Cross Validation Classification Confusion Matrix

from pandas import read_csv

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix



array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

test_size = 0.33

seed = 7

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,random_state=seed)

model = LogisticRegression()

model.fit(X_train, Y_train)

predicted = model.predict(X_test)

matrix = confusion_matrix(Y_test, predicted)

print(matrix)
# Cross Validation Classification Report

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

test_size = 0.33

seed = 7

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,

random_state=seed)

model = LogisticRegression()

model.fit(X_train, Y_train)

predicted = model.predict(X_test)

report = classification_report(Y_test, predicted)

print(report)
# Cross Validation Regression MAE

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

array = DF_1.values

X = array[:,0:13]

Y = array[:,13]

kfold = KFold(n_splits=10, random_state=7)

model = LinearRegression()

scoring = 'neg_mean_absolute_error'

results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print("MAE: %.3f (%.3f)") % (results.mean(), results.std())
# Cross Validation Regression MSE

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

array = DF_1.values

X = array[:,0:13]

Y = array[:,13]

num_folds = 10

kfold = KFold(n_splits=10, random_state=7)

model = LinearRegression()

scoring = 'neg_mean_squared_error'

results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print("MSE: %.3f (%.3f)") % (results.mean(), results.std())
# Cross Validation Regression R^2

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

array = DF_1.values

X = array[:,0:13]

Y = array[:,8]

kfold = KFold(n_splits=10, random_state=7)

model = LinearRegression()

scoring = 'r2'

results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print("R^2: %.3f (%.3f)") % (results.mean(), results.std())
# Logistic Regression Classification

import pandas as pd

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

DF_1 = pd.read_csv('F:\Careers\Machine Learning\Data Stets--\diabetes.csv')

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

num_folds = 10

kfold = KFold(n_splits=10, random_state=7)

model = LogisticRegression()

results = cross_val_score(model, X, Y, cv=kfold)

print(results.mean())
# LDA Classification

from pandas import read_csv

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

num_folds = 10

kfold = KFold(n_splits=10, random_state=7)

model = LinearDiscriminantAnalysis()

results = cross_val_score(model, X, Y, cv=kfold)

print(results.mean())
# KNN Classification

from pandas import read_csv

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier



dataframe = read_csv(filename, names=names)

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

num_folds = 10

kfold = KFold(n_splits=10, random_state=7)

model = KNeighborsClassifier()

results = cross_val_score(model, X, Y, cv=kfold)

print(results.mean())
# Gaussian Naive Bayes Classification

from pandas import read_csv

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

kfold = KFold(n_splits=10, random_state=7)

model = GaussianNB()

results = cross_val_score(model, X, Y, cv=kfold)

print(results.mean())
# CART Classification

from pandas import read_csv

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

kfold = KFold(n_splits=10, random_state=7)

model = DecisionTreeClassifier()

results = cross_val_score(model, X, Y, cv=kfold)

print(results.mean())
# SVM Regression

from pandas import read_csv

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVR

array = DF_1.values

X = array[:,0:13]

Y = array[:,13]

num_folds = 10

kfold = KFold(n_splits=10, random_state=7)

model = SVR()

scoring = 'neg_mean_squared_error'

results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print(results.mean())
# Compare Algorithms

from matplotlib import pyplot

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



# load dataset

#filename = 'pima-indians-diabetes.data.csv'

#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]



# prepare models

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))



# evaluate each model in turn

results = []

names = []

scoring = 'accuracy'

for name, model in models:

    kfold = KFold(n_splits=10, random_state=7)

    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    print("{}: {} ({})".format( name, cv_results.mean(), cv_results.std() ) )



# boxplot algorithm comparison

fig = pyplot.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

pyplot.boxplot(results)

ax.set_xticklabels(names)

pyplot.show()
# Create a pipeline that standardizes the data then creates a model

from pandas import read_csv

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# load data

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

# create pipeline

estimators = []

estimators.append(('standardize', StandardScaler()))

estimators.append(('lda', LinearDiscriminantAnalysis()))

model = Pipeline(estimators)

# evaluate pipeline

kfold = KFold(n_splits=10, random_state=7)

results = cross_val_score(model, X, Y, cv=kfold)

print(results.mean())
# Create a pipeline that extracts features from the data then creates a model

from pandas import read_csv

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectKBest

# load data

filename = 'pima-indians-diabetes.data.csv'

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

dataframe = read_csv(filename, names=names)

array = dataframe.values

X = array[:,0:8]

Y = array[:,8]

# create feature union

features = []

features.append(('pca', PCA(n_components=3)))

features.append(('select_best', SelectKBest(k=6)))

feature_union = FeatureUnion(features)

# create pipeline

estimators = []

estimators.append(('feature_union', feature_union))

estimators.append(('logistic', LogisticRegression()))

model = Pipeline(estimators)

# evaluate pipeline

kfold = KFold(n_splits=10, random_state=7)

results = cross_val_score(model, X, Y, cv=kfold)

print(results.mean())
# Bagged Decision Trees for Classification

from pandas import read_csv

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

seed = 7

kfold = KFold(n_splits=10, random_state=seed)

cart = DecisionTreeClassifier()

num_trees = 100

model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)

results = cross_val_score(model, X, Y, cv=kfold)

print(results.mean())
# Random Forest Classification

from pandas import read_csv

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

num_trees = 100

max_features = 3

kfold = KFold(n_splits=10, random_state=7)

model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)

results = cross_val_score(model, X, Y, cv=kfold)

print(results.mean())
# Extra Trees Classification

from pandas import read_csv

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import ExtraTreesClassifier

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

num_trees = 100

max_features = 7

kfold = KFold(n_splits=10, random_state=7)

model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)

results = cross_val_score(model, X, Y, cv=kfold)

print(results.mean())
# AdaBoost Classification

from pandas import read_csv

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import AdaBoostClassifier

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

num_trees = 30

seed=7

kfold = KFold(n_splits=10, random_state=seed)

model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)

results = cross_val_score(model, X, Y, cv=kfold)

print(results.mean())
# Stochastic Gradient Boosting Classification

from pandas import read_csv

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import GradientBoostingClassifier

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

seed = 7

num_trees = 100

kfold = KFold(n_splits=10, random_state=seed)

model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)

results = cross_val_score(model, X, Y, cv=kfold)

print(results.mean())
# Voting Ensemble for Classification

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

kfold = KFold(n_splits=10, random_state=7)

# create the sub models

estimators = []

model1 = LogisticRegression()

estimators.append(('logistic', model1))

model2 = DecisionTreeClassifier()

estimators.append(('cart', model2))

model3 = SVC()

estimators.append(('svm', model3))

# create the ensemble model

ensemble = VotingClassifier(estimators)

results = cross_val_score(ensemble, X, Y, cv=kfold)

print(results.mean())
# Grid Search for Algorithm Tuning

import numpy

from pandas import read_csv

from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])

param_grid = dict(alpha=alphas)

model = Ridge()

grid = GridSearchCV(estimator=model, param_grid=param_grid)

grid.fit(X, Y)

print('the optimal score achieved :',grid.best_score_,'\n')

print('set of parameters in the grid that achieved that score (ALPHA):',grid.best_estimator_.alpha,'\n')
# Randomized for Algorithm Tuning

import numpy

from pandas import read_csv

from scipy.stats import uniform

from sklearn.linear_model import Ridge

from sklearn.model_selection import RandomizedSearchCV

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

param_grid = {'alpha': uniform()}

model = Ridge()

rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100,

random_state=7)

rsearch.fit(X, Y)

print('example produces results much like those in the grid search example above. An optimal alpha value near 1.0 is discovered \n=======\n')

print('the optimal score achieved := ',rsearch.best_score_)

print('the set of parameters in the grid that achieved that score -Alpha- := ',rsearch.best_estimator_.alpha)
# Save Model Using Pickle

from pandas import read_csv

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from pickle import dump

from pickle import load

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=7)



# Fit the model on 33%

model = LogisticRegression()

model.fit(X_train, Y_train)



# save the model to disk

filename = 'finalized_model.sav'

dump(model, open(filename, 'wb'))

# some time later...

# load the model from disk

loaded_model = load(open(filename, 'rb'))

result = loaded_model.score(X_test, Y_test)

print('estimate of accuracy of the model on unseen data = \n','\t\t\t\t\t\t',result)
# Save Model Using joblib

from pandas import read_csv

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.externals.joblib import dump

from sklearn.externals.joblib import load

array = DF_1.values

X = array[:,0:8]

Y = array[:,8]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

# Fit the model on 33%

model = LogisticRegression()

model.fit(X_train, Y_train)

# save the model to disk

filename = 'finalized_model.sav'

dump(model, filename)

# some time later...

# load the model from disk

loaded_model = load(filename)

result = loaded_model.score(X_test, Y_test)

print(result)