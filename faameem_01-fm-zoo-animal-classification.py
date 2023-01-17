import seaborn

import numpy

import sys



from pandas import read_csv

from pandas import set_option

from matplotlib import pyplot



from sklearn.preprocessing import StandardScaler



from sklearn.feature_selection import VarianceThreshold



from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



from sklearn.pipeline import Pipeline



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
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# load file [class.csv] into dataframe [_df_class]

_df_class = read_csv('../input/class.csv')



# load file [zoo.csv] into dataframe [_df_zoo]

_df_zoo = read_csv('../input/zoo.csv')
# first 5 rows of dataframe [_df_class]

_df_class.head()
# first 5 rows of dataframe [_df_zoo]

_df_zoo.head()
# (rows, cols) of dataframe [_df_zoo]

_df_zoo.shape
# data types

_df_zoo.dtypes
set_option('precision',2)
_df_zoo.describe()
_df_zoo.corr(method='pearson')
# class distribution

_df_zoo.groupby('class_type').size()
_df_zoo.plot(kind='density', subplots=True, layout=(4,5), figsize=(13,20), sharex=False, sharey=False)

pyplot.show()
_df_zoo.plot(kind='box', subplots=True, layout=(4,5), figsize=(13,20), sharex=False, sharey=False)

pyplot.show()
def plot_correlation_map( df ):

    corr = df.corr()

    _ , ax = pyplot.subplots( figsize =( 14 , 12 ) )

    cmap = seaborn.diverging_palette( 220 , 10 , as_cmap = True )

    _ = seaborn.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

    )
plot_correlation_map(_df_zoo)
_df_zoo = _df_zoo.drop('hair',axis=1)

_df_zoo = _df_zoo.drop('eggs',axis=1)
plot_correlation_map(_df_zoo)
# column [animal_name] has to be dropped from the dataframe

# if we change the full dataframe to array and then exclude it by _array[:,1:17]

# numpy considers the data type inside the array to be object instead of int64

_df_zoo = _df_zoo.drop('animal_name', axis=1)

_df_zoo.head()
_array = _df_zoo.values
print(_array[:10,:], len(_array), type(_array), _array.shape, _array.ndim, _array.dtype.name)
_X = _array[:,0:14]
print(_X[:10,:], len(_X), type(_X), _X.shape, _X.ndim, _X.dtype.name)
_X = VarianceThreshold(threshold=(.8*(1-.8))).fit_transform(_X)
print(_X[:10,:], len(_X), type(_X), _X.shape, _X.ndim, _X.dtype.name)
_y = _array[:,14:]
print(_y[:5], len(_y), type(_y), _y.shape, _y.ndim, _y.dtype.name)
_y = numpy.ravel(_y)
print(_y[:5], len(_y), type(_y), _y.shape, _y.ndim, _y.dtype.name)
_test_size = 0.20
_random_seed = 7
X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size=_test_size, random_state=_random_seed)
print(X_train, len(X_train), type(X_train), X_train.shape, X_train.ndim, X_train.dtype.name)
print(X_test, len(X_test), type(X_test), X_test.shape, X_test.ndim, X_test.dtype.name)
print(y_train, len(y_train), type(y_train), y_train.shape, y_train.ndim, y_train.dtype.name)
print(y_test, len(y_test), type(y_test), y_test.shape, y_test.ndim, y_test.dtype.name)
_num_folds = 10
_scoring = 'accuracy'
_models = []



# linear algorithms

_models.append(('LR', LogisticRegression())) 

_models.append(('LDA', LinearDiscriminantAnalysis())) 



# non-linear algorithms

_models.append(('KNN', KNeighborsClassifier())) 

_models.append(('CART', DecisionTreeClassifier())) 

_models.append(('NB', GaussianNB())) 

_models.append(('SVM', SVC()))
_results = []

_names = []



for _name, _model in _models:

    _kfold = KFold(n_splits=_num_folds, random_state=_random_seed)

    _cv_results = cross_val_score(_model, X_train, y_train, cv=_kfold, scoring=_scoring)

    _results.append(_cv_results)

    _names.append(_name)

    _msg = '{}: {:.3%}, {:.3f}'.format(_name, _cv_results.mean(), _cv_results.std())

    print(_msg)
# compare algorithms

fig = pyplot.figure() 

fig.suptitle('Algorithm Comparison') 

ax = fig.add_subplot(111) 

pyplot.boxplot(_results) 

ax.set_xticklabels(_names) 

pyplot.show()
# ensembles

ensembles = []



# boosting methods

ensembles.append(('AB', AdaBoostClassifier())) 

ensembles.append(('GBM', GradientBoostingClassifier())) 



# bagging methods

ensembles.append(('RF', RandomForestClassifier())) 

ensembles.append(('ET', ExtraTreesClassifier()))
_results_en = []

_names_en = []



for _name, _model in ensembles:

    _kfold = KFold(n_splits=_num_folds, random_state=_random_seed)

    _cv_results = cross_val_score(_model, X_train, y_train, cv=_kfold, scoring=_scoring)

    _results_en.append(_cv_results)

    _names_en.append(_name)

    _msg = '{}: {:.3%}, {:.3f}'.format(_name, _cv_results.mean(), _cv_results.std())

    print(_msg)
# compare algorithms

fig = pyplot.figure() 

fig.suptitle('Ensemble Algorithm Comparison') 

ax = fig.add_subplot(111) 

pyplot.boxplot(_results_en) 

ax.set_xticklabels(_names_en) 

pyplot.show()
# prepare final model - Gradient Boosting Classifier



_model_final_a = GradientBoostingClassifier()

_model_final_a.fit(X_train, y_train)



# estimate accurary on test data



_predictions = _model_final_a.predict(X_test)

print(accuracy_score(y_test, _predictions))

print(confusion_matrix(y_test, _predictions))

print(classification_report(y_test, _predictions))
# prepare final model - ExtraTreesClassifier



_model_final_b = ExtraTreesClassifier()

_model_final_b.fit(X_train, y_train)



# estimate accurary on test data



_predictions = _model_final_b.predict(X_test)

print(accuracy_score(y_test, _predictions))

print(confusion_matrix(y_test, _predictions))

print(classification_report(y_test, _predictions))