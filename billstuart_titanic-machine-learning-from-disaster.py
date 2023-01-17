import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
my_data = pd.read_csv('../input/train.csv')
print(my_data.shape)
# now it's time to clear up the fields we won't use
print(my_data.shape)
to_drop = ['Name', 'PassengerId', 'Ticket', 'Cabin']
new_data = my_data.drop(to_drop, axis = 1)
print(new_data.shape)
# Find categorical features
from sklearn import preprocessing
categorical = new_data.select_dtypes(include=['object'])
numeric = new_data.select_dtypes(exclude=['object'])
print(categorical.columns.values)
# create dummy variables
for name, values in categorical.items():
    print(name)
    dummies = pd.get_dummies(values.str.strip(), prefix = name, dummy_na=True)
    numeric = pd.concat([numeric, dummies], axis=1)
# imputation
for name in numeric:
    print(name)
    if pd.isnull(numeric[name]).sum() > 0:
        numeric["%s_mi" % (name)] = pd.isnull(numeric[name])
        median = numeric[name].median()
        numeric[name] = numeric[name].apply(lambda x: median if pd.isnull(x) else x)     
y = numeric['Survived']
X = numeric.drop(['Survived'], axis = 1)
from sklearn import ensemble
import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# choose the model
my_tree = ensemble.RandomForestClassifier(criterion='gini'
#                                           , max_depth=None
#                                           , min_samples_split=2
#                                           , min_samples_leaf=1
#                                           , max_features='auto'
#                                           , max_leaf_nodes=None
#                                           , bootstrap=True
#                                           , oob_score=False
#                                           , n_jobs=-1
#                                           , random_state=None
#                                           , verbose=0
#                                           , warm_start=False
#                                           , class_weight=None
#                                           , min_weight_fraction_leaf=0.0
#                                           , min_impurity_split=1e-07
                                         )
# set up cv
from sklearn import model_selection
cv = model_selection.KFold(5)
# pipeline
from sklearn.pipeline import Pipeline
pipeline = Pipeline(steps=[('standardize', preprocessing.StandardScaler())
                           , ('model', my_tree) ])
# tune the model
estimators = [50, 100]
max_features = [4, 5, 6, 7]
max_depth = [3, 4, 5, 64, 6, 8, 10, 12]
print(estimators)
# my_alpha = np.logspace(0.001, 1, num = 10)
from sklearn.model_selection import GridSearchCV
optimized_lr = GridSearchCV(estimator=pipeline
                            , cv=cv
                            , param_grid=dict(model__n_estimators = estimators, model__max_features = max_features, model__max_depth=max_depth)
                            , scoring = 'roc_auc'
                            , verbose = 1
                            , n_jobs = -1
                           )
optimized_lr.fit(X_train, y_train)
print(optimized_lr.best_estimator_)
# print(optimized_lr.cv_results_)
# evaluate on holdout
from sklearn.metrics import roc_auc_score
y_pred = optimized_lr.predict_proba(X_test)[:, 1]
roc_on_holdout = roc_auc_score(y_test, y_pred)
print(roc_on_holdout)
# train model on entire dataset
final = pipeline.fit(X, y)
# create a holdout
import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# choose the model
from sklearn.svm import SVC
svm = SVC(probability=True)
# set up cv
from sklearn import model_selection
cv = model_selection.KFold(5)
# pipeline
from sklearn.pipeline import Pipeline
pipeline = Pipeline(steps=[('standardize', preprocessing.StandardScaler())
                           , ('model', svm) ])
# tune the model
my_max_iter = [-1]
#my_max_features = [5,10]
#my_max_depth = [4,6,8]
from sklearn.model_selection import GridSearchCV
optimized_svm = GridSearchCV(estimator=pipeline
                            , cv=cv
                            , param_grid =dict(model__max_iter = my_max_iter)
                            , scoring = 'roc_auc'
                            , verbose = 1
                            , n_jobs = -1
                           )
optimized_svm.fit(X_train, y_train)
print(optimized_svm.best_estimator_)
# print(optimized_lr.cv_results_)
# evaluate on holdout
from sklearn.metrics import roc_auc_score
y_pred = optimized_svm.predict_proba(X_test)[:, 1]
roc_on_holdout = roc_auc_score(y_test, y_pred)
print(roc_on_holdout)
# train model on entire dataset
final = pipeline.fit(X, y)
# create test
import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# choose the model
from sklearn import ensemble
gboost = ensemble.GradientBoostingClassifier()
# set up cv
from sklearn import model_selection
cv = model_selection.KFold(5)
# pipeline
from sklearn.pipeline import Pipeline
pipeline = Pipeline(steps=[('standardize', preprocessing.StandardScaler())
                           , ('model', gboost) ])
# tune the model
my_learning_rate = [.05,.1,.15,.3,.45, .6, .75, .90]
my_max_depth = [3, 4, 5, 6, 7, 8]
from sklearn.model_selection import GridSearchCV
optimized_gboost = GridSearchCV(estimator=pipeline
                            , cv=cv
                            , param_grid=dict(model__learning_rate = my_learning_rate, model__max_depth = my_max_depth)
                            , scoring = 'roc_auc'
                            , verbose = 1
                            , n_jobs = -1
                           )
optimized_gboost.fit(X_train, y_train)
print(optimized_gboost.best_estimator_)
# print(optimized_lr.cv_results_)
# evaluate on test
from sklearn.metrics import roc_auc_score
y_pred = optimized_gboost.predict_proba(X_test)[:, 1]
roc_on_test = roc_auc_score(y_test, y_pred)
print(roc_on_test)
# train model on entire dataset
final = pipeline.fit(X, y)