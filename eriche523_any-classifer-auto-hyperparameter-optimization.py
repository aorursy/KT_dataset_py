import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from sklearn import datasets

df = datasets.load_iris()
df=pd.DataFrame({

    'sepal length':df.data[:,0],

    'sepal width':df.data[:,1],

    'petal length':df.data[:,2],

    'petal width':df.data[:,3],

    'species':df.target

})

df.head()
X=df[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features

y=df['species']  # Label
# standarize data



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X=scaler.fit_transform(X)
# a simple pipeline for hyperparameter optimization of all the 5 different models

# (could have wrote a more condense function )



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.externals import joblib

from sklearn.metrics import classification_report



# Load and split the data, into 70/30 training and testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

import warnings

from sklearn.exceptions import DataConversionWarning

warnings.simplefilter(action='ignore', category=DataConversionWarning)

warnings.simplefilter(action='ignore', category=FutureWarning)
# 1. assign a name to the classifier that you want to use

cls_name = {0: 'random_forest', 1: 'logistic_regression', 2: 'support_vector'}



# 2. assign the actual classifier name in sklearn

classifiers = [

    RandomForestClassifier(),

    LogisticRegression(),

    SVC()

    ]

# 3. This is the manual part, specify the corresponding hyperparameters that you want to optimize, hyperparameters are different for each classifer in most cases. Be careful of conditional hyperparameters, you will get an error if not specified correctly

parameters = [{'clf__criterion': ['gini', 'entropy'],

 'clf__n_estimators': [20,30],

'clf__min_samples_leaf': [25,50],

'clf__max_depth': [3,4,5,6,7]},

    

{'clf__C': [0.001, 0.1, 1, 5],

'clf__class_weight': [None,'balanced',{0:0.25, 1:0.75}],

'clf__solver': ['lbfgs', 'liblinear']},

    

{'clf__C': [0.001, 0.1, 1, 5],

'clf__class_weight': [None,'balanced']}

]

    

# Fit the grid search objects

print('hyperparameters grid search in process... ')





#create a placehold for best accuracy and best model parameters, these place must be set within in the loop, not global.

best_val_acc = 0

best_val_clf = []

best_val_gs = []

    

best_test_acc = 0

best_test_clf = []

best_test_gs = []



# there are only 3 parameters to change, the name of the classifier, the actual classifer in sklearn and the corresponding hyperparameters

for idx, classifier, params in zip(cls_name, classifiers, parameters):

    

    clf_pipe = Pipeline([

        ('clf', classifier)

        ])

    gs_clf = GridSearchCV(clf_pipe, param_grid=params, n_jobs=-1)

    

    print('\nEstimator: %s' % cls_name[idx])

    # Fit grid search

    gs_clf.fit(X_train, y_train)

    # Best params

    print('Best params: %s' % gs_clf.best_params_)

    # Best validation data accuracy

    print('Mean cross-validated score of the best_estimator: %.3f' % gs_clf.best_score_)

    # Predict on test data with best params

    y_pred = gs_clf.predict(X_test)

    # Test data accuracy of model with best params and print classification report

    print('Test set accuracy score for best params: %.3f ' % accuracy_score(y_test, y_pred))

    print (classification_report(y_test, y_pred))

    

    # Track best validation accuracy model, default k=5 since v0.22

    if gs_clf.best_score_ > best_val_acc:

        best_val_acc = gs_clf.best_score_

        best_val_gs = gs_clf

        best_val_clf = cls_name[idx]

    # Track best test accuracy model

    if accuracy_score(y_test, y_pred) > best_test_acc:

        best_test_acc = accuracy_score(y_test, y_pred)

        best_test_gs = gs_clf

        best_test_clf = cls_name[idx]

    # note: Often the best validation is the also the best test accuracy model, however in rare instances that may not be the case. If the results of validation between two classifier are really close, then it can easily have a different best testing accuracy classifer. it's up to the data scientist to investigate further the most suitable model to use.