# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
ds = pd.read_csv("../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
ds.head()
ds.dtypes
ds.shape
ds.info()
ds.describe()
ds.isnull().mean() * 100
%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt
ds.groupby(['Education']).size().plot(kind='bar')

plt.show()
ds.groupby(['EnvironmentSatisfaction']).size().plot(kind='bar')

plt.show()
ds.groupby(['RelationshipSatisfaction']).size().plot(kind='bar')

plt.show()
ds.groupby(['WorkLifeBalance']).size().plot(kind='bar')

plt.show()
ds.groupby(['YearsAtCompany']).size().plot(kind='bar')

plt.show()
ds.groupby(['Gender']).size().plot(kind='bar')

plt.show()
ds.groupby(['Gender','WorkLifeBalance']).size().plot(kind='bar')

plt.show()
ds.groupby(['JobLevel','Gender']).size().plot(kind='bar')

plt.show()
clear_column = ['EmployeeCount', 'EmployeeNumber']

cleared = ds.drop(columns= clear_column)

cleared = cleared.drop(cleared[cleared['TotalWorkingYears'] < 5].index)
working_years = cleared['TotalWorkingYears']

working_years.sort_values()
create_dummys = pd.get_dummies(cleared)

create_dummys

create_dummys.info()
clear_column = ['Attrition_No','Gender_Female','OverTime_Yes']

done_ds = create_dummys.drop(columns= clear_column)

done_ds.info()
clear_column = ['Attrition_Yes']

X = done_ds.drop(columns= clear_column)

X.head()

X.info()
y = done_ds['Attrition_Yes']

y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 36)
from sklearn.dummy import DummyClassifier

dummy_model = DummyClassifier(strategy="most_frequent")

dummy_model.fit(X_train, y_train)

dummy_model.score(X_test, y_test)
y_prediction = dummy_model.predict(X_test)
from sklearn import metrics

auc_score = metrics.roc_auc_score(y_test, y_prediction)

print(f"AUC score: {auc_score}")
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
prms = {'max_depth':[1, 2, 3, 4, 5, 6, 7],

        'splitter':['best', 'random'],

        'max_features':['auto', 'sqrt', 'log2', None],

        'criterion':['gini', 'entropy']

        }
from sklearn.model_selection import GridSearchCV

grid_model = GridSearchCV(estimator = model,

                        param_grid = prms,

                        scoring = 'precision', 

                        cv = 10, 

                        verbose = 1,

                        n_jobs = -1

                        )
grid_model.fit(X_train, y_train)
# Identify optimal hyperparameter values

opt_criterion      = grid_model.best_params_['criterion']

opt_max_features = grid_model.best_params_['max_features'] 

opt_splitter = grid_model.best_params_['splitter'] 

opt_max_depth = grid_model.best_params_['max_depth'] 

 

print(f"Optimal cross-validation score: {grid_model.best_score_:.3f}")

print(f"Optimal performing criterion: {opt_criterion}")

print(f"Optimal performing max_features: {opt_max_features}")

print(f"Optimal performing splitter: {opt_splitter}")

print(f"Optimal performing max_depth: {opt_max_depth}")
model = DecisionTreeClassifier(criterion=best_criterion, 

                                max_depth=best_max_depth,

                                max_features = best_max_features,

                                splitter=best_splitter,

                                class_weight = {0:1,1:7}

                                )
model.fit(X_train, y_train)

y_prediction = model.predict(X_test)

auc_score = metrics.roc_auc_score(y_test, y_prediction)

print(f"AUC score: {auc_score}")

print(metrics.classification_report(y_test, y_prediction))
from yellowbrick.classifier import ConfusionMatrix

cmax = ConfusionMatrix(model, classes=['no', 'yes'],

                        label_encoder={0: 'no', 1: 'yes'}

                        )

cmax.score(X_test, y_test)

cmax.show()
%matplotlib inline

import matplotlib.pyplot as plt

from sklearn.tree import plot_tree

plt.figure(figsize = (20,20))

plot_tree(model, feature_names=X_train.columns, class_names = ['nicht_kündigen', 'kündigen'], filled = True)
prms = {'n_estimators':range(20,81,10),

         'max_depth':range(5,9,3),

         'min_samples_split':range(1000,2100,200),

         'min_samples_leaf':range(10,70,20),

         'max_features':[range(7,20,2),None],

         'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],

         'class_weight':[{0:1, 1:1},{0:1, 1:19}]

         }
from sklearn.model_selection import GridSearchCV                         

from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV                         

model = GridSearchCV(estimator = model,

                    param_grid = prms,

                    scoring = 'recall', 

                    cv = 10, 

                    verbose = 1,

                    n_jobs = -1

                    ) 
model = XGBClassifier()

model.fit(X_train, y_train)

y_prediction = model.predict(X_test)

print(metrics.classification_report(y_test, y_prediction))
from yellowbrick.classifier import ConfusionMatrix

cmax = ConfusionMatrix(

            model, classes = ['no', 'yes'],

            label_encoder = {0: 'no', 1: 'yes'}

            )

cmax.score(X_test, y_test)

cmax.show()
from sklearn.metrics import plot_roc_curve

rcurve = plot_roc_curve(model, X_test, y_test)

plt.show()