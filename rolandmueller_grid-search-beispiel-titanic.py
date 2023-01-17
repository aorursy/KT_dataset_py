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
from sklearn.impute import SimpleImputer



url = (

    "http://biostat.mc.vanderbilt.edu/"

    "wiki/pub/Main/DataSets/titanic3.xls"

)

df = pd.read_excel(url)





columns_to_drop = ['home.dest', 'body', 'boat', 'embarked', 'cabin', 'name', 'ticket']

df_droped = df.drop(columns= columns_to_drop)





im = SimpleImputer(strategy='median')  

df_droped[['fare', 'age']] = im.fit_transform(df_droped[['fare', 'age']])



cleaned_data = pd.get_dummies(df_droped, drop_first=True)



X = cleaned_data[['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male']]

y = cleaned_data['survived']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")

dummy_clf.fit(X_train, y_train)

dummy_clf.score(X_test, y_test)
y_pred = dummy_clf.predict(X_test)
from sklearn import metrics

auc_score = metrics.roc_auc_score(y_test, y_pred)



print(f"Test AUC score: {auc_score}")
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
params = {

    'max_depth' : [1, 2, 3, 4, 5, 6, None],

    'criterion': ['gini', 'entropy']

}



from sklearn.model_selection import GridSearchCV



grid_clf = GridSearchCV(estimator = clf,

                        param_grid = params,

                        scoring = 'accuracy', 

                        cv = 10, 

                        verbose = 1,

                        n_jobs = -1)
grid_clf.fit(X_train, y_train)
# Identify optimal hyperparameter values

best_criterion      = grid_clf.best_params_['criterion']

best_max_depth = grid_clf.best_params_['max_depth']  



print(f"Best cross-validation score: {grid_clf.best_score_:.3f}")

print(f"The best performing criterion value is: {best_criterion}")

print(f"The best performing max_depth value is: {best_max_depth}")
grid_clf.score(X_test, y_test)
print(grid_clf.cv_results_)
import pandas as pd



grid_search_results = pd.pivot_table(pd.DataFrame(grid_clf.cv_results_),

    values='mean_test_score', index='param_max_depth', columns='param_criterion')



grid_search_results
import seaborn as sns       

ax = sns.heatmap(grid_search_results, annot=True)
clf = DecisionTreeClassifier(criterion=best_criterion, max_depth=best_max_depth)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
auc_score = metrics.roc_auc_score(y_test, y_pred)



print(f"Test AUC score: {auc_score}")

print(metrics.classification_report(y_test, y_pred))
%matplotlib inline

import matplotlib.pyplot as plt

from sklearn.tree import plot_tree

plt.figure(figsize=(20,20))

plot_tree(clf, feature_names=X_train.columns, class_names=['dead', 'survived'], filled = True)
from xgboost import XGBClassifier



clf = XGBClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred))