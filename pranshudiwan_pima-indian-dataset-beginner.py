# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
from graphviz import Source
%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Importing key ML files

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score

from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.naive_bayes import BaseNB,GaussianNB

import xgboost as xgb
df = pd.read_csv('../input/diabetes.csv')
df.head()
df.info()
df.Pregnancies = df.Pregnancies.astype(float)
df.Glucose = df.Glucose.astype(float)
df.BloodPressure = df.BloodPressure.astype(float)
df.SkinThickness = df.SkinThickness.astype(float)
df.Insulin = df.Insulin.astype(float)
df.Age = df.Age.astype(float)
df.info()
df.describe()
df.hist(grid = False,figsize=(15,10))
plt.figure(figsize=(15,10))
for i in range(len(df.columns)-1):
    plt.subplot(2,4,i+1)
    sns.boxplot(x="Outcome",y=df.columns[i],data = df,)
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True, fmt='.0%')


X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,:-1],df['Outcome'],random_state = 42)
#decision trees

param_grid = [{"max_leaf_nodes" : np.arange(6,20)},
              {"max_depth" : np.arange(1,10)},
              {"min_samples_leaf" : np.arange(50,150,5)}]
grid1 = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv = 5, scoring = 'roc_auc')
grid1.fit(X_train,y_train)
y_pred = grid1.predict(X_test)

print("\nGrid-Search with AUC")
print("Best parameters:", grid1.best_params_)
print("Best cross-validation score (AUC): {:.3f}".format(grid1.best_score_))
print("Best estimator:\n{}".format(grid1.best_estimator_))

print("Train set AUC: {:.3f}".format(
    roc_auc_score(y_train, grid1.predict(X_train))))
print("Test set AUC: {:.3f}".format(
    roc_auc_score(y_test, grid1.predict(X_test))))

print("Training set accuracy: {:.3f}".format(grid1.score(X_train, y_train)))
print("Test set accuracy: {:.3f}".format(grid1.score(X_test, y_test)))

print(classification_report(y_test,y_pred))

Source(export_graphviz(grid1.best_estimator_, out_file=None, feature_names=X_train.columns, impurity=False, 
                       filled=True))
pipe = make_pipeline( StandardScaler(), PolynomialFeatures(), LogisticRegression())

param_grid = {'polynomialfeatures__degree': [1, 2, 3],
    "logisticregression__C" : np.logspace(-3,3,7)}

grid = GridSearchCV(pipe,param_grid=param_grid,cv=5, scoring = 'roc_auc')

grid.fit(X_train,y_train)
y_pred = grid.predict(X_test)

#copy paste this part
print("\nGrid-Search with AUC")
print("Best parameters:", grid.best_params_)
print("Best cross-validation score (AUC): {:.3f}".format(grid.best_score_))
print("Best estimator:\n{}".format(grid.best_estimator_))

print("Train set AUC: {:.3f}".format(
    roc_auc_score(y_train, grid.predict(X_train))))
print("Test set AUC: {:.3f}".format(
    roc_auc_score(y_test, grid.predict(X_test))))

print("Training set accuracy: {:.3f}".format(grid.score(X_train, y_train)))
print("Test set accuracy: {:.3f}".format(grid.score(X_test, y_test)))

print(classification_report(y_test,y_pred))

print("Logistic regression coefficients:\n{}".format(
      grid.best_estimator_.named_steps["logisticregression"].coef_))
#random forest
param_grid = {'max_features' : np.arange(1,5)}


grid = GridSearchCV(RandomForestClassifier(),param_grid=param_grid,cv=5, scoring = 'roc_auc')

grid.fit(X_train,y_train)
y_pred = grid.predict(X_test)

#copy paste this part
print("\nGrid-Search with AUC")
print("Best parameters:", grid.best_params_)
print("Best cross-validation score (AUC): {:.3f}".format(grid.best_score_))
print("Best estimator:\n{}".format(grid.best_estimator_))

print("Train set AUC: {:.3f}".format(
    roc_auc_score(y_train, grid.predict(X_train))))
print("Test set AUC: {:.3f}".format(
    roc_auc_score(y_test, grid.predict(X_test))))

print("Training set accuracy: {:.3f}".format(grid.score(X_train, y_train)))
print("Test set accuracy: {:.3f}".format(grid.score(X_test, y_test)))

print(classification_report(y_test,y_pred))

print("Feature Importance:\n{}\n".format(
      grid.best_estimator_.feature_importances_))
#xgboost
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

xgbclf = xgb.XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)

grid = GridSearchCV(estimator=xgbclf, param_grid=params, scoring='roc_auc', n_jobs=4, cv=5, verbose=3 )
grid.fit(X_train, y_train)
print('\n All results:')
print(grid.cv_results_)
print('\n Best estimator:')
print(grid.best_estimator_)
print('\n Best score:')
print(grid.best_score_ * 2 - 1)
print('\n Best parameters:')
print(grid.best_params_)

y_pred = grid.predict(X_test)
#copy paste this part


print("Train set AUC: {:.3f}".format(
    roc_auc_score(y_train, clf.predict(X_train))))
print("Test set AUC: {:.3f}".format(
    roc_auc_score(y_test, clf.predict(X_test))))

print("Training set accuracy: {:.3f}".format(clf.score(X_train, y_train)))
print("Test set accuracy: {:.3f}".format(clf.score(X_test, y_test)))

print(classification_report(y_test,y_pred))

