# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from sklearn import cross_validation, metrics   

from sklearn.grid_search import GridSearchCV  

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/loan-prediction/LoanStats3a.csv")

dataset['int_rate'] = dataset.int_rate.astype('float')
col_count = dataset.isnull().sum()

col_count = pd.DataFrame(col_count)

col_name = col_count.index[col_count[0]>10000].tolist()

#dataset = dataset.drop(col_names[0],axis=1)

#dataset.shape
dataset = dataset.drop(col_name,axis=1)

dataset.shape

unknown = dataset[dataset.loan_status.isnull()]

dataset = dataset[pd.notnull(dataset['loan_status'])]

dataset.shape
del_cols = [ 'policy_code',  'zip_code', 'addr_state',
            'pymnt_plan','emp_title','application_type','acc_now_delinq','title',
            'collections_12_mths_ex_med','collection_recovery_fee']

dataset = dataset.drop(del_cols,axis=1)

Cat_cols = dataset.select_dtypes(include=['object']).columns
# Process columns and apply LabelEncoder to categorical features
for c in Cat_cols:
    dataset[c] = dataset[c].fillna(dataset[c].mode()[0])
    
    

num_cols = dataset.select_dtypes(include=['int64','float64']).columns

for c in num_cols:
    dataset[c] = dataset[c].fillna(dataset[c].mean())

    #train_dataset[c] = lbl.transform(list(train_dataset[c].values))

#dataset.isnull().sum()

print('Total Features: ', len(Cat_cols), 'categorical', '+',
      len(num_cols), 'numerical', '=', len(Cat_cols)+len(num_cols), 'features')
dataset['loan_status'] = dataset['loan_status'].replace('Does not meet the credit policy. Status:Fully Paid','Fully Paid')

dataset['loan_status'] = dataset['loan_status'].replace('Does not meet the credit policy. Status:Charged Off','Charged Off')
from sklearn.preprocessing import LabelEncoder

# Process columns and apply LabelEncoder to categorical features
for c in Cat_cols:
    lbl = LabelEncoder() 
    lbl.fit(list(dataset[c].values)) 
    dataset[c] = lbl.transform(list(dataset[c].values))

dataset.head()
cols = list(dataset)

for c in num_cols:
    dataset[c] = (dataset[c] - dataset[c].mean()) / (dataset[c].max() - dataset[c].min())
    
dataset.head()
from sklearn.model_selection import train_test_split

col_count = dataset.isnull().sum()

col_count = pd.DataFrame(col_count)

col_name = col_count.index[col_count[0]>10000].tolist()

#dataset = dataset.drop(col_names[0],axis=1)

#dataset.shape
dataset = dataset.drop(col_name,axis=1)

train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)

#dataset.isnull().sum()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

train_dataset_Y = train_dataset.loan_status.values

train_dataset_X = train_dataset.drop('loan_status',axis =1)

train_dataset_X.shape

test_dataset_Y = test_dataset.loan_status.values

test_dataset = test_dataset.drop('loan_status',axis=1)

classifier = LogisticRegression(random_state=0)

classifier.fit(train_dataset_X,train_dataset_Y)

y_pred = classifier.predict(test_dataset)

confusion_matrix = confusion_matrix(test_dataset_Y, y_pred)

print(confusion_matrix)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(test_dataset, test_dataset_Y)))
from sklearn.metrics import classification_report

print(classification_report(test_dataset_Y, y_pred))
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn import tree

dtc = DecisionTreeClassifier(min_samples_split=4, random_state=0)

sample_split_range = list(range(2, 50))

param_grid = dict(min_samples_split=sample_split_range)

grid = GridSearchCV(dtc, param_grid, cv=10, scoring='accuracy')

grid.fit(train_dataset_X,train_dataset_Y)

print(grid.best_score_)

print(grid.best_params_)

print(grid.best_estimator_)

clf_gini = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=12,
            min_weight_fraction_leaf=0.0, presort=False, random_state=0,
            splitter='best')

clf_gini.fit(train_dataset_X,train_dataset_Y)

y_pred = clf_gini.predict(test_dataset)

print("Accuracy of Decision Tree classifier on test set is ", accuracy_score(test_dataset_Y,y_pred))
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90],
    'max_features': [2, 3],
    'min_samples_leaf': [3],
    'min_samples_split': [8, 10],
    'n_estimators': [100, 200]
}

rf = RandomForestClassifier()

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(train_dataset_X,train_dataset_Y)

grid_search.best_params_
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(bootstrap= True,max_depth = 90, max_features = 3, min_samples_leaf = 3, min_samples_split = 12, n_estimators = 200)

clf.fit(train_dataset_X,train_dataset_Y)

y_pred = clf.predict(test_dataset)

print("Accuracy of Random Forest classifier on test set is ", accuracy_score(test_dataset_Y,y_pred))
from sklearn.ensemble import GradientBoostingClassifier

from sklearn import cross_validation, metrics   

from sklearn.grid_search import GridSearchCV   

param_test1 = {'n_estimators':[20,30,40,50,60,70,80]}

gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=400,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10),param_grid = param_test1, n_jobs=4,iid=False, cv=5)

gsearch1.fit(train_dataset_X,train_dataset_Y)

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
param_test2 = {'max_depth':[5,7,9], 'min_samples_split':[200,400,600]}

gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80, max_features='sqrt', subsample=0.8, random_state=10), param_grid = param_test2,n_jobs=4,iid=False, cv=5)

gsearch2.fit(train_dataset_X,train_dataset_Y)

gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
gb = GridSearchCV(cv=5, error_score='raise',
       estimator=GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=9,
              max_features='sqrt', max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=50, subsample=0.8, tol=0.0001, validation_fraction=0.1,
              verbose=0, warm_start=False),
       fit_params={}, iid=False, n_jobs=4,
       param_grid={'n_estimators': [80]},
       pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=0)

gb.fit(train_dataset_X,train_dataset_Y)

y_pred = gb.predict(test_dataset)

print('Accuracy of Gradient Boosting classifier on test set: {:.2f}'.format(classifier.score(test_dataset, test_dataset_Y)))
from sklearn.externals import joblib

joblib.dump(clf,"classification.pkl")