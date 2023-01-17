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
train_df = pd.read_csv('/kaggle/input/hacked/train.csv')
test_df = pd.read_csv('/kaggle/input/hacked/test.csv')
len(train_df),len(test_df)
#as we have too many rows so some number of drops may not affect 
train_df = train_df.dropna()
# train_df.isnull().any().count()#gives total count of null values
# train_df.isnull().any().sum()#gives total count of columns that contains null values

#as we have too many rows so some number of drops may not affect 
test_df.isnull().any().count()
train = train_df.drop(['MULTIPLE_OFFENSE', 'DATE', 'INCIDENT_ID'], axis=1)
# test = test_df.drop(['DATE', 'INCIDENT_ID'], axis=1)
train_target = train_df['MULTIPLE_OFFENSE']
incident_ids_train = train_df['INCIDENT_ID']
incident_ids_test = test_df[['INCIDENT_ID']]
test = test_df.drop(['DATE', 'INCIDENT_ID'], axis=1)
test = test.fillna(0)
# test.isnull().any()
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
train, train_target = oversample.fit_resample(train, train_target)
len(train),len(train_target)
# import seaborn as sns
# import matplotlib.pyplot  as plt
# tra = pd.concat([train,train_target],axis=1)
# corrmat = tra.corr()
# k = 16 #number of variables for heatmap
# cols = corrmat.nlargest(k, 'MULTIPLE_OFFENSE')['MULTIPLE_OFFENSE'].index
# cm = np.corrcoef(tra[cols].values.T)
# sns.set(font_scale=1)
# plt.figure(figsize=(10,10))
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# plt.show()
# # Create correlation matrix
# corr_matrix = train.corr().abs()

# # Select upper triangle of correlation matrix
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# # Find index of feature columns with correlation greater than 0.95
# to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# print(to_drop)
# # # Drop Marked Features
# # Drop features 
# train = train.drop(train[to_drop], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, train_target, test_size=0.2) 
## Standarization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
test = pd.DataFrame(scaler.transform(test), columns=test.columns)
X_train_copied = X_train.copy() 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# Find the optimal number of PCA 
pca = PCA(n_components=X_train_copied.shape[1])
pca.fit(X_train_copied)
ratios = pca.explained_variance_ratio_

# Plot the explained variance ratios
x = np.arange(X_train_copied.shape[1])
plt.plot(x, np.cumsum(ratios), '-o')
plt.xlabel("Number of PCA's")
plt.ylabel("Cumulated Sum of Explained Variance")
plt.title("Variance Explained by PCA's")

# Find the optimal number of PCA's
for i in range(np.cumsum(ratios).shape[0]):
    if np.cumsum(ratios)[i] >= 0.99:
        num_pca = i + 1
        print(f"The optimal number of PCA's is: {num_pca}")
        break
    else:
        continue

pca = PCA(n_components=num_pca)
pca.fit(X_train)
pca.transform(X_train)
pca.transform(X_test)
pca.transform(test)
from sklearn.linear_model import LogisticRegression
# Create logistic regression
lrn = LogisticRegression()
# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)
param_grid = [
  {'penalty': 'l1', 'solver': ['liblinear', 'saga']},
  {'penalty': 'l2', 'solver': ['lbfgs','newton-cg', 'sag']},
 ]

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=['l1'],solver =  [  'liblinear', 'saga'])
hyperparameters
from sklearn.model_selection import GridSearchCV
# Create grid search using 5-fold cross validation
clf = GridSearchCV(lrn, hyperparameters, cv=5, verbose=0)
# Fit grid search
best_model = clf.fit(X_train, y_train)
# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
print('Best solver:', best_model.best_estimator_.get_params()['solver'])
y_train_pred = best_model.predict(X_train)
from sklearn.metrics import accuracy_score
print('Accuracy over train dataset',accuracy_score(y_train, y_train_pred))
y_test_pred = best_model.predict(X_test)
print('Accuracy over validation dataset',accuracy_score(y_test, y_test_pred))
testing_pred = best_model.predict(test)
testing_pred_df = pd.DataFrame(testing_pred)
submission = pd.concat([incident_ids_test,testing_pred_df],axis=1)
submission
sub = submission.rename(columns={0:'MULTIPLE_OFFENSE'})
sub.to_csv("submitted_logistic_regression.csv",index=False,header=True)
from sklearn import svm
svm_model = svm.SVC()
svm_model.fit(X_train,y_train)
y_train_svm_predicted = svm_model.predict(X_train)
print('Accuracy of SVM over train data',accuracy_score(y_train,y_train_svm_predicted))

y_test_svm_predicted = svm_model.predict(X_test)
print('Accuracy of SVM over validation data',accuracy_score(y_test,y_test_svm_predicted))

tested_svm = svm_model.predict(test)
testing_svm_df= pd.DataFrame(tested_svm)
submission = pd.concat([incident_ids_test,testing_svm_df],axis=1)
sub = submission.rename(columns={0:'MULTIPLE_OFFENSE'})
sub.to_csv("submitted_svm_without_grid.csv",index=False,header=True)
from sklearn import svm
from sklearn.model_selection import GridSearchCV
svm_grid = svm.SVC()

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(
        svm_grid, tuned_parameters, scoring='%s_macro' % score
    )
    best_svm_model = clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(best_svm_model.best_params_)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_train_dt_predicted = dt.predict(X_train)
acc_score_train_dt = accuracy_score(y_train,y_train_dt_predicted)
print('Accuracy of Decision Tree over train data',acc_score_train_dt)

y_test_dt_predicted = dt.predict(X_test)
acc_score_test_dt = accuracy_score(y_test,y_test_dt_predicted)
print('Accuracy of Decision Tree over validation data',acc_score_test_dt)
tested_dt = dt.predict(test)
testing_dt_df= pd.DataFrame(tested_dt)
submission = pd.concat([incident_ids_test,testing_dt_df],axis=1)
sub = submission.rename(columns={0:'MULTIPLE_OFFENSE'})
sub.to_csv("submitted_pred_decision_tree_normal.csv",index=False,header=True)
# Create Parameter Space

# Create lists of parameter for Decision Tree Classifier
criterion = ['gini', 'entropy']
max_depth = [4,6,8,10,12]
dt = DecisionTreeClassifier()
# Create a dictionary of all the parameter options 
# Note has you can access the parameters of steps of a pipeline by using '__’
parameters = dict(decisiontree__criterion=criterion,decisiontree__max_depth=max_depth)

# Conduct Parameter Optmization With Pipeline
# Create a grid search object
clf = GridSearchCV(dt,parameters)

# Fit the grid search
clf.fit(X, y)

# View The Best Parameters
print('Best Criterion:', clf.best_estimator_.get_params()['decisiontree__criterion'])
print('Best max_depth:', clf.best_estimator_.get_params()['decisiontree__max_depth'])
#print('Best Number Of Components:', clf.best_estimator_.get_params()['pca__n_components'])
print(); print(clf.best_estimator_.get_params()['decisiontree'])
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
#Create a Gaussian Classifier
rf = RandomForestClassifier(n_estimators=100)
#Train the model using the training sets y_pred=clf.predict(X_test)
rf.fit(X_train,y_train)
y_train_rf_predicted = rf.predict(X_train)
acc_score_train_rf = accuracy_score(y_train,y_train_rf_predicted)
print('Accuracy of Random Forest over train data',acc_score_train_rf)

y_test_rf_predicted = rf.predict(X_test)
acc_score_test_rf = accuracy_score(y_test,y_test_rf_predicted)
print('Accuracy of Random Forest over validation data',acc_score_test_rf)

tested_rf = rf.predict(test)
testing_rf_df= pd.DataFrame(tested_rf)
submission = pd.concat([incident_ids_test,testing_rf_df],axis=1)
sub = submission.rename(columns={0:'MULTIPLE_OFFENSE'})
sub.to_csv("submitted_pred_random_forest_normal.csv",index=False,header=True)
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_train)
acc_score_train_xgb = accuracy_score(y_train,y_pred_xgb)
print('Accuracy of xgboost over train data',acc_score_train_xgb)
y_test_xgb_predicted = dt.predict(X_test)
acc_score_test_xgb = accuracy_score(y_test,y_test_xgb_predicted)
print('Accuracy of xgboost over test data',acc_score_test_xgb)
from sklearn.ensemble import AdaBoostClassifier
# Create adaboost classifer object
ada = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
# Train Adaboost Classifer
ada_model = ada.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_train)
acc_score_train_ada = accuracy_score(y_train,y_pred_ada)
print('Accuracy of Adaboost over train data',acc_score_train_ada)

y_test_ada_predicted = dt.predict(X_test)
acc_score_test_ada = accuracy_score(y_test,y_test_ada_predicted)
print('Accuracy of Adaboost over train data',acc_score_test_ada)

tested_ada = ada_model.predict(test)
testing_ada_df= pd.DataFrame(tested_ada)
submission = pd.concat([incident_ids_test,testing_ada_df],axis=1)
sub = submission.rename(columns={0:'MULTIPLE_OFFENSE'})
sub.to_csv("submitted_pred_adaboost_normal.csv",index=False,header=True)
from sklearn.model_selection import RandomizedSearchCV
param_dist = {
 'n_estimators': [10,50, 100],
 'learning_rate' : [0.01,0.05,0.1,0.3,1],
 }
seed = np.random.seed(22)  #set seed because at every iteration it was giving different best value
#link : https://stackoverflow.com/questions/41516150/randomizedsearchcv-gives-different-results-using-the-same-random-state
# if base estimator is None, then the base estimator is DecisionTreeClassifier(max_depth=1)
base_estimator = DecisionTreeClassifier(max_depth=None)
ada = AdaBoostClassifier(base_estimator=base_estimator)
ada_rand = RandomizedSearchCV(ada,param_distributions = param_dist,cv=10)
ada_rand.fit(X_train, y_train)
# View The Best Parameters
print('Best no of estimators :', ada_rand.best_estimator_.get_params()['n_estimators'])
print('Best learning rate  :', ada_rand.best_estimator_.get_params()['learning_rate'])
base_estimator = DecisionTreeClassifier(max_depth=None)
ada = AdaBoostClassifier(base_estimator=base_estimator,n_estimators=10,learning_rate = 0.3)
# Train Adaboost Classifer
ada_model = ada.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_train)
acc_score_train_ada = accuracy_score(y_train,y_pred_ada)
print(acc_score_train_ada)
y_test_ada_predicted = dt.predict(X_test)
acc_score_test_ada = accuracy_score(y_test,y_test_ada_predicted)
print(acc_score_test_ada)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              'learning_rate' : [0.01,0.03,0.05,0.06,0.1,0.3,1],
              "n_estimators": [10,15,25, 50,100]
             }

DTC = DecisionTreeClassifier(random_state = 11,max_depth = None)
ABC = AdaBoostClassifier(base_estimator = DTC)
# run grid search
grid_search_ABC = GridSearchCV(ABC, param_grid=param_grid)
grid_search_ABC.fit(X_train,y_train)
# View The Best Parameters
print('Best criterion:', grid_search_ABC.best_estimator_.get_params()['base_estimator__criterion'])
print('Best splitter  :', grid_search_ABC.best_estimator_.get_params()['base_estimator__splitter'])
print('Best no. of estimator :', grid_search_ABC.best_estimator_.get_params()['n_estimators'])
print('Best learning rate :', grid_search_ABC.best_estimator_.get_params()['learning_rate'])
y_pred_ada = grid_search_ABC.predict(X_train)
acc_score_train_ada = accuracy_score(y_train,y_pred_ada)
print('Accuracy of Adaboost with grid search over train data',acc_score_train_ada)

y_test_ada_predicted = grid_search_ABC.predict(X_test)
acc_score_test_ada = accuracy_score(y_test,y_test_ada_predicted)
print('Accuracy of Adaboost with grid search over train data',acc_score_test_ada)

tested_ada = grid_search_ABC.predict(test)
testing_ada_df= pd.DataFrame(tested_ada)
submission = pd.concat([incident_ids_test,testing_ada_df],axis=1)
sub = submission.rename(columns={0:'MULTIPLE_OFFENSE'})
sub.to_csv("submitted_adaboost_with_grid_search_best_score.csv",index=False,header=True)
#ROC is a probability curve and AUC represents degree or measure of separability.
#It tells how much model is capable of distinguishing between classes.
#Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s.