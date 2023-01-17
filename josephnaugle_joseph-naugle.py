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
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn
import theano as thno
import theano.tensor as T
import warnings

from collections import OrderedDict
from scipy.optimize import fmin_powell
from scipy import integrate
from time import time
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
# designate target variable name
targetName = 'Survived'
#print(targetName)
targetSeries = train_data[targetName] #notice one column is considered a series in pandas
#print(targetSeries)
#remove target from current location and insert in column number 0
del train_data[targetName]
train_data.insert(0, targetName, targetSeries)
#reprint dataframe and see target is in position 0
train_data.head(10)
all = pd.concat([train_data, test_data], sort = False)
all.info()
g = seaborn.pairplot(all)
corr_matrix = all.iloc[:,2:].corr()
corr_matrix

#code chunk is attributed as follows:
#Shaw, Chris (2019). How to customize seaborn correlation heat map. Medium. 
#Retrieved from https://medium.com/@chrisshaw982/seaborn-correlation-heatmaps-customized-10246f4f7f4b.

import seaborn as sns

plt.figure(figsize=(15,15)) 
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr_matrix,
           vmin=-1,
           vmax=1,
           cmap='coolwarm',
           annot=True,
           mask=mask)
plt.show()
all.dtypes
all.shape
all.describe
print("Column Names", all.columns)
groupby = all.groupby(targetName)
targetEDA=groupby[targetName].aggregate(len)
print(targetEDA)
plt.figure()
targetEDA.plot(kind='bar', grid=False)
plt.axhline(0, color='k')
all['Age'] = all['Age'].fillna(value=all['Age'].median())
all['Fare'] = all['Fare'].fillna(value=all['Fare'].median())
all['Embarked'] = all['Embarked'].fillna('S')
all.loc[ all['Age'] <= 16, 'Age'] = 0
all.loc[(all['Age'] > 16) & (all['Age'] <= 32), 'Age'] = 1
all.loc[(all['Age'] > 32) & (all['Age'] <= 48), 'Age'] = 2
all.loc[(all['Age'] > 48) & (all['Age'] <= 64), 'Age'] = 3
all.loc[ all['Age'] > 64, 'Age'] = 4 
import re
def get_title(name):
    title_search = re.search(' ([A-Za-z]+\.)', name)
    
    if title_search:
        return title_search.group(1)
    return ""
all['Title'] = all['Name'].apply(get_title)
all['Title'] = all['Title'].replace(['Capt.', 'Dr.', 'Major.', 'Rev.'], 'Officer.')
all['Title'] = all['Title'].replace(['Lady.', 'Countess.', 'Don.', 'Sir.', 'Jonkheer.', 'Dona.'], 'Royal.')
all['Title'] = all['Title'].replace(['Mlle.', 'Ms.'], 'Miss.')
all['Title'] = all['Title'].replace(['Mme.'], 'Mrs.')
all['Cabin'] = all['Cabin'].fillna('Missing')
all['Cabin'] = all['Cabin'].str[0]
all['Family_Size'] = all['SibSp'] + all['Parch'] + 1
all['IsAlone'] = 0
all.loc[all['Family_Size']==1, 'IsAlone'] = 1
all_1 = all.drop(['Name', 'Ticket'], axis = 1)
all_dummies = pd.get_dummies(all_1, drop_first = True)
all_dummies.head()
all_train = all_dummies[all_dummies['Survived'].notna()]
all_test = all_dummies[all_dummies['Survived'].isna()]
#train_data = train_data.dropna(axis=1)
#missing_val_count_by_column = (train_data.isnull().sum())
#print(missing_val_count_by_column[missing_val_count_by_column > 0])
#test_data = test_data.dropna(axis=1)
#missing_val_count_by_column = (test_data.isnull().sum())
#print(missing_val_count_by_column[missing_val_count_by_column > 0])
#Add packages
#These are my standard packages I load for almost every project
%matplotlib inline 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#From Scikit Learn
from sklearn import preprocessing
from sklearn.model_selection  import train_test_split, cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(all_train.drop(['PassengerId','Survived'],axis=1), 
                                                    all_train['Survived'], test_size=0.30, 
                                                    random_state=101, stratify = all_train['Survived'])
#decision tree. Call up my model and name it clf
#clf is a notation used by many people for classifier
from sklearn import tree 
clf_dt = tree.DecisionTreeClassifier()
#Call up the model to see the parameters you can tune (and their default setting)
print(clf_dt)
features_train= X_train
target_train= y_train
features_test= X_test
target_test= y_test
np.size(target_train)
clf_dt = clf_dt.fit(features_train, target_train)
target_predicted_dt = clf_dt.predict(features_test)
print("DT Accuracy Score", accuracy_score(target_test, target_predicted_dt))
print(classification_report(target_test, target_predicted_dt))
print(confusion_matrix(target_test, target_predicted_dt))

#extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(target_test, target_predicted_dt).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)
scores = cross_val_score(clf_dt, features_train, target_train, cv=10)
print("Cross Validation Score for each K",scores)
scores.mean()
#Build
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler
knn = KNeighborsClassifier()
#call up the model to see the parameters you can tune (and their default settings)
print(knn)
print(scaler)

from sklearn.pipeline import make_pipeline
clf_knn = make_pipeline(StandardScaler(), KNeighborsClassifier())
clf_knn = clf_knn.fit(features_train, target_train)
target_predicted_knn = clf_knn.predict(features_test)
print("KNN Accuracy Score", accuracy_score(target_test, target_predicted_knn))
print(classification_report(target_test, target_predicted_knn))
print(confusion_matrix(target_test, target_predicted_knn))

#extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(target_test, target_predicted_knn).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)
#load packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os as os
from sklearn.metrics import mean_squared_error
%matplotlib inline 
import sys
from sklearn.linear_model import LinearRegression

#Step 1: Build a Model
clf_lr = LinearRegression(normalize=True)
clf_lr
clf_lr.fit(features_train, target_train)
predicted = clf_lr.predict(features_test)
# summarize the fit of the model
print("Coef", clf_lr.intercept_, clf_lr.coef_)
print("MSE", mean_squared_error(target_test, predicted))
# Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Step 1: Build
clf_rg = make_pipeline(StandardScaler(), Ridge(alpha=1)) #alpha is lambda (the tuning parameter)!
print(clf_rg) #use this to see all the steps in your pipeline
# Step 2: Train 
clf_rg.fit(features_train, target_train)

# Step 3: Validate
predicted_rg= clf_rg.predict(features_test)

# summarize the fit of the model
# Since I use a pipeline, I have to call back up the 'ridge' part of the pipeline to get access to the fitted model
print("Coef", clf_rg['ridge'].intercept_, clf_rg['ridge'].coef_)
# A student commented that it is not possible to print the line above without adding in the named_steps method. 
#You may want to try the following line if the one above does not work.
#print("Coef", clf_rg.named_steps['ridge'].intercept_, clf_rg.named_steps['ridge'].coef_)

print("MSE", mean_squared_error(target_test, predicted_rg)) #notice the slightly higher MSE compared to OLS
# use a full grid over several parameters and cross validate 5 times
from sklearn.model_selection import GridSearchCV
#param_grid = {"alpha": [2,10, 20, 50, 100, 500]}
#param_grid={"alpha": [1,10,1]} #this does a range 1 through 10 changes by a factor of 1. 
param_grid={"alpha": [.01,1,.05]} #this does a range .01 through 1 changes by a factor of .05

# run grid search
grid_search = GridSearchCV(clf_rg['ridge'], param_grid=param_grid,n_jobs=-1,cv=5)
#Look at the documentation for GridSearchCV
#n_jobs = how many jobs to run in parallel. Default is 1. -1 means run all jobs at the same time.
grid_search.fit(features_train, target_train)
print("Grid Scores", grid_search.cv_results_)
print("Best", grid_search.best_params_)
# Lasso Regression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Step 1: Build
clf_ls = make_pipeline(StandardScaler(), Lasso(alpha=2)) #alpha is lambda (the tuning parameter)!
print(clf_ls) #use this to see all the steps in your pipeline
# Step 2: Train 
clf_ls.fit(features_train, target_train)

# Step 3: Validate
predicted_ls= clf_ls.predict(features_test)

# summarize the fit of the model
# Since I use a pipeline, I have to call back up the 'ridge' part of the pipeline to get access to the fitted model
print("Coef", clf_ls['lasso'].intercept_, clf_ls['lasso'].coef_)
print("MSE", mean_squared_error(target_test, predicted_ls)) #notice the slightly higher MSE compared to OLS
# ElasticNet Regression
from sklearn.linear_model import ElasticNet

# Step 1: Build
clf_en = make_pipeline(StandardScaler(), ElasticNet(alpha=2)) #alpha is lambda (the tuning parameter)!
print(clf_en) #use this to see all the steps in your pipeline
# Step 2: Train 
clf_en.fit(features_train, target_train)

# Step 3: Validate
predicted_en= clf_en.predict(features_test)

# summarize the fit of the model
# Since I use a pipeline, I have to call back up the 'elasticnet' part of the pipeline to get access to the fitted model
print("Coef", clf_en['elasticnet'].intercept_, clf_en['elasticnet'].coef_)
print("MSE", mean_squared_error(target_test, predicted_en)) #notice the slightly higher MSE compared to OLS
#load packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os as os
%matplotlib inline 
import sys

#basic methods from scikit learn package
from sklearn.model_selection  import train_test_split, cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
#Build
clf_linearSVC = make_pipeline(StandardScaler(), SVC(kernel='linear', class_weight='balanced'))
print(clf_linearSVC)
clf_linearSVC.fit(features_train, target_train)
#Validate
target_predicted = clf_linearSVC.predict(features_test)
print(classification_report(target_test, target_predicted,target_names=['0','1'])) #no=0; yes=1
print(confusion_matrix(target_test, target_predicted))
#extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(target_test, target_predicted).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)

print("Accuracy Score", accuracy_score(target_test, target_predicted))
TestForPred = all_test.drop(['PassengerId', 'Survived'], axis = 1)
t_pred = clf_linearSVC.predict(TestForPred).astype(int)
PassengerId = all_test['PassengerId']
output = pd.DataFrame({'PassengerId': PassengerId, 'Survived': t_pred})
output.to_csv('my_submission4.csv', index=False)
output.head(5)
import time
start = time.time()
param_grid={"C": [.01,0.2]} #trying out two different C values.
clf_linearSVC = SVC(kernel='linear', class_weight='balanced')
grid_svm = GridSearchCV(clf_linearSVC, param_grid,n_jobs=-1, cv=5)
grid_svm.fit(features_train, target_train)
print("SCORES", grid_svm.cv_results_)
print("BEST SCORE", grid_svm.best_score_)
print("BEST PARAM", grid_svm.best_params_)
end = time.time()
print("Time to run", round(end-start), "seconds")
#code chunk is from Raschka & Mirjalili (2019). Python Machine Learning. 3rd ed. p. 207.

start=time.time()
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'svc__C': param_range, 
               'svc__kernel': ['linear']},
              {'svc__C': param_range, 
               'svc__gamma': param_range, 
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  refit=True,
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(features_train, target_train)
print(gs.best_score_)
print(gs.best_params_)
end=time.time()
print("Time to run", round(start-end), "seconds")
#load packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os as os
%matplotlib inline 
import sys

#basic methods from scikit learn package
from sklearn.model_selection  import train_test_split, cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import tree 
#Build
clf_dt = tree.DecisionTreeClassifier(class_weight="balanced", random_state=123)
print(clf_dt)
#Train
clf_dt = clf_dt.fit(features_train, target_train)
#Validate
target_predicted = clf_dt.predict(features_test)
print("DT Accuracy Score", accuracy_score(target_test, target_predicted))
# print classification report
target_names = ["Survived = no", "Surrvived = yes"]
print(classification_report(target_test, target_predicted, target_names=target_names))
#verify DT with Cross Validation
scores = cross_val_score(clf_dt, features_train, target_train, cv=10)
print("Cross Validation Score for each K",scores)
scores.mean()
from sklearn.ensemble import BaggingClassifier
#Build
clf_bag = BaggingClassifier(n_estimators=100, random_state=123)
print(clf_bag)
#Train
clf_bag.fit(features_train, target_train)
#Validate 
target_predicted=clf_bag.predict(features_test)
print("Bagging Accuracy", accuracy_score(target_test, target_predicted))
target_names = ["Survived = no", "Survived = yes"]
print(classification_report(target_test, target_predicted,target_names=target_names))
print(confusion_matrix(target_test, target_predicted))
from sklearn.ensemble import RandomForestClassifier
#Build
clf_rf = RandomForestClassifier(class_weight='balanced', max_features='auto', n_estimators=100, random_state=123)
#max_features = 'auto' for default. Calculated as square root of number of features.
print(clf_rf)
#Train
clf_rf.fit(features_train, target_train)
#Validate
target_predicted = clf_rf.predict(features_test)
print("Random Forfest Accuracy", accuracy_score(target_test, target_predicted))
target_names = ["Survived = no", "Survived = yes"]
print(classification_report(target_test, target_predicted, target_names=target_names))
print(confusion_matrix(target_test, target_predicted))
from sklearn.ensemble import ExtraTreesClassifier
clf_xdt = ExtraTreesClassifier(n_estimators= 100, n_jobs=-1,class_weight="balanced", random_state=123)
print(clf_xdt)
clf_xdt.fit(features_train, target_train)
target_predicted=clf_xdt.predict(features_test)
print("Extra Trees Accuracy", accuracy_score(target_test,target_predicted))
target_names = ["Survived = no", "Survived = yes"]
print(classification_report(target_test, target_predicted,target_names=target_names))
print(confusion_matrix(target_test, target_predicted))
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
clf_dt_ab = AdaBoostClassifier(tree.DecisionTreeClassifier(class_weight="balanced"),
                         algorithm="SAMME.R",
                         n_estimators=100, random_state=123)
clf_dt_ab.fit(features_train, target_train)
target_predicted=clf_dt_ab.predict(features_test)
print("Adaboost Accuracy", accuracy_score(target_test,target_predicted))
target_names = ["Survived = no", "Survived = yes"]
print(classification_report(target_test, target_predicted,target_names=target_names))
print(confusion_matrix(target_test, target_predicted))
from sklearn.ensemble import GradientBoostingClassifier
clf_GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=123) #default learning rate is 0.1
clf_GBC.fit(features_train, target_train)
target_predicted=clf_GBC.predict(features_test)
print("Gradient Boost Accuracy", accuracy_score(target_test,target_predicted))
target_names = ["Survived = no", "Survived = yes"]
print(classification_report(target_predicted, target_test, target_names=target_names))
print(confusion_matrix(target_predicted, target_test))
np.size(target_train)

#load packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os as os
%matplotlib inline 
import sys

#basic methods from scikit learn package
from sklearn.model_selection  import train_test_split, cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler  
#Build
scaler = StandardScaler()  
#Train
scaler.fit(features_train)  
#Validate
features_train_norm = scaler.transform(features_train)  
# apply same transformation to test data
features_test_norm = scaler.transform(features_test)
#Build
from sklearn.neural_network import MLPClassifier
clf_nn = MLPClassifier(hidden_layer_sizes=(5),random_state=0) #one hidden layer with five units/nodes
#clf_nn = MLPClassifier(hidden_layer_sizes=(5), solver="lbfgs",random_state=0) #one hidden layer with five units/nodes
#clf_nn = MLPClassifier(hidden_layer_sizes=(5), solver="sgd", learning_rate="adaptive", max_iter=1000,random_state=0)
#clf_nn = MLPClassifier(hidden_layer_sizes=(5,5), solver="sgd", learning_rate="adaptive", max_iter=1000,random_state=0)
#clf_nn = MLPClassifier(hidden_layer_sizes=(20,20), solver="sgd", learning_rate="adaptive", max_iter=10000,random_state=0)
#Train
clf_nn.fit(features_train_norm, target_train)
#Validate
target_predicted = clf_nn.predict(features_test_norm)
print("Accuracy", accuracy_score(target_test, target_predicted))
target_names = ["Survived = no", "Survived = yes"]
print(classification_report(target_test, target_predicted, target_names=target_names))
print(confusion_matrix(target_test, target_predicted))
#test_data= test_data.iloc[0:357,:]
TestForPred = all_test.drop(['PassengerId', 'Survived'], axis = 1)
t_pred = clf_nn.predict(TestForPred).astype(int)
PassengerId = all_test['PassengerId']
output = pd.DataFrame({'PassengerId': PassengerId, 'Survived': t_pred})
output.to_csv('my_submission.csv', index=False)
output.head(5)
#Build
from sklearn.linear_model import SGDClassifier
clf_sgd_linear_svm = SGDClassifier(random_state=0)
#Train
clf_sgd_linear_svm.fit(features_train_norm, target_train)
#Validate
target_predicted = clf_sgd_linear_svm.predict(features_test_norm)
print("Accuracy", accuracy_score(target_test, target_predicted))
target_names = ["Churn = no", "Churn = yes"]
print(classification_report(target_test, target_predicted, target_names=target_names))
print(confusion_matrix(target_test, target_predicted))
#Build
from sklearn.linear_model import LogisticRegression
import time
start = time.time()
clf_lr_logit = LogisticRegression(solver = 'lbfgs', random_state=0) #default solver
#Train
clf_lr_logit.fit(features_train_norm, target_train)
end = time.time()
print("Training time is", end-start)
#Validate
target_predicted = clf_lr_logit.predict(features_test_norm)
print("Accuracy", accuracy_score(target_test, target_predicted))
target_names = ["Churn = no", "Churn = yes"]
print(classification_report(target_test, target_predicted, target_names=target_names))
print(confusion_matrix(target_test, target_predicted))
#Build
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
#clf_nn = make_pipeline(MinMaxScaler(), MLPRegressor(hidden_layer_sizes=5, max_iter=10000, random_state=0))
clf_nn = make_pipeline(MinMaxScaler(), MLPRegressor(hidden_layer_sizes=(20,20), max_iter=10000, random_state=0))
#Train
clf_nn.fit(features_train, target_train)
#Validate
target_predicted = clf_nn.predict(features_test)
#print(target_predicted)
#print(target_test.head)
print("MSE", mean_squared_error(target_test, target_predicted))
TestForPred = all_test.drop(['PassengerId', 'Survived'], axis = 1)
t_pred = clf_nn.predict(TestForPred).astype(int)
PassengerId = all_test['PassengerId']
output = pd.DataFrame({'PassengerId': PassengerId, 'Survived': t_pred})
output.to_csv('my_submission3.csv', index=False)
output.head(5)