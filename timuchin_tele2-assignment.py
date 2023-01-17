# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/mushrooms.csv") # Load data
df # Take a look
# Check unique values 
for col in df.columns:
    print(df[col].unique())
df = df.drop('veil-type', axis = 1) # Drop veil-type as it is a constant 
df_encoded = pd.get_dummies(df) # One-hot encoding
df_encoded
# Check encoded dataset
for col in df_encoded.columns:
    print(df_encoded[col].unique())
df_encoded.describe() # Everything seems fine
# Split the data into independent and explanatory variables 
y = df_encoded.iloc[:, 0]
X = df_encoded.iloc[:, 2:118]
X.head()
y.head()
# Split datasets into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.linear_model.logistic import LogisticRegression
logitclassifier = LogisticRegression()
logitclassifier.fit(X_train, y_train)
predictions = logitclassifier.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, predictions)
print(confusion_matrix)
import matplotlib.pyplot as plt
plt.matshow(confusion_matrix)
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
from sklearn.model_selection import cross_val_score
# Standard classification performance metrics
accuracy = cross_val_score(logitclassifier, X_train, y_train) # Accuracy measures a fraction of the classifier's predictions that are correct
precisions = cross_val_score(logitclassifier, X_train, y_train, scoring = 'precision') # Precision is the fraction of positive predictions that are correct
recalls = cross_val_score(logitclassifier, X_train, y_train, scoring = 'recall') # Recall is the fraction of the truly positive instances that the classifier recognizes
f1s = cross_val_score(logitclassifier, X_train, y_train, scoring = 'f1') # Harmonic mean of the precision and recall
print('Accuracy', np.mean(accuracy), accuracy)
print('Precision', np.mean(precisions), precisions)
print('Recalls', np.mean(recalls), recalls)
print('F1', np.mean(f1s), f1s)
from sklearn.metrics import roc_curve, auc
predictions = logitclassifier.predict_proba(X_test)
false_positive_rate, recall, thresholds = roc_curve(y_test, predictions[:, 1])
# ROC curve is insensitive to data sets with unbalanced class proportions; unlike precision and recall, 
# the ROC curve illustrates the classifier's performance for all values of the discrimination threshold
roc_auc = auc(false_positive_rate, recall)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()
# Tuning the model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
parameters = {
    'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty' : ['l1', 'l2']
}
logit = GridSearchCV(logitclassifier, parameters, n_jobs = -1, iid = 'True')
logit.fit(X_train, y_train)
print ('Best score: %0.3f' % logit.best_score_)
for param_name in sorted(parameters.keys()):
    print ('\t%s: %r' % (param_name, logit.best_params_[param_name]))
predictions = logit.predict(X_test)
print (classification_report(y_test, predictions))
from sklearn.tree import DecisionTreeClassifier
treeclassifier = DecisionTreeClassifier(criterion = 'entropy')
parameters = {
    'max_depth' : [150, 155, 160],
    'min_samples_leaf' : [1, 2, 3]
}
# Tuning the tree
tree = GridSearchCV(treeclassifier, parameters, n_jobs = -1, scoring = 'f1')
tree.fit(X_train, y_train)
print ('Best score: %0.3f' % tree.best_score_)
print ('Best parameter set:')
best_parameters = tree.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print ('\t%s: %r' % (param_name, best_parameters[param_name]))
predictions = tree.predict(X_test)
print (classification_report(y_test, predictions))
from sklearn.ensemble import RandomForestClassifier
# A random forest is a collection of decision trees that have been trained on randomly selected subsets
# of the training instances and explanatory variables.
forestclassifier = RandomForestClassifier(criterion = 'entropy')
parameters = {
    'n_estimators' : [5, 10, 20, 50],
    'max_depth' : [50, 150, 250],
    'min_samples_leaf' : [1, 2, 3]
}
# Tuning random forest
forest = GridSearchCV(forestclassifier, parameters, n_jobs = -1, scoring = 'f1')
forest.fit(X_train, y_train)
print ('Best score: %0.3f' % forest.best_score_)
print ('Best parameter set:')
best_parameters = forest.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print ('\t%s: %r' % (param_name, best_parameters[param_name]))
predictions = forest.predict(X_test)
print (classification_report(y_test, predictions))
from sklearn.svm import SVC
# A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane.
svmclassifier = SVC()
parameters = {
    'C' : [0.1, 0.5, 1, 2, 5, 10, 50, 100],
    'kernel' : ['rbf', 'linear', 'poly', 'sigmoid']
}
svm = GridSearchCV(svmclassifier, parameters, n_jobs = -1, scoring = 'f1')
svm.fit(X_train, y_train)
print ('Best score: %0.3f' % svm.best_score_)
print ('Best parameter set:')
best_parameters = svm.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print ('\t%s: %r' % (param_name, best_parameters[param_name]))
predictions = svm.predict(X_test)
print (classification_report(y_test, predictions))
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
adaboost = AdaBoostClassifier(random_state = 1)
gradientboost = GradientBoostingClassifier()
parameters = {
    'learning_rate': [0.1, 0.5, 1, 2, 5, 10],
    'n_estimators': [5, 10, 20, 50, 100, 150],
}
# Tuning AdaBoost
Adaboost = GridSearchCV(adaboost, parameters, n_jobs = -1, iid = 'True')
Adaboost.fit(X_train, y_train)
print ('Best score: %0.3f' % Adaboost.best_score_)
print ('Best parameter set:')
best_parameters = Adaboost.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print ('\t%s: %r' % (param_name, best_parameters[param_name]))
predictions = Adaboost.predict(X_test)
print (classification_report(y_test, predictions))
parameters = {
    'learning_rate' : [0.1, 0.5, 1, 2, 5, 10],
    'n_estimators' : [5, 10, 20, 50, 100, 150],
    'max_depth' : [1, 2, 5, 10],
    'min_samples_leaf' : [1, 2, 3]
}
# Tuning GradientBoost
GradientBoosting = GridSearchCV(gradientboost, parameters, n_jobs = -1, iid = 'True')
GradientBoosting.fit(X_train, y_train)
print ('Best score: %0.3f' % GradientBoosting.best_score_)
print ('Best parameter set:')
best_parameters = GradientBoosting.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print ('\t%s: %r' % (param_name, best_parameters[param_name]))
predictions = GradientBoosting.predict(X_test)
print (classification_report(y_test, predictions))
from sklearn.ensemble import VotingClassifier
# Use previously defined best parameters for each model
eclf = VotingClassifier(estimators = [('lr', LogisticRegression(C = 10, penalty = 'l1')), 
                                      ('rf', RandomForestClassifier(max_depth = 150, min_samples_leaf = 1)), 
                                      ('SVM', SVC(C = 0.5, kernel = 'linear')), 
                                      ('AB', AdaBoostClassifier(learning_rate = 0.5, n_estimators = 50)), 
                                      ('GB', GradientBoostingClassifier(learning_rate = 0.1, max_depth = 5, min_samples_leaf = 1, n_estimators = 50))], 
                        voting = 'hard', 
                        n_jobs = -1)
eclf.fit(X_train, y_train)
predictions = eclf.predict(X_test)
print (classification_report(y_test, predictions))