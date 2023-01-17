import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
## import the datasets
from sklearn.datasets import load_digits
dataset = load_digits()
dataset.data[0].shape
dataset.target[0]
dataset.images
plt.imshow(dataset.images[0])
dataset.images[0].shape
X,y = dataset.data,dataset.target
i=0

for item1,item2 in zip(X,y):

    plt.imshow(item1.reshape(8,8))

    plt.show()

    print(item2)

    i+=1

    if i==10:

        break
y_binary_imbalanced = y.copy()

y_binary_imbalanced[y_binary_imbalanced != 1] = 0
y_binary_imbalanced
## we can run a simple bin count

np.bincount(y_binary_imbalanced)
print('Original labels:\t', y[1:30])

print('New binary labels:\t', y_binary_imbalanced[1:30])
## train test split
X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)



# Accuracy of Support Vector Machine classifier

from sklearn.svm import SVC



svm = SVC(kernel='rbf', C=1).fit(X_train, y_train)

svm.score(X_test, y_test)
from sklearn.dummy import DummyClassifier



# Negative class (0) is most frequent

## so we classify with the most frequent

## you will see this classifier will show all the output is zero

## because most of it is negative and still it gives a good accuracy

dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)

# Therefore the dummy 'most_frequent' classifier always predicts class 0

y_dummy_predictions = dummy_majority.predict(X_test)



y_dummy_predictions
dummy_majority.score(X_test, y_test)
from sklearn.metrics import confusion_matrix



# Negative class (0) is most frequent

dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)

y_majority_predicted = dummy_majority.predict(X_test)

confusion = confusion_matrix(y_test, y_majority_predicted)



print('Most frequent class (dummy classifier)\n', confusion)
dummy_classprop = DummyClassifier(strategy='stratified').fit(X_train, y_train)

y_classprop_predicted = dummy_classprop.predict(X_test)

confusion = confusion_matrix(y_test, y_classprop_predicted)



print('Random class-proportional prediction (dummy classifier)\n', confusion)
svm = SVC(kernel='linear', C=1).fit(X_train, y_train)

svm_predicted = svm.predict(X_test)

confusion = confusion_matrix(y_test, svm_predicted)



print('Support vector machine classifier (linear kernel, C=1)\n', confusion)
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression().fit(X_train, y_train)

lr_predicted = lr.predict(X_test)

confusion = confusion_matrix(y_test, lr_predicted)



print('Logistic regression classifier (default settings)\n', confusion)
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)

tree_predicted = dt.predict(X_test)

confusion = confusion_matrix(y_test, tree_predicted)



print('Decision tree classifier (max_depth = 2)\n', confusion)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Accuracy = TP + TN / (TP + TN + FP + FN)

# Precision = TP / (TP + FP)

# Recall = TP / (TP + FN)  Also known as sensitivity, or True Positive Rate

# F1 = 2 * Precision * Recall / (Precision + Recall) 

print('Accuracy: {:.2f}'.format(accuracy_score(y_test, tree_predicted)))

print('Precision: {:.2f}'.format(precision_score(y_test, tree_predicted)))

print('Recall: {:.2f}'.format(recall_score(y_test, tree_predicted)))

print('F1: {:.2f}'.format(f1_score(y_test, tree_predicted)))
# Combined report with all above metrics

from sklearn.metrics import classification_report



print(classification_report(y_test, tree_predicted, target_names=['not 1', '1']))
print('Random class-proportional (dummy)\n', 

      classification_report(y_test, y_classprop_predicted, target_names=['not 1', '1']))

print('SVM\n', 

      classification_report(y_test, svm_predicted, target_names = ['not 1', '1']))

print('Logistic regression\n', 

      classification_report(y_test, lr_predicted, target_names = ['not 1', '1']))

print('Decision tree\n', 

      classification_report(y_test, tree_predicted, target_names = ['not 1', '1']))
%matplotlib notebook

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import datasets

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.dummy import DummyRegressor



diabetes = datasets.load_diabetes()



X = diabetes.data[:, None, 6]

y = diabetes.target



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



lm = LinearRegression().fit(X_train, y_train)
lm_dummy_mean = DummyRegressor(strategy = 'mean').fit(X_train, y_train)



y_predict = lm.predict(X_test)

y_predict_dummy_mean = lm_dummy_mean.predict(X_test)
print('Linear model, coefficients: ', lm.coef_)

print("Mean squared error (dummy): {:.2f}".format(mean_squared_error(y_test, 

                                                                     y_predict_dummy_mean)))

print("Mean squared error (linear model): {:.2f}".format(mean_squared_error(y_test, y_predict)))

print("r2_score (dummy): {:.2f}".format(r2_score(y_test, y_predict_dummy_mean)))

print("r2_score (linear model): {:.2f}".format(r2_score(y_test, y_predict)))

# Plot outputs

plt.scatter(X_test, y_test,  color='black')

plt.plot(X_test, y_predict, color='green', linewidth=2)

plt.plot(X_test, y_predict_dummy_mean, color='red', linestyle = 'dashed', 

         linewidth=2, label = 'dummy')



plt.show()
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC



dataset = load_digits()

# again, making this a binary problem with 'digit 1' as positive class 

# and 'not 1' as negative class

X, y = dataset.data, dataset.target == 1

clf = SVC(kernel='linear', C=1)



# accuracy is the default scoring metric

print('Cross-validation (accuracy)', cross_val_score(clf, X, y, cv=5))

# use AUC as scoring metric

print('Cross-validation (AUC)', cross_val_score(clf, X, y, cv=5, scoring = 'roc_auc'))

# use recall as scoring metric

print('Cross-validation (recall)', cross_val_score(clf, X, y, cv=5, scoring = 'recall'))
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score



dataset = load_digits()

X, y = dataset.data, dataset.target == 1

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



clf = SVC(kernel='rbf')

grid_values = {'gamma': [0.001, 0.01, 0.05, 0.1, 1, 10, 100]}



# default metric to optimize over grid parameters: accuracy

grid_clf_acc = GridSearchCV(clf, param_grid = grid_values)

grid_clf_acc.fit(X_train, y_train)

y_decision_fn_scores_acc = grid_clf_acc.decision_function(X_test) 



print('Grid best parameter (max. accuracy): ', grid_clf_acc.best_params_)

print('Grid best score (accuracy): ', grid_clf_acc.best_score_)



# alternative metric to optimize over grid parameters: AUC

grid_clf_auc = GridSearchCV(clf, param_grid = grid_values, scoring = 'roc_auc')

grid_clf_auc.fit(X_train, y_train)

y_decision_fn_scores_auc = grid_clf_auc.decision_function(X_test) 



print('Test set AUC: ', roc_auc_score(y_test, y_decision_fn_scores_auc))

print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)

print('Grid best score (AUC): ', grid_clf_auc.best_score_)

from sklearn.metrics.scorer import SCORERS



print(sorted(list(SCORERS.keys())))