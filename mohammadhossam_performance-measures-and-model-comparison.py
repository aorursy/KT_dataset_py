import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(dataset.DESCR)
from sklearn.svm import SVC

clf = SVC(kernel = 'linear').fit(X_train, y_train)
clf.score(X_test, y_test)
from sklearn.metrics import confusion_matrix

y_predicted = clf.predict(X_test)
confusion_matrix(y_test, y_predicted)
from sklearn.metrics import precision_score, recall_score, f1_score

#We will calculate the previously discussed metrics for the model we created before
#precision_score, recall_score and f1_score are methods that takes the original labels and the predicted labels 
#as parameters and return the desired score
print('Precision score is {:.3f}'.format(precision_score(y_test, y_predicted)))
print('Recall score is {:.3f}'.format(recall_score(y_test, y_predicted)))
print('F1-score is {:.3f}'.format(f1_score(y_test, y_predicted)))
#desicion_function
y_scores = clf.decision_function(X_test)
y_scores_list = list(zip(y_test, y_scores))
y_scores_list[:20]
from sklearn.metrics import precision_recall_curve

precision, recall, threshold = precision_recall_curve(y_test, y_scores)
plt.figure(figsize = (8,6), dpi = 120)
plt.plot(precision, recall)
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision-recall curve')
plt.show()
#obviously, the best threshold is the curve is near to the top right corner as both precision and recall are high, unless
#we want to maximize one of them at the expense of the other
from sklearn.metrics import roc_curve

#like precision-recall curve, roc_curve method returns 3 parameters: fpr (false positives rate), tpr (true positives rate)
#and thresholds. We will explain the first 2 in detail.

fpr, tpr, thresholds = roc_curve(y_test, y_scores)
plt.figure(dpi = 130)
plt.plot(fpr, tpr)
plt.xlabel('False Positives Rate')
plt.ylabel('True Positives Rate')
plt.title('ROC curve')
plt.plot([0,1], [0,1], color= 'r', linestyle = '--')
plt.show()
plt.figure(dpi = 120, figsize = (10, 2.5))
positive = [19, 20, 23, 24, 26, 27, 28, 29, 30]
plt.plot(positive, [0] * 9, 'o', markersize = 10, label = 'Positive')
plt.plot(list(set(range(1,31)) - set(positive)), [0] * 21, 'o', markersize = 10, label = 'Negative')
plt.hlines(0, 1, 30)
plt.xlabel('Decision Function Score')
plt.legend(title = 'Original Labels')
plt.title('Hypothetical Dataset')
plt.gca().axes.get_yaxis().set_visible(False)
scores = np.array(list(range(1, 31)))
labels = np.zeros(31)
labels[positive] = 1
fpr, tpr, _ = roc_curve(labels[1:], scores)
plt.figure(dpi = 130)
plt.plot(fpr, tpr)
plt.xlabel('False Positives Rate')
plt.ylabel('True Positives Rate')
plt.title('ROC curve for nearly perfect model')
plt.show()
plt.figure(dpi = 120, figsize = (10, 2.5))
plt.plot(list(set(range(1,31)) - set(positive)), [0] * 21, 'o', markersize = 10, label = 'Positive')
plt.plot(positive, [0] * 9, 'o', markersize = 10, label = 'Negative')
plt.hlines(0, 1, 30)
plt.xlabel('Decision Function Score')
plt.legend(title = 'Original Labels')
plt.title('Hypothetical Dataset')
tplt.gca().axes.get_yaxis().set_visible(False)
scores = np.array(list(range(1, 31)))
labels = np.ones(31)
labels[positive] = 0
fpr, tpr, _ = roc_curve(labels[1:], scores)
plt.figure(dpi = 130)
plt.plot(fpr, tpr)
plt.xlabel('False Positives Rate')
plt.ylabel('True Positives Rate')
plt.title('ROC curve for a bad model')
plt.show()
random_positive = [1, 2, 4, 7,8,9, 14, 20, 22, 23, 24, 27, 29, 30]
plt.figure(dpi = 120, figsize = (10, 2.5))
plt.plot(random_positive, [0] * len(random_positive), 'o', markersize = 10, label = 'Positive')
plt.plot(list(set(range(1, 31)) - set(random_positive)), [0] * (30 - len(random_positive)),
         'o', markersize = 10, label = 'Negative')
plt.hlines(0, 1, 30)
plt.xlabel('Decision Function Score')
plt.legend(title = 'Original Labels')
plt.title('Hypothetical Dataset')
plt.gca().axes.get_yaxis().set_visible(False)
scores = np.array(list(range(1, 31)))
labels = np.zeros(31)
labels[random_positive] = 1
fpr, tpr, _ = roc_curve(labels[1:], scores)
plt.figure(dpi = 130)
plt.plot(fpr, tpr)
plt.xlabel('False Positives Rate')
plt.ylabel('True Positives Rate')
plt.title('ROC curve for a random model')
plt.show()
from sklearn.metrics import auc

#We will calculate AUC for the original model we were using
y_predicted = clf.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, y_predicted)
#auc method takes false positives rates and true positives rates as parameters
auc_score = auc(fpr, tpr)
print(auc_score)
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

dataset = load_diabetes()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
linreg = LinearRegression().fit(X_train, y_train)
y_predicted = linreg.predict(X_test)
print('R2 score is {:.3f}'.format(r2_score(y_test, y_predicted)))
print('Mean squared error is {:.3f}'.format(mean_squared_error(y_test, y_predicted)))
from sklearn.model_selection import GridSearchCV

dataset = load_breast_cancer()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

param_list = {'gamma' : [0.01, 0.1, 1, 10], 'C': [0.1, 1, 10, 100]}
model = GridSearchCV(SVC(kernel = 'rbf'), param_grid = param_list, scoring = 'recall')
model.fit(X_train, y_train)
#GridSearchCV has attributes called best_params_ and best_score_ that stores the values of parameters that produced
#the maximum desired score and the score itself respectively
print('The best parameters are: ', model.best_params_)
print('The best recall score is {:.3f}'.format(model.best_score_))