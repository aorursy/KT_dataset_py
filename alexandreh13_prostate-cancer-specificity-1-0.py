import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("../input/prostate-cancer/Prostate_Cancer.csv")
data.shape
data.tail()
data.dtypes
data.isnull().sum()
data_model = data.drop(['id'], axis=1)

data_model['diagnosis_result'] = data_model['diagnosis_result'].astype('category')

data_model['diagnosis_result'] = data_model['diagnosis_result'].cat.codes

data_model['diagnosis_result'].dtype
data_model.tail()
correlations = data_model.corr(method='pearson')

correlations
plt.figure(figsize = (20, 8))

sb.heatmap(correlations, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.8)
sb.swarmplot(x=data_model['diagnosis_result'],

              y=data_model['perimeter'])
sb.swarmplot(x=data_model['diagnosis_result'],

              y=data_model['area'])
sb.swarmplot(x=data_model['diagnosis_result'],

              y=data_model['compactness'])
y = data_model.diagnosis_result

X = data_model[['perimeter', 'area', 'compactness']]
data_model['diagnosis_result'].value_counts()
data_model['diagnosis_result'].value_counts().plot(kind='bar', title='Count (target)')
from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.tree import DecisionTreeRegressor

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, KFold, train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

kf = KFold(n_splits=10, random_state=0, shuffle=True)
X_train.shape
X_test.shape
results_dict = {}
lr = LogisticRegression(C=0.5, random_state=1)

mean_auc_lr = cross_val_score(lr, X_train, y_train, n_jobs=-1, cv=kf, scoring='roc_auc').mean()

results_dict['Logistic Regression'] = mean_auc_lr

results_dict
svm = svm.SVC()

mean_auc_svm = cross_val_score(svm, X_train, y_train, n_jobs=-1, cv=kf, scoring='roc_auc').mean()

results_dict['SVM'] = mean_auc_svm

results_dict
dt = DecisionTreeRegressor()

mean_auc_dt = cross_val_score(dt, X_train, y_train, n_jobs=-1, cv=kf, scoring='roc_auc').mean()

results_dict['Decision Tree'] = mean_auc_dt

results_dict
nb = GaussianNB()

mean_auc_nb = cross_val_score(nb, X_train, y_train, n_jobs=-1, cv=kf, scoring='roc_auc').mean()

results_dict['NB'] = mean_auc_nb

results_dict
x = ['Logistic Regression', 'SVM', 'Decision Tree', 'NB']

y = [results_dict['Logistic Regression'], results_dict['SVM'], results_dict['Decision Tree'], results_dict['NB']]

plt.title("AUC comparison")

plt.ylabel("AUC")

plt.bar(x,y)
from sklearn.metrics import roc_auc_score

from sklearn.metrics import mean_absolute_error

nb.fit(X_train, y_train)

predicted = nb.predict(X_test)

roc_auc = roc_auc_score(y_test, predicted)

mae = mean_absolute_error(y_test, predicted)



print("Mean Absolute Error: {} | ROC AUC: {}".format(mae, roc_auc))
from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_test, predicted)

confusion
from sklearn.metrics import plot_confusion_matrix



disp = plot_confusion_matrix(nb, X_test, y_test,

                                 display_labels=data_model['diagnosis_result'],

                                 cmap=plt.cm.Blues)



disp.ax_.set_title("Confusion Matrix")

disp.confusion_matrix

plt.show()
TP = confusion[1, 1]

TN = confusion[0, 0]

FP = confusion[0, 1]

FN = confusion[1, 0]
sensitivity = TP/(TP+FN)

specificity = TN/(TN+FP)



"Sensitivity: {} | Specifictity: {}".format(sensitivity, specificity)